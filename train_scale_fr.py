"""
SCaLE-FR Training Script
========================
Extends the base SOTA training framework with:
  - Momentum encoder + 65K embedding queue
  - Fisher projector (refreshed every 1000 steps)
  - Tail ranking loss + hardest positive loss
  - Staged activation: warmup → projector → losses

Usage:
  torchrun --nproc_per_node=NUM_GPUS train_scale_fr.py \
      --config_file ./configs/scale_fr_r50.py

Phase 1 (warmup):    Pure ArcFace, queue filling, no SCaLE losses
Phase 2 (activate):  Projector online, losses ramp in linearly
Phase 3 (full):      Full SCaLE-FR training
"""

import argparse
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from torch import optim, distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import LMDBDataLoader, get_val_pair, setup_seed
from lr_scheduler import PolyScheduler
from model import iresnet, PartialFC_V2, get_vit
import verification
from utils import *
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

# SCaLE-FR components
from scale_fr.memory_bank import MomentumBank
from scale_fr.fisher_projector import FisherProjector
from scale_fr.losses import ScaleFRLoss


try:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")
except KeyError:
    rank = 0
    local_rank = 0
    world_size = 1
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:13584",
        rank=rank,
        world_size=world_size,
    )


class TrainScaleFR:
    def __init__(self, config):
        self.config = config
        self.sf = config.scale_fr  # shorthand for SCaLE-FR config

        if local_rank == 0:
            create_path(self.config.model_path)
            create_path(self.config.log_path)
            init_logging(self.config.work_path)

        torch.cuda.set_device(local_rank)

        # ─── Data ─────────────────────────────────────────────────────
        self.dataset = LMDBDataLoader(config=self.config, train=True)
        self.train_loader = self.dataset.get_loader()
        class_num = self.dataset.class_num()
        img_num = self.dataset.get_length()

        # ─── Backbone ─────────────────────────────────────────────────
        if self.config.model == "iresnet":
            self.backbone = iresnet(
                self.config.depth, fp16=self.config.fp16,
                mode=self.config.mode
            ).to(local_rank)
        elif self.config.model == "vit":
            self.backbone = get_vit(self.config.depth).to(local_rank)

        # ─── Classification head (ArcFace + PartialFC) ────────────────
        self.cls_head = self.config.recognition_head
        paras_only_bn, paras_wo_bn = separate_bn_param(self.backbone)

        # TensorBoard
        if rank == 0:
            self.writer = SummaryWriter(config.log_path)
            dummy_input = torch.zeros(1, 3, 112, 112).to(local_rank)
            self.writer.add_graph(self.backbone, dummy_input)
            for key, value in self.config.items():
                logging.info("%-25s %s", key, value)
        else:
            self.writer = None

        # DDP wrapper
        self.backbone = torch.nn.parallel.DistributedDataParallel(
            module=self.backbone, broadcast_buffers=False,
            device_ids=[local_rank], bucket_cap_mb=16
        )
        self.backbone.register_comm_hook(None, fp16_compress_hook)
        self.backbone._set_static_graph()

        # PartialFC for classification loss
        self.cls_head = PartialFC_V2(
            self.cls_head, self.config.embedding_size, class_num,
            self.config.sample_rate, self.config.fp16
        ).to(local_rank)

        # ─── SCaLE-FR: Momentum Bank (deferred to save RAM) ─────────
        # Created lazily when warmup ends — the momentum encoder is a
        # full copy of the backbone and doubles RAM usage.
        self.memory_bank = None  # initialized in _init_memory_bank()

        # ─── SCaLE-FR: Fisher Projector ───────────────────────────────
        self.fisher = FisherProjector(
            embedding_dim=self.config.embedding_size,
            proj_dim=self.sf.proj_dim,
            cov_ema_alpha=self.sf.proj_ema_alpha,
            epsilon=self.sf.proj_epsilon,
            max_classes_for_cov=self.sf.max_classes_for_cov,
            max_samples_per_class=self.sf.max_samples_per_class,
        ).to(local_rank)

        # ─── SCaLE-FR: Losses ─────────────────────────────────────────
        self.scale_loss = ScaleFRLoss(
            lambda_tail=self.sf.lambda_tail,
            lambda_pos=self.sf.lambda_pos,
            tail_margin=self.sf.tail_margin,
            tau_p=self.sf.tau_p,
            top_m=self.sf.top_m,
            top_q=self.sf.top_q,
            beta=self.sf.beta,
            ramp_steps=self.sf.ramp_steps,
        ).to(local_rank)

        # ─── Optimizer ────────────────────────────────────────────────
        if self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(
                [
                    {"params": paras_wo_bn,
                     "weight_decay": self.config.weight_decay},
                    {"params": self.cls_head.parameters(),
                     "weight_decay": self.config.weight_decay},
                    {"params": paras_only_bn},
                ],
                lr=self.config.lr,
                momentum=self.config.momentum,
            )
        elif self.config.optimizer == "adamw":
            self.optimizer = optim.AdamW(
                params=[
                    {"params": self.backbone.parameters()},
                    {"params": self.cls_head.parameters()},
                ],
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )

        # ─── LR Scheduler ────────────────────────────────────────────
        total_batch = self.config.batch_size * world_size
        if self.config.scheduler:
            warmup_step = img_num // total_batch * self.config.warmup_epoch
            total_step = img_num // total_batch * self.config.epochs
            self.lr_scheduler = PolyScheduler(
                optimizer=self.optimizer,
                base_lr=self.config.lr,
                max_steps=total_step,
                warmup_steps=warmup_step,
                last_epoch=-1
            )

        # ─── Validation (lazy-loaded to save RAM) ─────────────────────
        # On 16GB systems, preloading all val sets at init causes OOM.
        # Load each set on-demand during evaluation, then free.
        self.validation_names = list(config.val_list)

        self.train_logger = TrainLogger(
            total_batch,
            self.config.frequency_log,
            self.dataset.get_length() // total_batch * self.config.epochs,
            self.config.epochs,
            self.writer
        )

        self.save_file(self.config, "config.txt")
        self.best_acc = -1
        self.best_step = 0

        # ─── Step tracking ────────────────────────────────────────────
        self.global_step = 0
        self.start_epoch = 0
        self.steps_per_epoch = img_num // total_batch
        self.scale_fr_activated = False
        self.memory_bank_initialized = False

    def _init_memory_bank(self):
        """Lazily create momentum encoder + queue when warmup ends."""
        if self.memory_bank_initialized:
            return
        if rank == 0:
            logging.info("[SCaLE-FR] Initializing momentum bank...")
        self.memory_bank = MomentumBank(
            backbone=self.backbone.module,
            queue_size=self.sf.queue_size,
            embedding_dim=self.config.embedding_size,
            momentum=self.sf.momentum,
        ).to(local_rank)
        self.memory_bank_initialized = True
        if rank == 0:
            logging.info("[SCaLE-FR] Momentum bank ready.")

    # ─── CHECKPOINT: save everything needed to resume ─────────────────
    def save_checkpoint(self, epoch):
        """Save full training state for crash recovery."""
        if local_rank != 0:
            return

        ckpt_path = os.path.join(self.config.model_path, "checkpoint.pt")
        ckpt_tmp = ckpt_path + ".tmp"

        state = {
            # Core model
            'backbone': self.backbone.module.state_dict(),
            'cls_head': self.cls_head.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            # LR scheduler
            'lr_scheduler': self.lr_scheduler.state_dict()
                if self.config.scheduler else None,
            # Training progress (batch-size-independent)
            'epoch': epoch + 1,  # next epoch to run
            'global_step': self.global_step,
            'best_acc': self.best_acc,
            'best_step': self.best_step,
            'batch_size': self.config.batch_size,  # for cross-device resume
            # SCaLE-FR components
            'fisher': self.fisher.state_dict(),
            'scale_loss': self.scale_loss.state_dict(),
            'memory_bank_extra': (self.memory_bank.state_dict_extra()
                                  if self.memory_bank is not None else None),
            'momentum_encoder': (self.memory_bank.encoder_m.state_dict()
                                 if self.memory_bank is not None else None),
            'scale_fr_activated': self.scale_fr_activated,
            'memory_bank_initialized': self.memory_bank_initialized,
        }

        # Write to tmp first, then atomic rename (crash-safe)
        torch.save(state, ckpt_tmp)
        os.replace(ckpt_tmp, ckpt_path)
        logging.info(f"[Checkpoint] Saved at epoch {epoch}, "
                     f"step {self.global_step}")

    def load_checkpoint(self, resume_path=None):
        """Load full training state to resume after crash.

        Handles cross-device resume (different batch sizes):
        - Recalculates global_step based on epoch, not raw step count
        - Rebuilds LR scheduler at correct position for current batch size
        - Loads model weights and optimizer momentum (these are batch-size independent)
        """
        if resume_path is None:
            resume_path = os.path.join(
                self.config.model_path, "checkpoint.pt")

        if not os.path.exists(resume_path):
            if rank == 0:
                logging.info("[Resume] No checkpoint found, starting fresh.")
            return False

        if rank == 0:
            logging.info(f"[Resume] Loading checkpoint from {resume_path}")

        state = torch.load(resume_path, map_location='cpu')

        # Core model (batch-size independent)
        self.backbone.module.load_state_dict(state['backbone'])
        self.cls_head.load_state_dict(state['cls_head'])

        # Detect batch size mismatch
        ckpt_batch_size = state.get('batch_size', self.config.batch_size)
        batch_size_changed = (ckpt_batch_size != self.config.batch_size)

        if batch_size_changed and rank == 0:
            logging.info(
                f"[Resume] Batch size changed: {ckpt_batch_size} → "
                f"{self.config.batch_size}. Recalculating LR schedule.")

        # Training progress: use epoch as the ground truth, recompute step
        self.start_epoch = state['epoch']
        self.best_acc = state.get('best_acc', -1)
        self.best_step = state.get('best_step', 0)

        if batch_size_changed:
            # Recompute global_step for current batch size
            self.global_step = self.start_epoch * self.steps_per_epoch

            # Rebuild LR scheduler from scratch at correct position
            # Do NOT load old scheduler state — it's for wrong step count
            if self.config.scheduler:
                for _ in range(self.global_step):
                    self.lr_scheduler.step()

            # Load optimizer state but skip LR-related parts
            # (momentum buffers are still valid across batch sizes)
            self.optimizer.load_state_dict(state['optimizer'])
            for opt_state in self.optimizer.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(local_rank)
            # Override LR with scheduler's current value
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.lr_scheduler.get_last_lr()[0]
        else:
            # Same batch size: restore everything exactly
            self.global_step = state['global_step']
            self.optimizer.load_state_dict(state['optimizer'])
            for opt_state in self.optimizer.state.values():
                for k, v in opt_state.items():
                    if isinstance(v, torch.Tensor):
                        opt_state[k] = v.to(local_rank)
            if state['lr_scheduler'] is not None and self.config.scheduler:
                self.lr_scheduler.load_state_dict(state['lr_scheduler'])

        self.best_acc = state.get('best_acc', -1)
        self.best_step = state.get('best_step', 0)

        # SCaLE-FR components
        self.fisher.load_state_dict(state['fisher'])
        self.scale_loss.load_state_dict(state['scale_loss'])
        self.scale_fr_activated = state['scale_fr_activated']
        self.memory_bank_initialized = state.get(
            'memory_bank_initialized', False)

        # Restore memory bank if it was initialized before checkpoint
        if self.memory_bank_initialized and state['memory_bank_extra'] is not None:
            self._init_memory_bank()
            self.memory_bank.load_state_dict_extra(state['memory_bank_extra'])
            if state['momentum_encoder'] is not None:
                self.memory_bank.encoder_m.load_state_dict(
                    state['momentum_encoder'])

        if rank == 0:
            logging.info(
                f"[Resume] Restored: epoch={self.start_epoch}, "
                f"step={self.global_step}, best_acc={self.best_acc:.5f}, "
                f"scale_active={self.scale_fr_activated}")

        return True

    def run(self):
        self.backbone.train()
        self.cls_head.train()
        loss_am = AverageMeter()
        scale_loss_am = AverageMeter()
        amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

        for epoch in range(self.start_epoch, self.config.epochs):
            if isinstance(self.train_loader, DataLoader):
                self.train_loader.sampler.set_epoch(epoch)
            if not self.config.scheduler and epoch + 1 in self.config.reduce_lr:
                self.reduce_lr()

            # ─── Init momentum bank when warmup is nearly done ─────────
            if (epoch >= self.sf.warmup_epochs - 1 and
                    not self.memory_bank_initialized):
                self._init_memory_bank()

            for idx, data in enumerate(self.train_loader):
                imgs, labels = data
                self.global_step += 1

                # ═══ FORWARD: backbone ════════════════════════════════
                embeddings = self.backbone(imgs)

                # ═══ FORWARD: classification loss (ArcFace) ═══════════
                loss_cls = self.cls_head(embeddings, labels)

                # ═══ SCaLE-FR: enqueue momentum embeddings ════════════
                if self.memory_bank is not None:
                    with torch.no_grad():
                        self.memory_bank.update_momentum_encoder(
                            self.backbone.module)
                        self.memory_bank.encode_and_enqueue(imgs, labels)

                # ═══ SCaLE-FR: refresh Fisher projector ═══════════════
                if (self.memory_bank is not None and
                        self.global_step % self.sf.proj_refresh_steps == 0 and
                        self.global_step > self.steps_per_epoch *
                        self.sf.warmup_epochs):
                    q_emb, q_lbl = self.memory_bank.get_queue()
                    if q_emb is not None:
                        refresh_diag = self.fisher.refresh(q_emb, q_lbl)
                        if rank == 0 and refresh_diag is not None:
                            logging.info(
                                f"[Fisher refresh] step={self.global_step} "
                                f"classes_used={refresh_diag['n_classes_used']} "
                                f"BW_ratio={refresh_diag['BW_ratio']:.4f} "
                                f"eig_max={refresh_diag['fisher_eig_max']:.4f}")
                            if self.writer:
                                for k, v in refresh_diag.items():
                                    if isinstance(v, (int, float)):
                                        self.writer.add_scalar(
                                            f"fisher/{k}", v,
                                            self.global_step)

                # ═══ SCaLE-FR: activate when ready (checked every step) ═
                if (epoch >= self.sf.activate_epoch and
                        not self.scale_fr_activated and
                        self.fisher.initialized):
                    self.scale_loss.activate()
                    self.scale_fr_activated = True
                    if rank == 0:
                        logging.info(
                            f"[SCaLE-FR] Activated at epoch {epoch}, "
                            f"step {self.global_step}")

                # ═══ SCaLE-FR: compute tail + positive losses ═════════
                loss_scale = torch.tensor(0.0, device=imgs.device)
                scale_diag = {}

                if (self.scale_fr_activated and self.fisher.initialized
                        and self.memory_bank is not None):
                    # Get queue
                    q_emb, q_lbl = self.memory_bank.get_queue()

                    if q_emb is not None and q_emb.shape[0] > 100:
                        # Project online embeddings (gradient flows)
                        emb_norm = F.normalize(embeddings.float(), dim=1)
                        proj_online = self.fisher.project(emb_norm)

                        # Project queue embeddings (no gradient)
                        with torch.no_grad():
                            proj_queue = self.fisher.project(q_emb)

                        # SCaLE-FR loss
                        loss_scale, scale_diag = self.scale_loss(
                            proj_online, labels,
                            proj_queue, q_lbl,
                            fisher_projector=self.fisher
                        )

                # ═══ TOTAL LOSS ═══════════════════════════════════════
                loss = loss_cls + loss_scale

                # ═══ BACKWARD + OPTIMIZE ══════════════════════════════
                if self.config.fp16:
                    amp.scale(loss).backward()
                    amp.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.backbone.parameters(), 5)
                    amp.step(self.optimizer)
                    amp.update()
                    self.optimizer.zero_grad()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.backbone.parameters(), 5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.config.scheduler:
                    self.lr_scheduler.step()

                loss_am.update(loss_cls.item(), 1)
                if loss_scale.item() > 0:
                    scale_loss_am.update(loss_scale.item(), 1)

                # ═══ MID-EPOCH CHECKPOINT (every 5000 steps) ═════════
                if self.global_step % 5000 == 0:
                    self.save_checkpoint(epoch)

                # ═══ LOGGING ══════════════════════════════════════════
                self.train_logger(
                    self.global_step, epoch, loss_am, local_rank)

                # SCaLE-FR diagnostics (every 500 steps on rank 0)
                if (rank == 0 and self.writer is not None and
                        self.global_step % 500 == 0):
                    self.writer.add_scalar(
                        "loss/cls", loss_cls.item(), self.global_step)
                    self.writer.add_scalar(
                        "loss/scale_fr", loss_scale.item(), self.global_step)
                    self.writer.add_scalar(
                        "queue/filled",
                        self.memory_bank.get_queue_size_filled()
                            if self.memory_bank is not None else 0,
                        self.global_step)
                    self.writer.add_scalar(
                        "fisher/initialized",
                        float(self.fisher.initialized.item()),
                        self.global_step)

                    if scale_diag:
                        for k, v in scale_diag.items():
                            if isinstance(v, (int, float)):
                                self.writer.add_scalar(
                                    f"scale_fr/{k}", v, self.global_step)

            # ═══ END OF EPOCH ═════════════════════════════════════════
            # Save model weights for this epoch (lightweight, no validation)
            if local_rank == 0:
                epoch_model_path = os.path.join(
                    self.config.model_path,
                    f"backbone_epoch{epoch}_step{self.global_step}.pth")
                torch.save(self.backbone.module.state_dict(), epoch_model_path)
                logging.info(f"[Save] Model saved: {epoch_model_path}")

            # Save full checkpoint for crash recovery
            self.save_checkpoint(epoch)

            if rank == 0:
                q_filled = (self.memory_bank.get_queue_size_filled()
                            if self.memory_bank is not None else 0)
                logging.info(
                    f"[Epoch {epoch}] cls_loss={loss_am.avg:.4f} "
                    f"scale_loss={scale_loss_am.avg:.4f} "
                    f"queue={q_filled}/{self.sf.queue_size} "
                    f"fisher_init={self.fisher.initialized.item()} "
                    f"scale_active={self.scale_fr_activated}")
                loss_am.reset()
                scale_loss_am.reset()

    def save_model(self, step):
        if local_rank == 0:
            val_acc, _ = self.evaluate(step)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_step = step
            save_state(self.backbone, self.optimizer, self.config,
                       val_acc, step, head=self.cls_head)

            # Save SCaLE-FR components
            scale_path = os.path.join(
                self.config.model_path, f"scale_fr_step{step}.pt")
            torch.save({
                'fisher_state': self.fisher.state_dict(),
                'scale_loss_state': self.scale_loss.state_dict(),
                'memory_bank_extra': self.memory_bank.state_dict_extra(),
                'global_step': self.global_step,
                'scale_fr_activated': self.scale_fr_activated,
            }, scale_path)

            logging.info(
                f"Best accuracy: {self.best_acc:.5f} at step {self.best_step}")

    def reduce_lr(self):
        for params in self.optimizer.param_groups:
            params["lr"] /= 10

    def evaluate(self, step):
        if local_rank == 0:
            self.backbone.eval()
            # Free GPU cache before loading val data into RAM
            torch.cuda.empty_cache()
            import gc; gc.collect()

            val_acc = 0
            n_val = 0
            logging.info("Validating...")

            for val_name in self.validation_names:
                # Load on-demand (saves ~3GB RAM vs preloading)
                dataset, issame = get_val_pair(
                    self.config.val_source, val_name)

                if (self.sf.use_projected_inference and
                        self.fisher.initialized):
                    acc, std = self.evaluate_projected(dataset, issame)
                    acc_raw, std_raw = self.evaluate_recognition(
                        dataset, issame)
                    logging.info(
                        f"{val_name}: proj={acc:.5f}+-{std:.5f} "
                        f"raw={acc_raw:.5f}+-{std_raw:.5f}")
                    if self.writer:
                        self.writer.add_scalar(
                            f"{val_name} acc (projected)", acc, step)
                        self.writer.add_scalar(
                            f"{val_name} acc (raw)", acc_raw, step)
                else:
                    acc, std = self.evaluate_recognition(dataset, issame)
                    logging.info(f"{val_name}: {acc:.5f}+-{std:.5f}")
                    if self.writer:
                        self.writer.add_scalar(
                            f"{val_name} acc", acc, step)

                val_acc += acc
                n_val += 1

                # Free memory immediately after each val set
                del dataset, issame

            val_acc /= max(n_val, 1)
            if self.writer:
                self.writer.add_scalar("Mean acc", val_acc, step)
            logging.info(f"Mean accuracy: {val_acc:.5f}")
            self.backbone.train()
            return val_acc, 0
        return 0, 0

    def evaluate_recognition(self, samples, issame, nrof_folds=10):
        """Standard evaluation in raw embedding space."""
        embedding_length = len(samples) // 2
        embeddings = np.zeros(
            [embedding_length, self.config.embedding_size])

        with torch.no_grad():
            for idx in range(0, embedding_length, self.config.batch_size):
                batch_flip = torch.tensor(
                    samples[embedding_length + idx:
                            embedding_length + idx + self.config.batch_size])
                batch_or = torch.tensor(
                    samples[idx: idx + batch_flip.shape[0]])

                actual = batch_or.shape[0]
                if self.config.add_flip:
                    embeddings[idx:idx + actual] = \
                        self.backbone(batch_or.to(local_rank)).cpu() + \
                        self.backbone(batch_flip.to(local_rank)).cpu()
                else:
                    embeddings[idx:idx + actual] = \
                        self.backbone(batch_or.to(local_rank)).cpu()

        normalized = np.divide(
            embeddings, np.linalg.norm(embeddings, 2, 1, True))
        tpr, fpr, accuracy = verification.evaluate(
            normalized, issame, nrof_folds)
        return round(accuracy.mean(), 5), round(accuracy.std(), 5)

    def evaluate_projected(self, samples, issame, nrof_folds=10):
        """Evaluation in Fisher-projected space (train/test consistency)."""
        embedding_length = len(samples) // 2
        embeddings = np.zeros(
            [embedding_length, self.sf.proj_dim])

        with torch.no_grad():
            for idx in range(0, embedding_length, self.config.batch_size):
                batch_flip = torch.tensor(
                    samples[embedding_length + idx:
                            embedding_length + idx + self.config.batch_size])
                batch_or = torch.tensor(
                    samples[idx: idx + batch_flip.shape[0]])
                actual = batch_or.shape[0]

                # Get raw embeddings
                if self.config.add_flip:
                    raw = self.backbone(batch_or.to(local_rank)) + \
                          self.backbone(batch_flip.to(local_rank))
                else:
                    raw = self.backbone(batch_or.to(local_rank))

                # L2-normalize then project
                raw_norm = F.normalize(raw.float(), dim=1)
                proj = self.fisher.project(raw_norm)
                embeddings[idx:idx + actual] = proj.cpu().numpy()

        normalized = np.divide(
            embeddings, np.linalg.norm(embeddings, 2, 1, True))
        tpr, fpr, accuracy = verification.evaluate(
            normalized, issame, nrof_folds)
        return round(accuracy.mean(), 5), round(accuracy.std(), 5)

    def save_file(self, string, file_name):
        file = open(os.path.join(self.config.work_path, file_name), "w")
        file.write(str(string))
        file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train SCaLE-FR model.")
    parser.add_argument(
        "--config_file", "-config",
        help="path of config file.",
        default="./configs/scale_fr_r50.py", type=str)
    parser.add_argument(
        "--device", default='0', type=str,
        help='cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument(
        "--resume", default=None, type=str,
        help='path to checkpoint.pt to resume from, '
             'or "auto" to find latest in model_path')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    config = get_config(args.config_file)
    setup_seed(seed=42, cuda_deterministic=False)

    train = TrainScaleFR(config)

    # Resume from checkpoint if requested
    if args.resume:
        if args.resume == "auto":
            train.load_checkpoint()  # uses default path
        else:
            train.load_checkpoint(args.resume)

    train.run()
