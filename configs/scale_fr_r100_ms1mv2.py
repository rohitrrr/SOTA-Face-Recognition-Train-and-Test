"""
SCaLE-FR R100 + MS1MV2
======================
Parameters optimized for IResNet-100 on MS1MV2 (5.8M images, 85K classes)
with RTX 3060 12GB.

Training schedule:
  Phase 1 (epochs 0-5):   Pure ArcFace warmup, queue filling
  Phase 2 (epoch 6+):     Fisher projector activates, losses ramp in
  Phase 3 (epochs 11+):   Full SCaLE-FR (ramp complete)
  Phase 4 (epoch 26):     End

Run:
    torchrun --nproc_per_node=1 train_scale_fr.py \
        --config_file ./configs/scale_fr_r100_ms1mv2.py
"""
from configs.base import config as base_config
from easydict import EasyDict

config = EasyDict(base_config)

# ─── Identity ────────────────────────────────────────────────────────────────
config.prefix = "scale_fr-r100-ms1mv2"
config.head = "arcface"

# ─── Backbone ────────────────────────────────────────────────────────────────
config.depth = "100"

# ─── Dataset ─────────────────────────────────────────────────────────────────
config.train_source = "./dataset/ms1mv2.lmdb"
config.num_ims = 5822653

# ─── Training (tuned for R100 + RTX 3060 12GB) ──────────────────────────────
config.batch_size = 64
config.lr = 0.05
config.epochs = 26
config.scheduler = True
config.warmup_epoch = 1

# ─── ArcFace ─────────────────────────────────────────────────────────────────
config.margin = 0.5

# ─── Validation (skip during training to avoid OOM) ──────────────────────────
config.val_list = []
config.val_source = "./test_set_package_5"

# ─── SCaLE-FR ────────────────────────────────────────────────────────────────
config.scale_fr = EasyDict()

# Queue: 16K entries, ~19% of 85K classes visible per fill cycle
config.scale_fr.queue_size          = 16384
config.scale_fr.momentum            = 0.999

# Fisher projector: refresh every 500 steps (more frequent for smaller batch)
config.scale_fr.proj_dim            = 256
config.scale_fr.proj_refresh_steps  = 500
config.scale_fr.proj_ema_alpha      = 0.9
config.scale_fr.proj_epsilon        = 1e-4
config.scale_fr.max_classes_for_cov = 500
config.scale_fr.max_samples_per_class = 20

# Loss weights
config.scale_fr.lambda_tail         = 0.3
config.scale_fr.lambda_pos          = 0.2
config.scale_fr.tail_margin         = 0.15

# Fixed
config.scale_fr.beta                = 20.0
config.scale_fr.top_m               = 20
config.scale_fr.top_q               = 0.1
config.scale_fr.tau_p               = 0.5
config.scale_fr.ramp_steps          = 5000

# Schedule
config.scale_fr.warmup_epochs       = 5
config.scale_fr.activate_epoch      = 6

# Inference
config.scale_fr.use_projected_inference = True
