"""
SCaLE-FR Loss Functions
=======================
Two losses operating in Fisher-projected identity subspace:

1. TailRankingLoss:
   For each anchor, enforce margin between hardest positive and
   smooth-max of top-M queue impostors.
   Uses CVaR (top-q% anchor selection) to focus on the tail.

2. HardestPositiveLoss:
   Ensure every identity's worst same-class sample is still
   close in projected space. Attacks the centroid-vs-sample gap.

Both losses take projected embeddings g(x) as input, not raw f(x).
This ensures training optimizes the same metric used at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TailRankingLoss(nn.Module):
    """
    Tail ranking loss in Fisher-projected space.

    For each anchor a:
      - Find hardest positive: p* = argmin_{p in same class} g(a)·g(p)
      - Find top-M impostors from queue: smooth-max of cosine similarities
      - Loss: softplus(n_tail - g(a)·g(p*) + margin)
      - Apply CVaR: only backprop through top-q% hardest anchors

    Fix from review: per-anchor M_i = min(top_m, n_valid_negatives_i)
    to avoid bias when some anchors have fewer valid impostors than top_m.

    Args:
        top_m: Number of top queue negatives per anchor. Default 20.
        top_q: Fraction of hardest anchors to backprop. Default 0.1 (10%).
        margin: Required gap between positive and impostor. Default 0.1.
        beta: Temperature for smooth-max over negatives. Default 20.0.
    """

    def __init__(self, top_m=20, top_q=0.1, margin=0.1, beta=20.0):
        super().__init__()
        self.top_m = top_m
        self.top_q = top_q
        self.margin = margin
        self.beta = beta

    def forward(self, proj_online, labels_online, proj_queue, labels_queue):
        """
        Args:
            proj_online: (B, k) projected online embeddings (GRADIENT flows).
            labels_online: (B,) class labels for online batch.
            proj_queue: (Q, k) projected queue embeddings (NO gradient).
            labels_queue: (Q,) class labels for queue.

        Returns:
            loss: scalar tail ranking loss.
            diagnostics: dict with monitoring info.
        """
        B = proj_online.shape[0]
        Q = proj_queue.shape[0]
        device = proj_online.device

        # ─── Find hardest positive per anchor ─────────────────────────
        sim_batch = proj_online @ proj_online.T  # (B, B)

        label_eq = labels_online.unsqueeze(0) == labels_online.unsqueeze(1)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask = label_eq & self_mask

        has_positive = pos_mask.any(dim=1)  # (B,)

        sim_pos = sim_batch.clone()
        sim_pos[~pos_mask] = float('inf')
        hardest_pos_sim, _ = sim_pos.min(dim=1)  # (B,)

        # ─── Find top-M impostors from queue (per-anchor M_i) ────────
        sim_queue = proj_online @ proj_queue.T  # (B, Q)

        # Mask same-class entries
        same_class_mask = (labels_online.unsqueeze(1) ==
                           labels_queue.unsqueeze(0))  # (B, Q)
        sim_queue_masked = sim_queue.clone()
        sim_queue_masked[same_class_mask] = float('-inf')

        # Count valid negatives per anchor
        n_valid_neg = (~same_class_mask).sum(dim=1)  # (B,)
        has_negatives = n_valid_neg >= 1

        # Per-anchor M_i = min(top_m, n_valid_neg_i)
        M_per_anchor = torch.clamp(n_valid_neg, max=self.top_m)  # (B,)

        # Take top-M scores (use global top_m for topk, then mask)
        M_global = min(self.top_m, Q)
        top_m_scores, _ = sim_queue_masked.topk(M_global, dim=1)  # (B, M_global)

        # Compute smooth-max with per-anchor normalization
        # n_tail_i = (1/β) · log( (1/M_i) · Σ_{j=1}^{M_i} exp(β · s_j) )
        # For anchors with M_i < M_global, the extra entries are -inf and
        # contribute ~0 to logsumexp, but we still normalize by M_i not M_global
        n_tail = (1.0 / self.beta) * torch.logsumexp(
            self.beta * top_m_scores, dim=1)  # (B,)
        # Subtract log(M_i) instead of log(M_global)
        log_M_i = torch.log(M_per_anchor.float().clamp(min=1.0))
        n_tail = n_tail - (1.0 / self.beta) * log_M_i

        # ─── Per-anchor loss ──────────────────────────────────────────
        per_anchor_loss = F.softplus(
            n_tail - hardest_pos_sim + self.margin)

        # Zero out anchors without valid positives or negatives
        valid_anchor = has_positive & has_negatives
        per_anchor_loss = per_anchor_loss * valid_anchor.float()

        # ─── CVaR: top-q% anchor selection ────────────────────────────
        n_valid = int(valid_anchor.sum().item())
        if n_valid < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'tail_loss': 0.0, 'n_valid_anchors': 0,
                'mean_hardest_pos': 0.0, 'mean_n_tail': 0.0,
                'margin_gap': 0.0}

        n_select = max(1, int(n_valid * self.top_q))
        sorted_loss, _ = per_anchor_loss.sort(descending=True)
        cvar_loss = sorted_loss[:n_select].mean()

        # ─── Diagnostics ──────────────────────────────────────────────
        with torch.no_grad():
            vm = valid_anchor
            diag = {
                'tail_loss': cvar_loss.item(),
                'n_valid_anchors': n_valid,
                'n_cvar_anchors': n_select,
                'mean_hardest_pos': hardest_pos_sim[vm].mean().item()
                    if n_valid > 0 else 0.0,
                'mean_n_tail': n_tail[vm].mean().item()
                    if n_valid > 0 else 0.0,
                'margin_gap': (hardest_pos_sim[vm] -
                               n_tail[vm]).mean().item()
                    if n_valid > 0 else 0.0,
                'mean_valid_neg_count': M_per_anchor[vm].float().mean().item()
                    if n_valid > 0 else 0.0,
            }

        return cvar_loss, diag


class HardestPositiveLoss(nn.Module):
    """
    Hardest-positive compactness loss using queue-based positive mining.

    For each anchor, find the hardest (least similar) same-class sample
    across BOTH the batch AND the queue, then penalize if below threshold τ_p.

    L_pos = (1/|V|) · Σ_{a ∈ V} max(0, τ_p - g(a)·g(p*(a)))

    Queue mining solves the core problem: with 85K classes and batch=64,
    same-class pairs appear in only ~2.3% of batches. The queue (16K entries,
    ~2-3K unique classes) provides positives for far more anchors.

    Gradient flows through proj_online (anchor side) only. Queue positives
    are detached — we pull the anchor toward where its class sits in the
    projected space, not the other way around.

    Args:
        tau_p: Minimum required cosine similarity for positives. Default 0.5.
    """

    def __init__(self, tau_p=0.5):
        super().__init__()
        self.tau_p = tau_p

    def forward(self, proj_online, labels_online,
                proj_queue=None, labels_queue=None):
        """
        Args:
            proj_online: (B, k) projected embeddings (gradient flows).
            labels_online: (B,) class labels for online batch.
            proj_queue: (Q, k) projected queue embeddings (no gradient). Optional.
            labels_queue: (Q,) queue labels. Optional.

        Returns:
            loss: scalar hardest-positive compactness loss.
            diagnostics: dict with monitoring info.
        """
        B = proj_online.shape[0]
        device = proj_online.device

        if B < 2:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'pos_loss': 0.0, 'n_valid_pos_anchors': 0,
                'mean_hardest_pos_sim': 0.0, 'frac_below_tau': 0.0}

        # ─── Batch positives (same as before) ────────────────────────
        sim_batch = proj_online @ proj_online.T  # (B, B)
        label_eq_batch = labels_online.unsqueeze(0) == labels_online.unsqueeze(1)
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        pos_mask_batch = label_eq_batch & self_mask

        sim_pos_batch = sim_batch.clone()
        sim_pos_batch[~pos_mask_batch] = float('inf')
        hardest_batch, _ = sim_pos_batch.min(dim=1)  # (B,)
        has_batch_pos = pos_mask_batch.any(dim=1)

        # ─── Queue positives (the key addition) ─────────────────────
        if proj_queue is not None and labels_queue is not None and proj_queue.shape[0] > 0:
            sim_queue = proj_online @ proj_queue.T  # (B, Q)
            label_eq_queue = labels_online.unsqueeze(1) == labels_queue.unsqueeze(0)  # (B, Q)

            sim_pos_queue = sim_queue.clone()
            sim_pos_queue[~label_eq_queue] = float('inf')
            hardest_queue, _ = sim_pos_queue.min(dim=1)  # (B,)
            has_queue_pos = label_eq_queue.any(dim=1)

            # Combine: take the harder (lower sim) of batch vs queue
            hardest_pos_sim = torch.minimum(hardest_batch, hardest_queue)
            has_positive = has_batch_pos | has_queue_pos
        else:
            hardest_pos_sim = hardest_batch
            has_positive = has_batch_pos

        # ─── Loss ────────────────────────────────────────────────────
        per_anchor_loss = F.relu(self.tau_p - hardest_pos_sim)
        per_anchor_loss = per_anchor_loss * has_positive.float()

        n_valid = int(has_positive.sum().item())
        loss = per_anchor_loss.sum() / max(n_valid, 1)

        with torch.no_grad():
            diag = {
                'pos_loss': loss.item(),
                'mean_hardest_pos_sim': hardest_pos_sim[has_positive].mean().item()
                    if n_valid > 0 else 0.0,
                'frac_below_tau': (
                    (hardest_pos_sim[has_positive] < self.tau_p)
                    .float().mean().item()
                    if n_valid > 0 else 0.0),
                'n_valid_pos_anchors': n_valid,
                'n_batch_pos': int(has_batch_pos.sum().item()),
                'n_queue_pos': int(has_positive.sum().item()) - int(has_batch_pos.sum().item()),
            }

        return loss, diag


class ScaleFRLoss(nn.Module):
    """
    Combined SCaLE-FR loss wrapper.

    L = L_cls + λ_tail · L_tail + λ_pos · L_pos

    Handles:
      - Staged activation (losses only active after warmup)
      - Linear ramp-in over ramp_steps
      - Guards against None/empty queue or uninitialized projector

    Args:
        lambda_tail: Weight for tail ranking loss. Default 0.3.
        lambda_pos: Weight for hardest positive loss. Default 0.3.
        tail_margin: Margin for tail ranking. Default 0.1.
        tau_p: Threshold for positive compactness. Default 0.5.
        top_m: Top negatives per anchor. Default 20.
        top_q: CVaR fraction. Default 0.1.
        beta: Smooth-max temperature. Default 20.0.
        ramp_steps: Number of steps to linearly ramp in losses. Default 5000.
    """

    def __init__(self, lambda_tail=0.3, lambda_pos=0.3,
                 tail_margin=0.1, tau_p=0.5,
                 top_m=20, top_q=0.1, beta=20.0,
                 ramp_steps=5000):
        super().__init__()
        self.lambda_tail = lambda_tail
        self.lambda_pos = lambda_pos
        self.ramp_steps = ramp_steps

        self.tail_loss = TailRankingLoss(
            top_m=top_m, top_q=top_q, margin=tail_margin, beta=beta)
        self.pos_loss = HardestPositiveLoss(tau_p=tau_p)

        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))
        self.register_buffer('active', torch.tensor(False, dtype=torch.bool))

    def activate(self):
        """Call this when warmup is done and projector is ready."""
        self.active.fill_(True)
        self.step_counter.zero_()

    def get_ramp_weight(self):
        """Linear ramp from 0 to 1 over ramp_steps."""
        if not self.active:
            return 0.0
        step = self.step_counter.item()
        return min(1.0, step / max(self.ramp_steps, 1))

    def forward(self, proj_online, labels_online, proj_queue, labels_queue,
                fisher_projector=None):
        """
        Compute combined SCaLE-FR loss.

        Args:
            proj_online: (B, k) projected online embeddings.
            labels_online: (B,) online labels.
            proj_queue: (Q, k) projected queue embeddings (can be None).
            labels_queue: (Q,) queue labels (can be None).
            fisher_projector: FisherProjector instance (for diagnostics).

        Returns:
            total_loss: scalar loss (0 if not active or inputs invalid).
            diagnostics: dict with all monitoring info.
        """
        device = proj_online.device
        diag = {}

        if not self.active:
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'scale_fr_active': False, 'ramp_weight': 0.0}

        # Guard: queue must be valid
        if (proj_queue is None or labels_queue is None or
                proj_queue.shape[0] < 100):
            return torch.tensor(0.0, device=device, requires_grad=True), {
                'scale_fr_active': True, 'ramp_weight': 0.0,
                'queue_too_small': True}

        ramp = self.get_ramp_weight()
        self.step_counter += 1

        # Tail ranking loss
        loss_tail, diag_tail = self.tail_loss(
            proj_online, labels_online, proj_queue, labels_queue)

        # Hardest positive loss (with queue-based mining)
        loss_pos, diag_pos = self.pos_loss(
            proj_online, labels_online, proj_queue, labels_queue)

        # Combined with ramp
        total = ramp * (self.lambda_tail * loss_tail +
                        self.lambda_pos * loss_pos)

        # Merge diagnostics
        diag.update(diag_tail)
        diag.update(diag_pos)
        diag['scale_fr_active'] = True
        diag['ramp_weight'] = ramp
        diag['total_scale_loss'] = total.item()
        diag['lambda_tail'] = self.lambda_tail
        diag['lambda_pos'] = self.lambda_pos

        if fisher_projector is not None:
            diag.update(fisher_projector.get_diagnostics())

        return total, diag
