import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupConLoss)

    Reference:
        - Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf
        - SimCLR (unsupervised contrastive loss): https://arxiv.org/abs/2002.05709

    Args:
        temperature (float): Temperature scaling factor.
        contrast_mode (str): 'one' for single view, 'all' for all views in batch as anchors.
        base_temperature (float): Scaling base temperature for loss.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features (Tensor): Shape [batch_size, n_views, feature_dim]
            labels (Tensor, optional): Shape [batch_size], class labels.
            mask (Tensor, optional): Shape [batch_size, batch_size], precomputed mask for positives.

        Returns:
            torch.Tensor: Contrastive loss (scalar).
        """
        device = features.device

        if features.ndim < 3:
            raise ValueError("Expected features with shape [bsz, n_views, ...]")

        batch_size = features.shape[0]
        n_views = features.shape[1]
        features = features.view(batch_size, n_views, -1)  # Flatten features

        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")

        # Create contrastive mask
        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, dim]

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = n_views
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # Numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask for positive pairs
        mask = mask.repeat(anchor_count, n_views)  # Expand mask to match views
        logits_mask = torch.ones_like(mask).scatter_(
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # Zero out self-comparisons

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # Average over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # Final loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
