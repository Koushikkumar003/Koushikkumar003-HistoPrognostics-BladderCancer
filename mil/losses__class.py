import torch

class FocalTverskyLoss(torch.nn.Module):
    """
    Focal Tversky Loss for image segmentation tasks.
    
    Combines Tversky index with focal loss mechanism to handle:
    - Class imbalance (via alpha, beta parameters)
    - Hard/easy sample imbalance (via gamma parameter)
    
    Particularly effective for medical image segmentation where:
    - Foreground (positive) class is much smaller than background
    - Model needs to focus on hard-to-segment regions
    """
    
    def __init__(self, alpha, beta, gamma):
        """
        Initialize Focal Tversky Loss.
        
        Args:
            alpha (float): Weight for False Positives. Higher alpha penalizes FP more.
                          Typical range: 0.3-0.7. Use higher values when precision is important.
            beta (float): Weight for False Negatives. Higher beta penalizes FN more.
                         Typical range: 0.3-0.7. Use higher values when recall is important.
            gamma (float): Focal parameter. Higher gamma focuses more on hard samples.
                          Typical range: 1.0-2.0. Use 1.0 for standard Tversky, >1.0 for focal effect.
        
        Note: alpha + beta should typically equal 1.0 for balanced weighting
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha  # Controls FP penalty
        self.beta = beta    # Controls FN penalty  
        self.gamma = gamma  # Controls focus on hard samples
    
    def forward(self, inputs, targets, epsilon=1e-3):
        """
        Calculate Focal Tversky Loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits), shape: (N, C, H, W) or (N, H, W)
            targets (torch.Tensor): Ground truth labels (0/1), same shape as inputs
            epsilon (float): Small constant to avoid division by zero
            
        Returns:
            torch.Tensor: Focal Tversky loss value (scalar)
        """
        
        # Apply sigmoid activation to convert logits to probabilities [0,1]
        # Comment out if your model already includes sigmoid/softmax activation
        inputs = torch.sigmoid(inputs)
        
        # Flatten tensors to 1D for easier computation
        # Converts (N, C, H, W) -> (N*C*H*W,) for pixel-wise comparison
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate confusion matrix components
        # TP: True Positives - correctly predicted positive pixels
        TP = (inputs * targets).sum()
        
        # FP: False Positives - incorrectly predicted as positive (background predicted as foreground)
        FP = ((1 - targets) * inputs).sum()
        
        # FN: False Negatives - incorrectly predicted as negative (foreground predicted as background)
        FN = (targets * (1 - inputs)).sum()
        
        # Calculate Tversky Index
        # Tversky = TP / (TP + α*FP + β*FN)
        # - Similar to F1-score but with adjustable weights for FP and FN
        # - α controls penalty for false positives (precision focus)
        # - β controls penalty for false negatives (recall focus)
        # - epsilon prevents division by zero
        Tversky = (TP + epsilon) / (TP + self.alpha * FP + self.beta * FN + epsilon)
        
        # Apply Focal mechanism
        # FocalTversky = (1 - Tversky)^γ
        # - γ=1: Standard Tversky loss
        # - γ>1: Focuses more on hard samples (low Tversky index)
        # - Higher γ puts more weight on difficult-to-segment regions
        FocalTversky = (1 - Tversky) ** self.gamma
        
        return FocalTversky


# Example usage:
if __name__ == "__main__":
    # Example configurations for different scenarios:
    
    # Balanced segmentation (equal importance to precision and recall)
    loss_balanced = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)
    
    # Precision-focused (penalize false positives more)
    loss_precision = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.5)
    
    # Recall-focused (penalize false negatives more, good for medical segmentation)
    loss_recall = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
    
    # Example tensors
    predictions = torch.randn(2, 1, 256, 256)  # Batch of 2, 1 channel, 256x256
    ground_truth = torch.randint(0, 2, (2, 1, 256, 256)).float()  # Binary masks
    
    # Calculate loss
    loss = loss_recall(predictions, ground_truth)
    print(f"Focal Tversky Loss: {loss.item():.4f}")