import torch
import torch.nn as nn
import torch.nn.functional as F

# class Multimodal_Contrastive_Loss(nn.Module):
#     def __init__(self, temperature=0.07, lambda_weights=[0.4, 0.4, 0.2]):
#         super(Multimodal_Contrastive_Loss, self).__init__()
#         self.supcon_loss = SupConLoss(temperature=temperature)
#         self.iemcl_loss = InterModalityConLoss(temperature=temperature)
#         self.iamcl_loss = InterAugmentModalityConLoss(temperature=temperature)

#         self.lambda_weight = lambda_weights # [λ1, λ2, λ3]
    
#     def forward(self, features, labels, stay_ids):
#         """
        
#         """
#         loss_supcon = self.supcon_loss(features, labels)
#         loss_iemcl = self.iemcl_loss(features, stay_ids)
#         loss_iamcl = self.iamcl_loss(features, labels)

#         total_loss = self.lambda_weight[0] * loss_supcon + self.lambda_weight[1] * loss_iemcl + self.lambda_weight[2] * loss_iamcl
#         return total_loss

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None): 
        """
        Compute contrastive loss for multimodal embeddings.
        Args: 
            features: [B, M, D] - B=batch, M=modality count, D=embedding dim
            labels: [B] - ground truth labels (optional)
        Returns:
            loss: contrastive loss scalar
        """
        device = features.device
        batch_size, modality_count, _ = features.shape

        # Flatten features: [B*M, D]
        contrast_feature = features.permute(1, 0, 2).contiguous().view(batch_size * modality_count, -1)

        # Compute similarity matrix
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature
        )

        # Avoid numerical instability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Handle label-based or unsupervised contrastive loss
        if labels is None:
            mask = torch.eye(batch_size * modality_count).to(device)  # Unsupervised mode
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
            logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
            mask = mask * logits_mask  # Remove self-comparisons

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp(min=1e-6))

        # Compute loss (only valid pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-6)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()

    # def forward(self, features, labels=None): 
    #     """
    #     Compute contrastive loss for multimodal embeddings.
    #     Args: 
    #         features: [B, M, D] - B=batch, M=modality count, D=embedding dim
    #         labels: [B] - ground truth labels
    #     Returns:
    #         loss: contrastive loss scalar
    #     """
    #     device = features.device
    #     batch_size, modality_count, _ = features.shape

    #     # Flatten features: [B*M, D]
    #     # contrast_feature = features.view(batch_size * modality_count, -1)
    #     contrast_feature = features.permute(1, 0, 2).contiguous().view(batch_size * modality_count, -1)


    #     # Compute similarity matrix
    #     anchor_dot_contrast = torch.div(
    #         torch.matmul(contrast_feature, contrast_feature.T),
    #         self.temperature
    #     )

    #     # Avoid numerical instability
    #     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    #     logits = anchor_dot_contrast - logits_max.detach()

    #     # Label similarity mask
    #     labels = labels.contiguous().view(-1, 1)
    #     mask = torch.eq(labels, labels.T).float().to(device)

    #     # Mask diagonal (self-contrast 제거)
    #     logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
    #     mask = mask * logits_mask

    #     # Compute log_prob
    #     exp_logits = torch.exp(logits) * logits_mask
    #     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    #     # Compute loss (only valid pairs)
    #     mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    #     loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    #     return loss.mean()


# class InterModalityConLoss(nn.Module): 
#     def __init__():

#     def forward(): 

# class InterAugmentModalityConLoss(nn.Module):
#     def __init__():

#     def forward():