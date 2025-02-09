import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))
        batch_size = features.shape[0]
        
        # check
        if labels is not None and mask is not None:
            print("error 1")
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            print("error 2")
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # print("good")
            labels = labels.contiguous().view(-1, 1)
            # print(labels.shape)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # print("contrast_count", contrast_count)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print("contrast_Feature", contrast_feature)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            # print("all")
            anchor_feature = contrast_feature
            # print("anchor_feature")
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print(anchor_dot_contrast)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

def intra_modal_contrastive_loss(ts_emb, cxr_emb, text_emb, supcon_loss_fn): 
    """
    각 (환자, 시간) 단위에서 모달리티 간의 일관성을 강화하는 손실함수수

    Args:
        ts_emb, cxr_emb, text_emb: 각각 [B, T, dim]

    Returns:
        intra modal contrastive loss
    """
    B, T, dim = ts_emb.shape
    multi_view = torch.stack([ts_emb, cxr_emb, text_emb], dim=2) # shape: [B, T, 3, dim] (n_views=3)
    multi_view = multi_view.view(B*T, 3, dim) # (환자, 시간 단위)

    unique_labels = torch.arange(B*T, device=ts_emb.device) # 동일한 시간대의 모달리티들에 임시 라벨을 부여하여 Intra Modality Contrastive Loss 학습 (IMCL)
    loss_intra = supcon_loss_fn(multi_view, unique_labels) # ex. 0, 1, 2, ..., B*T
    # print("loss_intra scalar", loss_intra)
    return loss_intra

def inter_label_contrastive_loss(fused_emb, time_labels, supcon_loss_fn):
    """
    (환자, 시간) 단위의 fused embedding을 가지고, 라벨이 같은 샘플끼리 거리를 가깝게 학습하는 손실함수.
    fused_embedding에 gaussian noise를 부여함.

    Args: 
        fused_emb: [B, T, emb_dim]: modality fusion embedding(acutally concat)
        time_labels: [B, T]
        supcon_loss
    Returns: 
        Inter-label contrastive loss (scalar)
    """
    B, T, dim = fused_emb.shape
    emb = fused_emb.view(B*T, dim)
    # print("inter_modal 적용 시 중간 shape: ", emb.shape)
    view_real = emb
    
    # gaussian noise
    mean = 0
    std = 0.1
    noise = np.random.normal(mean, std, emb.shape)
    noise = torch.from_numpy(noise).to(emb.device)
    # print("noise.shape:", noise.shape)
    # print("Noise:", noise)
    view_noise = emb + noise

    views = torch.stack([view_real, view_noise], dim=1)
    # print("views.shape:", views.shape)

    labels = time_labels.view(B*T)
    # print("labels.shape", labels.shape)
    loss_inter = supcon_loss_fn(views, labels=labels)
    # print("loss_inter scalar", loss_inter)
    return loss_inter

def total_contrastive_loss(ts_emb, cxr_emb, text_emb, fused_emb, time_labels, supcon_loss_fn, lambda_intra=0.5, lambda_inter=1.0):
    loss_intra = intra_modal_contrastive_loss(ts_emb, cxr_emb, text_emb, supcon_loss_fn)
    # print("fused_embedding.shape:", fused_emb.shape)
    loss_inter = inter_label_contrastive_loss(fused_emb, time_labels, supcon_loss_fn)

    if loss_intra is None or loss_intra is None: 
        raise ValueError("Check your loss_intra or loss_inter loss function.")
    total_loss = lambda_intra * loss_intra + lambda_inter * loss_inter
    return total_loss