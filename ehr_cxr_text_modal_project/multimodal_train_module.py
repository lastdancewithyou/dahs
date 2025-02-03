import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

from tqdm import tqdm
import wandb

from contrastive_dataset import *
from loss import *
# from tsne import *

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=29, hidden_dim=256, output_dim=512, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Bidirectional이므로 2배 크기

    def forward(self, x):
        # x: (B, T, input_dim) → (배치, 시간 길이, 입력 차원)
        _, (hn, _) = self.lstm(x)  # 최종 hidden state (hn)를 가져옴

        # 양방향 LSTM이므로 두 방향의 hidden state를 concat
        x = torch.cat([hn[-2], hn[-1]], dim=-1)  # (B, hidden_dim * 2)

        x = self.fc(x)  # 최종 임베딩 차원 (B, output_dim)
        return x

# ==================== MLP ====================
# Time-seires modality MLP
class MLP_ts(nn.Module): 
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=256):
        super(MLP_ts, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Image modality MLP
class MLP_img(nn.Module): 
    def __init__(self, input_dim=2048, hidden_dim=1024, output_dim=256):
        super(MLP_img, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Text modality MLP
class MLP_text(nn.Module): 
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=256):
        super(MLP_text, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

#######################################################################
# Multimodal Model Define
#######################################################################
class MultiModalModel(nn.Module):
    """

    """

    def __init__(self, alpha=0.03, max_length=256, freeze_bert=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.max_length = max_length

        # ==================== Time-series Modality Encoder + MLP_ts ====================
        self.ts_encoder = LSTMEncoder(input_dim=29, hidden_dim=256, output_dim=512, num_layers=2)
        self.mlp_ts = MLP_ts(input_dim=512, hidden_dim = 256, output_dim=256)

        # ==================== CXR Modality Encoder + MLP_img ====================
        # ResNet50
        resnet = models.resnet50(pretrained=True)
        self.resnet_backbone = nn.Sequential(*list(resnet.children()[:-1])) # FC 제외

        # 부분 freeze
        total_layers = sum(1 for _ in self.resnet_backbone.parameters())
        print(f"Total layers: ", total_layers)
        count = 0
        for param in self.resnet_backbone.parameters():
            if count > 140:
                break
            param.requires_grad=False
            count += 1
        self.mlp_img = MLP_img(input_dim=2048, hidden_dim=1024, output_dim=256)

        # ==================== Radiology report Modality Encoder + MLP_text ====================
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.language_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # model freeze
        if freeze_bert: 
            for param in self.language_model.parameters(): 
                param.requires_grad = False

        self.mlp_text = MLP_text(input_dim=768, hidden_dim=512, output_dim=256).to(self.device)
        # ==================== Classifier ====================
        self.fusion_fc = nn.Linear(256*3, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, 3)

    def forward(self, ts_series, img_series, text_series, time_steps): 
        """
        Multimodal data forward pass
        
        Args: 
            ts_series (torch.Tensor): (batch_size, 24, ts_dim)
            img_series (torch.Tensor): (batch_size, 24, 256)
            text_series (list of str): (batch_size, 24)

        Returns:
            all_seq_embeddings: 모든 모달리티가 concat 된 임베딩
            ts_seq_embeddings, img_seq_embeddings, text_seq_embeddings, logtis
        """
        device = self.device
        batch_size = ts_series.shape[0]

        # ================ Time-series Embedding ================
        ts_encoded = self.ts_encoder(ts_series) # (B, 512)
        ts_embeddings = self.mlp_ts(ts_encoded).to(device) # (B, 256)

        # ================ Img Embedding + decay ================
        # For CXR decay
        last_cxr_emb = torch.zeros((batch_size, 256), device=device)
        last_cxr_time = torch.full((batch_size,), -1, device=device)

        cxr_embeddings = []
        for t in range(img_series.shape):
            valid_idx = img_series[:, t, :].sum(dim=1) != 0

            if valid_idx.any(): 
                feature = self.resnet_backbone(img_series[valid_idx, t, :].unsqueeze(1))  # (N_valid, 2048, 1, 1)
                feature = feature.view(feature.size(0), -1)  # (N_valid, 2048)
                cxr_emb = self.mlp_img(feature)  # (N_valid, 256)
                last_cxr_emb[valid_idx] = cxr_emb
                last_cxr_time[valid_idx] = time_steps[valid_idx, t]

            # Decay 적용
            time_diff = time_steps[:, t] - last_cxr_time
            decay_factor = torch.clamp(1 - self.alpha * time_diff, min=0)
            cxr_emb = last_cxr_emb * decay_factor.unsqueeze(1)
            cxr_embeddings.append(cxr_emb)

        cxr_embeddings = torch.stack(cxr_embeddings, dim=1) # (B, T, 256)

        # ================ Text Embedding + decay ================
        # For Reports decay
        last_text_emb = torch.zeros((batch_size, 256), device=device)
        last_text_time = torch.full((batch_size,), -1, device=device)

        text_embeddings = []
        for t in range(len(text_series[0])):
            valid_idx = [bool(txt) for txt in text_series[:, t]]

            if any(valid_idx):
                inputs = self.tokenizer(
                    [text_series[i][t] for i in range(batch_size) if valid_idx[i]],
                    padding="max_length", truncation=True,
                    max_length=self.max_length, return_tensors='pt'
                ).to(device)

                with torch.no_grad(): 
                    outputs = self.language_model(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :] # (1, 768)
                
                text_emb = self.mlp_text(cls_emb.squeeze(0)) # (768) -> (256)
                last_text_emb[valid_idx] = text_emb
                last_text_time[valid_idx] = time_steps[valid_idx, t]

            # Decay 적용
            time_diff = time_steps[:, t] - last_text_time
            decay_factor = torch.clamp(1 - self.alpha * time_diff, min=0)
            text_emb = last_text_emb * decay_factor.unsqueeze(1)
            text_embeddings.append(text_emb)

        text_embeddings = torch.stack(text_embeddings, dim=1) # (B, T, 256)

        # ================ Multimodal Fusion ================
        fused_embeddings = torch.cat([ts_embeddings.unsqueeze(1), cxr_embeddings, text_embeddings], dim=-1)  # (B, T, 768)
        fused_embeddings = self.relu(self.fusion_fc(fused_embeddings))  # (B, T, 256)

        logits = self.classifier(fused_embeddings.mean(dim=1))  # (B, num_classes=3)

        return fused_embeddings, ts_embeddings, cxr_embeddings, text_embeddings, logits 

def train_multimodal_model(ts_df, img_df, text_df, lambda_ce=0.3, patience=5):
    """
    Multimodal Model Training (Supervised Contrastive Learning + Classification loss optimization)

    Args: 
        ts_df: time-series data
        img_df: cxr data
        text_df: radiology report data
        lambda_ce: Contrastive Loss 대비 Cross-entropy weight
    
    Returns:
        Validation & Test loss, Accuracy, AUROC, AUPRCS
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(ts_df, img_df, text_df, batch_size=8)

    model = MultiModalModel().to(device)
    contrastive_loss_fn = SupConLoss(temperature=0.07, base_temperature=0.07)
    classification_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopper = Earlystopping(patience=patience, save_path= root_dir + "best_model.pth")

    num_epochs = 10 # temp
    batch_size = 8
    args = {
        "learning_rate": 1e-4,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "alpha": 0.03
    }

    wandb.init(
        project="Supervised Contrastive Module Train",
        name = "First experiment: SCL Only + Classification",
        config = args
        )
    
    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        for batch in train_loader: 
            optimizer.zero_grad()

            # masking batches generation
            augmented_batches = apply_modality_masking(batch)

            for masked_batch in augmented_batches: 
                labels = masked_batch['labels'].to(device)

                ts_series = torch.stack([torch.tensor([time_step['ts_features'] for time_step in series], dtype=torch.float32) for series in masked_batch['modality_series']]).to(device)
                img_series = torch.stack([torch.tensor([time_step['cxr_tensor'] for time_step in series], dtype=torch.float32) for series in masked_batch['modality_series']]).to(device)
                text_series = [[time_step['text_content'] for time_step in series] for series in masked_batch['modality_series']]
                time_steps = torch.stack([torch.tensor([time_step['time_step'] for time_step in series], dtype=torch.float32) for series in masked_batch['modality_series']]).to(device)

                fused_embeddings, ts_embeddings, cxr_embeddings, text_embeddings, logits = model(ts_series, img_series, text_series, time_steps)

                contrastive_loss = contrastive_loss_fn(fused_embeddings, labels)
                classification_loss = classification_loss_fn(logits, labels.view(-1))
                loss = contrastive_loss + lambda_ce * classification_loss

                # backpropagation and optimization
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
        
        avg_loss = train_loss / (len(train_loader) * 4) # 4배로 증강했기 때문에 이와 같이 설정함.
        print(f"[Epoch {epoch}] Train Loss = {avg_loss: .4f}")
        wandb.log({"Train loss": avg_loss.item()})

        # tsne visualization
        # if epoch % 5 == 0:
        #     embeddings = torch.cat(embeddings_list, dim=0)
        #     labels = torch.cat(labels_list, dim=0)
        #     plot_tsne(embeddings, labels, epoch)

        # early_stopping 검증
        val_metrics = evaluate_model(model, val_loader, device, classification_loss_fn)

        if early_stopper.early_stop(val_metrics['loss'], model, epoch):
            print(f"Early stopping. Best Model: Epoch {epoch}")
            break

    test_metrics = evaluate_model(model, test_loader, device, classification_loss_fn)

    return model, val_metrics, test_metrics

def evaluate_model(model, loader, device, criterion): 
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            # masking batches generation
            augmented_batches = apply_modality_masking(batch)

            for masked_batch in augmented_batches: 
                labels = masked_batch['labels'].to(device)

                ts_series = torch.stack([torch.tensor([time_step['ts_features'] for time_step in series], dtype=torch.float32) for series in masked_batch['modality_series']]).to(device)
                img_series = torch.stack([torch.tensor([time_step['cxr_tensor'] for time_step in series], dtype=torch.float32) for series in masked_batch['modality_series']]).to(device)
                text_series = [[time_step['text_content'] for time_step in series] for series in masked_batch['modality_series']]
                time_steps = torch.stack([torch.tensor([time_step['time_step'] for time_step in series], dtype=torch.float32) for series in masked_batch['modality_series']]).to(device)

                # Forward
                fused_embeddings, ts_embeddings, cxr_embeddings, text_embeddings, logits = model(ts_series, img_series, text_series, time_steps)

                # loss calculation
                loss = criterion(logits, labels.view(-1))
                total_loss += loss.item()

                probs = F.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                all_labels.extend(labels.view(-1).cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

                correct += (preds == labels.view(-1)).sum().item()
                total += labels.numel()

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    auroc = roc_auc_score(all_labels, all_probs)
    auprc = average_precision_score(all_labels, all_probs)

    wandb.log({'test_loss': avg_loss, "Accuracy": accuracy, "AUROC": auroc, "AUPRC": auprc})
    return {
        "loss": avg_loss,
        "Accuracy": accuracy,
        "AUROC": auroc, 
        "AUPRC": auprc
    }

class Earlystopping:
    def __init__(self, patience=5, save_path="best_model.pth"):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.save_path = save_path

    def early_stop(self, loss, model, epoch):
        if loss < self.best_loss: 
            self.best_loss = loss
            self.counter = 0
            save_filename = f"{self.save_path}_epoch{epoch}.pth"
            torch.save(model.state_dict(), save_filename)
            print(f"성능 향상. 모델 가중치 저장을 완료했습니다. Epoch: {epoch}, Loss: {self.best_loss}")
        else: 
            self.counter += 1
            print(f"성능 개선 없음. patience 추가 {self.counter + 1}")

        return self.counter >= self.patience