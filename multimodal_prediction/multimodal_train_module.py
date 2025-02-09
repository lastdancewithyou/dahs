import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
from tqdm import tqdm
import wandb
import os
import argparse
import matplotlib.pyplot as plt
import math

from contrastive_dataset import *
from loss import *

# parser setting
# parser = argparse.ArgumentParser(description=)
# parser.add_argument()
# args = parser.parse_args()
# print(args)

# time-series modality
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=256, output_dim=512, num_layers=2, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size = input_dim, 
            hidden_size = hidden_dim, 
            num_layers = num_layers, 
            batch_first=True, 
            bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)  # Bidirectional이므로 2배 크기

    def forward(self, x):
        # x: (B, T, input_dim) → (배치, 시간 길이, 입력 차원)
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out)
        return out

# # Temporal attention
# class PositionalEncoding(nn.Module):
#     """
#     최대 T length(현재 24hr)을 max_len으로 설정. 
#     """
#     def __init__(self, dim_model, dropout_p, max_len):
#         super().__init__()
#         self.dropout = nn.Dropout(dropout_p)

#         # Encoding
#         pos_encoding = torch.zeros(max_len, dim_model)
#         position_list = torch.arange(0, max_len, dtype=torch.float).view(-1,1)
#         division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

#         pos_encoding[:, 0::2] = torch.sin(position_list * division_term)
#         pos_encoding[:, 1::2] = torch.cos(position_list * division_term)


#     def forward(self, ):

# class TemporalTransformer(nn.Module):
#     def __init__(self, dim_model, num_heads, num_layers, dim_feedforward, max_len=24):
#         super().__init__()

#         # Learnable positional encoding
#         self.pos_encoding = PositionalEncoding(d_model=d_model)

#         self.transformer = nn.Transformer(
#             d_model=dim_model,
#             nhead=num_heads,
#             num_encoder_layers= ,
#             num_decoder_layers= ,
#             dropout= ,
#         )

#         self.transformer_encoder = 

    # def forward(self, x):
    #     x = self.pos_encoding(x) # [B, 24, 256]
    #     x = x.permute(1,0,2) # [24, B, 256]
    #     out = self.transformer_encoder(x)
    #     out = out.permute(1,0,2) # [B, 24, 256]
    #     return out

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
    Args:
        ts_series.shape: [B, 24(t), 28(features)]
        img_series.shape: [B, 24(t), 3, 224, 224]
        time_steps.shape: [B, 24(t)]
        text_series.shape: 현재 list 형태

    """

    def __init__(self, alpha=0.03, max_length=256, freeze_bert=True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        self.max_length = max_length

        # ==================== Time-series Modality Encoder + MLP_ts ====================
        self.ts_encoder = LSTMEncoder(input_dim=28, hidden_dim=256, output_dim=512, num_layers=2)
        self.mlp_ts = MLP_ts(input_dim=512, hidden_dim = 256, output_dim=256)

        # ==================== CXR Modality Encoder + MLP_img ====================
        # ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet_backbone = nn.Sequential(*list(resnet.children())[:-1]) # FC 제외

        # 부분 freeze
        # total_layers = sum(1 for _ in self.resnet_backbone.parameters())
        # print(f"Total layers: ", total_layers) # total 159 layers
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

        # ==================== Temporal Attention ====================
        # self.temporal_transformer = TemporalTransformer(
        #     d_model=,
        #     nhead=,
        #     num_layers=,
        #     dim_feedforward=,
        #     max_len=30 # max_len > 24
        # )

        # ==================== Classifier ====================
        # time-series, image, text -> 각각 (B, T, 256)
        # concat -> (B, T, 256*3) -> fusion_fc -> (B, T, 256)
        self.fusion_fc = nn.Linear(256*3, 256)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(256, 3)

    def forward(self, ts_series, img_series, text_series, time_steps, return_logits=False): 
        device = self.device
        B, T, _ = ts_series.shape

        # ================ Time-series Embedding ================
        # LSTM -> (B, T, 512)
        ts_encoded = self.ts_encoder(ts_series) # (B, T, hidden_dim(=512))
        ts_embeddings = self.mlp_ts(ts_encoded).to(device) # (B, T, 256)

        # ================ Img Embedding + decay ================
        # ResNet50 -> (B, T, 3, 224, 224)
        # For CXR decay
        last_cxr_emb = torch.full((B, 256), -999.0, device=device)
        last_cxr_time = torch.full((B,), -1.0, device=device)

        cxr_embeddings = []
        for t in range(T):
            # masked dataset
            is_masked_dataset_cxr = torch.all(img_series[:, t] == -999.0, dim=(1,2,3))

            # original dataset
            is_all_zero = torch.all(img_series[:, t] == 0, dim=(1,2,3))
            valid_idx = ~is_all_zero & ~is_masked_dataset_cxr

            if valid_idx.any(): 
                cur_imgs = img_series[valid_idx, t].to(device) # (N_valid, 3, 224, 224)
                # print(cur_imgs.shape)
                feature = self.resnet_backbone(cur_imgs) # (N_valid, 2048, 1, 1)
                feature = feature.view(feature.size(0), -1) # (N_valid, 2048)
                cxr_emb = self.mlp_img(feature)  # (N_valid, 256)

                last_cxr_emb[valid_idx] = cxr_emb
                last_cxr_time[valid_idx] = time_steps[valid_idx, t]

                # decay 적용
                time_diff = time_steps[:, t].to(device) - last_cxr_time
                decay_factor = torch.clamp(1.0 - self.alpha * time_diff, min=0.0)
                # last_cxr_emb[valid_idx] *= decay_factor[valid_idx].unsqueeze(-1)
                last_cxr_emb = last_cxr_emb * decay_factor.unsqueeze(-1)
                cxr_embeddings.append(last_cxr_emb.clone())
                continue
                
            last_cxr_emb[is_masked_dataset_cxr] = 0.0
            cxr_embeddings.append(last_cxr_emb.clone())

        cxr_embeddings = torch.stack(cxr_embeddings, dim=1) # (B, T, 256)

        # ================ Text Embedding + decay ================
        # For Reports decay
        last_text_emb = torch.full((B, 256), -999.0, device=device)
        last_text_time = torch.full((B,), -1.0, device=device)

        text_embeddings = []
        for t in range(T):
            # masked dataset
            is_masked_dataset_txt = torch.tensor(
                [text_series[b][t] == str(-999.0) for b in range(B)], device=device
            )

            # original dataset
            valid_txt = []
            valid_b_indices = []

            for b in range(B): 
                txt = text_series[b][t]
                if txt != "": 
                    valid_txt.append(txt)
                    valid_b_indices.append(b)
            valid_b_indices = torch.tensor(valid_b_indices, device=device)

            if len(valid_txt) > 0 :
                inputs = self.tokenizer(
                    valid_txt, padding="max_length", truncation=True, max_length=self.max_length, return_tensors='pt'
                ).to(device)

                with torch.no_grad(): 
                    outputs = self.language_model(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :] # (N_valid, 768)
                
                # print(cls_emb.shape)
                text_emb = self.mlp_text(cls_emb) # (768) -> (256)
                last_text_emb[valid_b_indices] = text_emb
                last_text_time[valid_b_indices] = time_steps[valid_b_indices, t].to(device)

                # Decay 적용
                time_diff = time_steps[:, t].to(device) - last_text_time
                decay_factor = torch.clamp(1.0 - self.alpha * time_diff, min=0.0)
                last_text_emb = last_text_emb * decay_factor.unsqueeze(-1)
                # last_text_emb[valid_idx] *= decay_factor[valid_idx].unsqueeze(-1)
                text_embeddings.append(last_text_emb.clone())
                continue

            last_text_emb[is_masked_dataset_txt] = 0.0
            text_embeddings.append(last_text_emb.clone())

        text_embeddings = torch.stack(text_embeddings, dim=1) # (B, T, 256)

        # ================ Multimodal Fusion ================
        """
        ts_embeddings: (B, T, 256)
        cxr_embeddings: (B, T, 256)
        text_embeddings: (B, T, 256)
        Concat embeddings : (B, T, 768)
        """
        # L2 Normalization
        ts_embeddings = F.normalize(ts_embeddings, p=2, dim=-1)
        cxr_embeddings = F.normalize(cxr_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        fusion_input = torch.cat([ts_embeddings, cxr_embeddings, text_embeddings], dim=-1)  # (B, T, 256*3=768)
        fused_embeddings = self.relu(self.fusion_fc(fusion_input))  # (B, T, 256)

        logits = self.classifier(fused_embeddings) # (B, T, 3)

        if return_logits: 
            return fused_embeddings, logits
        
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
    train_loader, val_loader, test_loader = get_dataloaders(ts_df, img_df, text_df, batch_size=8, train_ratio=0.7, val_ratio=0.2, random_seed=0)
    print("(3) Dataloader 정의 완료.")

    model = MultiModalModel().to(device)
    supcon_loss_fn = SupConLoss(temperature=0.07)
    classification_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopper = Earlystopping(patience=patience)

    lambda_intra = 1.0
    lambda_inter = 0.7

    num_epochs = 1
    batch_size = 6
    args = {
        "learning_rate": 1e-4,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "alpha": 0.03
    }

    # wandb.init(
    #     project="Supervised Contrastive Module Train",
    #     name = "First experiment: (SCL Only + Classification) without augmentation",
    #     config = args
    #     )
    
    for epoch in tqdm(range(num_epochs), total=num_epochs, desc="Training Processing"):
        model.train()
        train_loss = 0.0

        for batch in train_loader: 
            optimizer.zero_grad()

            stay_ids = batch['stay_ids'].to(device)
            stay_ids = stay_ids.unsqueeze(1).expand(-1, 24)
            labels = batch['labels'].to(device)
            # print("stay_id.shape: ", stay_ids.shape)
            # print("labels.shape: ", labels.shape)

            all_ts_series = []
            all_img_series = []
            all_text_series = []
            all_time_steps = []

            for series in batch['modality_series']:
                ts_list = [time_step['ts_features'] for time_step in series]
                ts_array = np.stack(ts_list, axis=0)
                ts_tensor = torch.from_numpy(ts_array).float().to(device)
                all_ts_series.append(ts_tensor)

                img_list = [time_step['cxr_tensor'] for time_step in series]
                img_tensor = torch.stack(img_list, dim=0)
                all_img_series.append(img_tensor)

                text_list = [time_step['text_content'] if time_step['text_content'] is not None else "" for time_step in series]
                all_text_series.append(text_list)

                time_list = [time_step['time_step'] for time_step in series]
                time_tensor = torch.tensor(time_list, dtype=torch.float32, device=device)
                all_time_steps.append(time_tensor)

            ts_series = torch.stack(all_ts_series, dim=0) # [1, 24]
            img_series = torch.stack(all_img_series, dim=0) # [1, 24, 28]
            text_series = all_text_series
            time_steps = torch.stack(all_time_steps, dim=0) # [1, 24, 3, 224, 224]

            # shape check
            # print("stay_ids.shape: ", stay_ids.shape)
            # print("labels.shape: ", labels.shape)
            # print("ts_series.shape: ", ts_series.shape)
            # print("img_series.shape: ", img_series.shape)
            # print("time_steps.shape: ", time_steps.shape)
            # print("len(text_series): ", len(text_series))
            # print("len(text_series[0]): ", len(text_series[0]))

            fused_embeddings, ts_embeddings, cxr_embeddings, text_embeddings, logits = model(ts_series, img_series, text_series, time_steps)
            contrastive_loss = total_contrastive_loss(ts_embeddings, cxr_embeddings, text_embeddings, fused_embeddings, labels, supcon_loss_fn, lambda_inter, lambda_inter)

            logits_2d = logits.view(-1, logits.shape[-1])
            labels_1d = labels.view(-1)
            labels_1d = labels_1d + 1 # -1 -> 0, 0 -> 1, 1 -> 2

            # print(f"Reshaped logits.shape: {logits_2d.shape}")
            # print(f"Reshaped labels.shape: {labels_1d.shape}")

            # classification_loss = classification_loss_fn
            classification_loss = classification_loss_fn(logits_2d, labels_1d)
            loss = contrastive_loss + lambda_ce * classification_loss

            # backpropagation and optimization
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_avg_loss = train_loss / (len(train_loader)) # 4배로 증강했기 때문에 이와 같이 설정함.
        print(f"[Epoch {epoch}] Train Loss = {train_avg_loss: .4f}")
        # wandb.log({"Train loss": avg_loss.item()})

        print("<Validation Process>")
        val_metrics = evaluate_model(model, val_loader, device, classification_loss_fn)
        val_loss = val_metrics['loss']

        if early_stopper.early_stop(val_loss, model, epoch):
            print(f"Early stopping is triggered. Best Model at Epoch {epoch}")

        # tsne visualization
        if epoch == 0 or epoch % 10 == 0:
            print(f"[Epoch {epoch}] tsne 시각화")
            embedding_tsne(val_metrics['embeddings'], val_metrics['labels'], epoch, save_path="tsne/contrastive_space")

    print("<Test Process>")
    test_metrics = evaluate_model(model, test_loader, device, classification_loss_fn)
    embedding_tsne(test_metrics['embeddings'], test_metrics['labels'], epoch, save_path="tsne/contrastive_space/test")
    return model, train_avg_loss, val_metrics, test_metrics

def evaluate_model(model, dataloader, device, criterion):
    """
    val, test dataloader에 대한 평가 진행함.

    Args:
        Trained Multi-modal Model
        Dataloader (Val / Test)
        device
        criterion: CrossEntropy Loss
    Returns: 
        loss
        accuracy
        auroc
        auprc
    """
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []
    all_embeddings = [] # for t-SNE

    buffer_min_samples = 24 * 3

    with torch.no_grad():
        for batch in dataloader:
            stay_ids = batch['stay_ids'].to(device)
            stay_ids = stay_ids.unsqueeze(1).expand(-1, 24)
            labels = batch['labels'].to(device)

            all_ts_series = []
            all_img_series = []
            all_text_series = []
            all_time_steps = []

            for series in batch['modality_series']:
                ts_list = [time_step['ts_features'] for time_step in series]
                ts_array = np.stack(ts_list, axis=0)
                ts_tensor = torch.from_numpy(ts_array).float().to(device)
                all_ts_series.append(ts_tensor)

                img_list = [time_step['cxr_tensor'] for time_step in series]
                img_tensor = torch.stack(img_list, dim=0)
                all_img_series.append(img_tensor)

                text_list = [time_step['text_content'] if time_step['text_content'] is not None else "" for time_step in series]
                all_text_series.append(text_list)

                time_list = [time_step['time_step'] for time_step in series]
                time_tensor = torch.tensor(time_list, dtype=torch.float32, device=device)
                all_time_steps.append(time_tensor)

            ts_series = torch.stack(all_ts_series, dim=0) # [1, 24]
            img_series = torch.stack(all_img_series, dim=0) # [1, 24, 28]
            text_series = all_text_series
            time_steps = torch.stack(all_time_steps, dim=0) # [1, 24, 3, 224, 224]

            fused_embeddings, logits = model(ts_series, img_series, text_series, time_steps, return_logits=True)

            logits_2d = logits.view(-1, logits.shape[-1])
            # num_classes = logits_2d.shape[1]
            # print("num_classes:", num_classes)

            labels_1d = labels.view(-1)
            labels_1d = labels_1d + 1

            # print(f"logits_2d.shape: {logits_2d.shape}")
            # print(f"labels_1d.shape: {labels_1d.shape}")

            loss = criterion(logits_2d, labels_1d)
            total_loss += loss.item()

            preds = torch.argmax(logits_2d, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_1d.cpu().numpy())

            probs = torch.softmax(logits_2d, dim=1)
            all_probs.extend(probs.cpu().numpy())

            all_embeddings.append(fused_embeddings.cpu().numpy())
            embeddings = np.concatenate(all_embeddings, axis=0)

            # if save_embeddings: 
            #     all_embeddings.append(fused_embeddings.cpu().numpy())
            #     embeddings = np.concatenate(all_embeddings, axis=0)

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    # print(all_preds)
    # print(all_labels)
    # all_labels가 [B, T] 형태인 경우:
    all_labels_flat = all_labels.flatten()   # shape: [B*T]
    # print("all_labels_flat.shape", all_labels_flat.shape)

    if all_labels_flat.shape[0] < buffer_min_samples or np.unique(all_labels_flat).size < 3: 
        print("Warning: 3개의 클래스가 모두 포함되지 않아 AUROC/AUPRC를 계산할 수 없습니다.")
        auroc = None
        auprc = None
    else: 
        auroc = roc_auc_score(all_labels_flat, all_probs, multi_class='ovr', average='macro')
        auprc = average_precision_score(nn.functional.one_hot(torch.tensor(all_labels_flat), num_classes=3).numpy(), all_probs, average='macro')
    print("AUROC: ", auroc)
    print("AUPRC: ", auprc)

    return {
        'loss': avg_loss,
        'AUROC': auroc,
        'AUPRC': auprc,
        'embeddings': embeddings,
        'labels': all_labels_flat
    }

class Earlystopping:
    def __init__(self, patience=5, save_path="checkpoints/"):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.save_path = save_path

    def early_stop(self, loss, model, epoch):
        if loss < self.best_loss: 
            self.best_loss = loss
            self.counter = 0
            save_filename = os.path.join(self.save_path, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_filename)
            print(f"성능 향상! 모델 가중치 저장을 완료했습니다. [Epoch {epoch}], Val Loss: {self.best_loss}")
        else: 
            self.counter += 1
            print(f"성능 개선 없음. patience 추가 {self.counter + 1}")

        return self.counter >= self.patience
    
def embedding_tsne(embeddings, labels, epoch, save_path="tsne"):

    n_samples = embeddings[0].shape
    print("n_samples: ", n_samples)

    print("embeddings.shape")

    if embeddings.ndim==3:
        embeddings = embeddings.reshape(-1, embeddings.shape[-1])
    
    print("Flattened embeddings shape:", embeddings.shape)

    tsne = TSNE(n_components=2, perplexity=20, random_state=0)
    tsne_result = tsne.fit_transform(embeddings)

    tsne_df = pd.DataFrame(tsne_result, columns = ['component 0', 'component 1'])
    tsne_df['class'] = labels.flatten()

    plt.figure(figsize=(12, 12))

    # modality_marker = {0: 'o', 1: '^', 2:'x'}
    colors = ['forestgreen', 'sandybrown', 'tomato']
    class_names = ['Uncertain', 'Negative', 'Positive']
    for idx in range(3): 
        subset = tsne_df[tsne_df['class']==idx]
        plt.scatter(
            subset['component 0'],
            subset['component 1'],
            color = colors[idx],
            label = class_names[idx],
            s = 10,
            alpha = 0.8
        )

    filename = os.path.join(save_path, f"tsne_epoch_{epoch}.png")

    plt.legend(fontsize=12, handlelength=3)
    plt.title(f"t-SNE visualization (Epoch {epoch})")
    plt.savefig(filename)
    plt.close()
    print(f"[t-SNE] 결과가 '{filename}'에 저장되었습니다.")