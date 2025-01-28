# library import
import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import Dataset, DataLoader

# file path
root_dir = 'C:/Users/gangmin/dahs/my research'
sequential_df_reports = pd.read_feather(root_dir + '/processed/sequential_df_reports.ftr')

# Radiology report modality
class TextDataset_decay(Dataset): 
    def __init__(self, dataframe, language_model, tokenizer, alpha=0.03, max_length=256, device=None):
        self.df = dataframe.reset_index(drop=True)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cuda check
        if torch.cuda.is_available():
            print(f"Dataset will use GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("Dataset will use CPU")

        # model & tokenizer
        self.language_model = language_model.to(self.device)
        self.tokenizer = tokenizer

        self.alpha = alpha
        self.max_length = max_length

        self.text_embeddings = []

        # 직전 슬롯 위치 저장
        last_emb = None
        last_slot_with_text = None

        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)): 
            hour_slot = row['hour_slot']
            text_flag = row['text_flag']
            text = row.get('extracted_text', "") # 실제 텍스트

            # embedding extraction
            if text_flag == 1:
                emb = self.get_text_embeddings(text)
                self.text_embeddings.append(emb)
                last_emb = emb
                last_slot_with_text = hour_slot

            else: 
                if last_emb is None: # 텍스트가 없으면 zero vector로 대체함.
                    self.text_embeddings.append(torch.zeros(768).to(self.device))

                else: 
                    slot_diff = hour_slot - last_slot_with_text # 앞서 텍스트가 존재할 경우 linear decay 적용.
                    decay_factor = max(1 - self.alpha * slot_diff, 0)
                    decay_emb = last_emb * decay_factor # 슬롯 간격에 따른 상대적 적용.
                    self.text_embeddings.append(decay_emb)
        
        self.text_embeddings = torch.stack(self.text_embeddings)
        print(self.text_embeddings.shape)

    def get_text_embeddings(self, text): 
        inputs = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.language_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            cls_emb = cls_emb.squeeze(0) # 배치 차원 제거
        return cls_emb
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        stay_id = row['stay_id']
        text_emb = self.text_embeddings[idx]
        # 이미지 모달리티 라벨부분과 비교해서 None 처리 확인할 것
        label = torch.tensor(row['Edema'], dtype=torch.float32).to(self.device)

        return {
            'stay_id': stay_id,
            'hour_slot': row['hour_slot'],
            'text_emb': text_emb,
            'label': label
        }