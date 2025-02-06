# library import
import pandas as pd
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path
import copy
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
os.environ["PYTORCH_NO_FLASHATTN"] = "1"

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

class SCL_Multi_Dataset(Dataset): 
    def __init__(self, merged_df, alpha=0.03, device=None):
        
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Cuda check
        # if torch.cuda.is_available():
        #     print(f"Dataset will use GPU: {torch.cuda.get_device_name(self.device)}")
        # else:
        #     print("Dataset will use CPU")

        # dataframe
        self.merged_df = merged_df.copy()
        
        # stay_id grouping
        self.stay_groups = self.merged_df.groupby('stay_id')
        self.stay_ids = list(self.stay_groups.groups.keys())

    def load_image(self, image_path):
        if pd.isna(image_path):
            return torch.zeros((3,224,224), dtype=torch.float32)
        
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # BGR to RGB
        processed_image = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA) # Resize
        image_tensor = torch.from_numpy(processed_image).float() # tensor
        if image_tensor.shape == (224,224,3): 
            image_tensor = image_tensor.permute(2,0,1)
        return image_tensor
    
    #######################################################################
    # stay_id grouping -> hour_slot iteration -> decay logic
    #######################################################################
    def __getitem__(self, idx): 
        stay_id = self.stay_ids[idx]
        stay_data = self.stay_groups.get_group(stay_id).sort_values('hour_slot')
        # print(stay_data)

        hour_slots = stay_data['hour_slot'].to_numpy()
        labels = stay_data['label'].to_numpy()

        # ts_columns = []
        ts_values = stay_data.drop(columns=['stay_id', 'hour_slot', 'image_path', 'cxr_flag', 'extracted_text', 'text_flag', 'label']).to_numpy()
        
        image_paths = stay_data['image_path'].to_numpy()
        cxr_flags = stay_data['cxr_flag'].to_numpy()
        images = [self.load_image(img_path).to(self.device) for img_path in image_paths]

        extracted_texts = stay_data['extracted_text'].to_numpy()
        text_flags = stay_data['text_flag'].to_numpy()

        sequence_list = [
            {
                # 'stay_id': stay_id,
                'time_step': hour_slots[i],
                'ts_features': ts_values[i],
                'cxr_tensor': images[i],
                'has_cxr': cxr_flags[i],
                'text_content': extracted_texts[i],
                'has_text': text_flags[i]
            }
            for i in range(len(stay_data))
        ]

        return {
            'stay_id': stay_id,
            'modality_series': sequence_list,
            'label_series': labels.tolist()  # NumPy 배열을 리스트로 변환
        }

    def __len__(self): 
        return len(self.stay_ids)
    
#######################################################################
# 데이터 구조 정의
#######################################################################
# merging dataframes
def merged_dataframes(ts_df, img_df, text_df): 
    ts_df = ts_df.rename(columns={'Edema':'label'})

    img_df = img_df[['stay_id', 'hour_slot', 'cxr_flag', 'image_path', 'Edema']].copy() # cxr_dataset
    img_df.drop(columns=['Edema'], inplace=True)

    text_df = text_df[['stay_id', 'hour_slot', 'text_flag', 'extracted_text', 'Edema']].copy() # label_imputed_report
    text_df.drop(columns=['Edema'], inplace=True)

    merged_df = (
            ts_df
            .merge(img_df, on=['stay_id', 'hour_slot'], how='outer')
            .merge(text_df, on=['stay_id','hour_slot'], how='outer'))
    return merged_df

# stratified split
def stratified_split_dataset(merged_df, label_column='label', train_ratio=0.7, val_ratio=0.2, random_seed=0):
    """
    merged_df를 train, val, test로 층화추출(Stratified)하여 분할
    """
    print("(2) Stratified Data Split 시작")

    stay_labels = merged_df.groupby("stay_id")[label_column].last().reset_index()

    train_stay_ids, temp_stay_ids = train_test_split(
        stay_labels['stay_id'],
        test_size=(1 - train_ratio),
        stratify=stay_labels[label_column],
        random_state=random_seed
    )

    val_size = val_ratio / (1 - train_ratio)
    val_stay_ids, test_stay_ids = train_test_split(
        temp_stay_ids,
        test_size=(1 - val_size), 
        stratify=stay_labels.loc[stay_labels['stay_id'].isin(temp_stay_ids), label_column],
        random_state=random_seed
    )

    train_df = merged_df[merged_df['stay_id'].isin(train_stay_ids)]
    val_df = merged_df[merged_df['stay_id'].isin(val_stay_ids)]
    test_df = merged_df[merged_df['stay_id'].isin(test_stay_ids)]

    print(f" - Train: {len(train_stay_ids)}개 stay_id")
    print(f" - Val:   {len(val_stay_ids)}개 stay_id")
    print(f" - Test:  {len(test_stay_ids)}개 stay_id")
    return train_df, val_df, test_df

def get_dataloaders(ts_df, img_df, text_df, batch_size, train_ratio=0.7, val_ratio=0.2, random_seed=0):
    print("(1) Dataloader 정의 시작.")

    # 데이터 병합
    merged_df = merged_dataframes(ts_df, img_df, text_df)
    print(merged_df.columns)

    # 층화 추출
    train_df, val_df, test_df = stratified_split_dataset(
        merged_df=merged_df,
        label_column='label',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )

    # dataset 생성
    train_dataset = SCL_Multi_Dataset(train_df)
    val_dataset = SCL_Multi_Dataset(val_df)
    test_dataset = SCL_Multi_Dataset(test_df)

    # dataloader 정의
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn)
    return train_dataloader, val_dataloader, test_dataloader

def multimodal_collate_fn(batch): 
    """
    1. stay_id 내에서 time_step 순서 유지
    2. 개별 데이터 샘플이 섞이지 않도록 stay_id 단위로 묶어서 shuffle
    """
    stay_ids = torch.tensor([item['stay_id'] for item in batch], dtype=torch.long)
    modality_series = [item['modality_series'] for item in batch]
    labels = torch.tensor([item['label_series'] for item in batch], dtype=torch.long)
    # print(f"Labels shape: {labels.shape}")
    # print(labels)
    return {'stay_ids': stay_ids,
            'modality_series': modality_series, # 24hr data in each modalities
            'labels': labels # 24hr answer labels(Pulmonary Edema)
            }

#######################################################################
# missing modality generator by masking
#######################################################################
DEFAULT_MISSING_VALUE = -999.0

def apply_modality_masking(batch):
    """
    Augments the dataset to include different masked combinations.
    1. All modalities present
    2. Time-series and text present, image missing (Ts, missing, text)
    3. Time-series and image present, text missing (Ts, image, missing)
    4. Only Time-series present, text and image missing (Ts, missing, missing)

    Args: 
        batch (dict)
        - 'modality_series: (batch_size, 24, each modality data)
        - 'labels': (batch_size, 24)

    Returns:
        list of dict: 각 마스킹된 배치가 포함된 리스트
    """
    print("모달리티 마스킹 및 증강 시작.")
    masked_batches = []

    # 1. 원본 데이터 (Missing modality가 없음.)
    original_batch = copy.deepcopy(batch)
    masked_batches.append(original_batch)

    # 2. 이미지 모달리티 마스킹
    masked_cxr_batch = copy.deepcopy(batch)
    for series in masked_cxr_batch['modality_series']:
        for time_step in series:
            time_step['cxr_tensor'] = torch.full((3,224,224), DEFAULT_MISSING_VALUE, device=batch['labels'].device)
    masked_batches.append(masked_cxr_batch)

    # 3. 텍스트 모달리티 마스킹
    masked_text_batch = copy.deepcopy(batch)
    for series in masked_cxr_batch['modality_series']:
        for time_step in series:
            time_step['text_content'] = str(DEFAULT_MISSING_VALUE)
    masked_batches.append(masked_text_batch)

    # 4. 이미지 + 텍스트 마스킹
    masked_cxr_text_batch = copy.deepcopy(batch)
    for series in masked_cxr_batch['modality_series']:
        for time_step in series:
            time_step['cxr_tensor'] = torch.full((3,224,224), DEFAULT_MISSING_VALUE, device=batch['labels'].device)
            time_step['text_content'] = str(DEFAULT_MISSING_VALUE)
    masked_batches.append(masked_cxr_text_batch)

    return masked_batches