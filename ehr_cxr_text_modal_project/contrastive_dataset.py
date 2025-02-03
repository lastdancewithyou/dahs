# library import
import pandas as pd
import cv2
from tqdm import tqdm
import pickle
from pathlib import Path
import copy

from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

root_dir = 'C:/Users/gangmin/dahs/my research'
ts_df = pd.read_feather(root_dir + '/processed/ts_dataset.ftr')
img_df = pd.read_feather(root_dir + '/processed/cxr_dataset.ftr')
text_df = pd.read_feather(root_dir + '/processed/label_imputed_report.ftr')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# naming rule
"""
# time feature
hour_slot -> time_step

# modality features
image_data, image_tensor -> cxr_tensor
text, extracted_text -> text_content
ts_data -> ts_features

# batch data features
sequence -> sequences -> modality_series
labels_seq -> label_series

# modality flag
cxr_flag -> has_cxr
text_flag -> has_text
"""

class SCL_Multi_Dataset(Dataset): 
    def __init__(self, ts_df, img_df, text_df, alpha=0.03, device=None):
        
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Cuda check
        if torch.cuda.is_available():
            print(f"Dataset will use GPU: {torch.cuda.get_device_name(self.device)}")
        else:
            print("Dataset will use CPU")

        # dataframe
        self.ts_df = ts_df # ts_dataset
        self.ts_df.rename(columns={'Edema':"label"}, inplace=True)

        def to_tensor(data):
            data = pickle.loads(data)
            return torch.tensor(data, dtype=torch.float32).to(self.device)
        self.img_df = img_df[['stay_id', 'hour_slot', 'cxr_flag', 'processed_images', 'Edema']].copy() # cxr_dataset
        self.img_df['processed_images'] = self.img_df['processed_images'].map(to_tensor)
        self.img_df.rename(columns={'Edema':"label"}, inplace=True)


        self.text_df = text_df[['stay_id', 'hour_slot', 'text_flag', 'extracted_text', 'Edema']].copy() # label_imputed_report
        self.text_df.rename(columns={'Edema':"label"}, inplace=True)

        self.merged_df = (
            self.ts_df.merge(self.img_df, on=['stay_id', 'hour_slot', 'label'], how='outer').merge(self.text_df, on=['stay_id','hour_slot', 'label'], how='outer'))
        
        # stay_id grouping
        self.stay_groups = self.merged_df.groupby('stay_id')
        self.stay_ids = list(self.stay_groups.groups.keys())

    #######################################################################
    # stay_id grouping -> hour_slot iteration -> decay logic
    #######################################################################
    def __getitem__(self, idx): 
        """
        
        """
        stay_id = self.stay_ids[idx]
        stay_data = self.stay_groups.get_group(stay_id).sort_values('hour_slot')

        sequence_list = [] # modality data
        label_list = [] # answer label

        for _, row in stay_data.iterrows():
            sequence_list.append({
                'time_step': row['hour_slot'],
                'ts_features': row.drop(columns=['stay_id', 'hour_slot', 'processed_images', 'cxr_flag', 'extracted_text', 'text_flag']).values,
                'cxr_tensor': row['processed_images'],
                'has_cxr': row['cxr_flag'],
                'text_content': row['extracted_text'],
                'has_text': row['text_flag']
            })
            label_list.append(row['label'])

        return {
            'stay_id': stay_id, # patient index
            'modality_series': sequence_list, # main data for modeling
            'label_series': label_list # answer label for predict pulmonary edema
        }

    def __len__(self): 
        return len(self.stay_ids)
    
#######################################################################
# collate function, split_dataset, get_dataloaders
#######################################################################
def multimodal_collate_fn(batch): 
    """
    1. stay_id 내에서 time_step 순서 유지
    2. 개별 데이터 샘플이 섞이지 않도록 stay_id 단위로 묶어서 shuffle
    """
    stay_ids = [item['stay_id'] for item in batch]
    modality_series = [item['modality_series'] for item in batch]
    # labels = torch.stack([torch.tensor(item['label_series'], dtype=torch.long) for item in batch]) # (B, 24), batches are all same size
    # print(f"Labels shape: {labels.shape}")
    labels = torch.tensor([item['label_series'] for item in batch], dtype=torch.long)
    print(f"Labels shape: {labels.shape}")

    return {'stay_ids': stay_ids,
            'modality_series': modality_series, # 24hr data in each modalities
            'labels': labels # 24hr answer labels(Pulmonary Edema)
            }

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2):
    """
    Splits the dataset into training, validation and testing sets.
    """
    total_size = len(dataset)
    train_size = int(round(total_size * train_ratio))
    val_size = int(round(total_size * val_ratio))
    test_size = total_size - (train_size + val_size)
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def get_dataloaders(ts_df, img_df, text_df, batch_size=4):
    dataset = SCL_Multi_Dataset(ts_df=ts_df, img_df=img_df, text_df=text_df)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # dataloader define
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=multimodal_collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=multimodal_collate_fn)

    return train_dataloader, val_dataloader, test_dataloader

#######################################################################
# missing modality generator by masking
#######################################################################
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

    masked_batches = []

    # 1. 원본 데이터 (Missing modality가 없음.)
    masked_batches.append(copy.deepcopy(batch))

    # 2. 이미지 모달리티 마스킹
    masked_cxr_batch = copy.deepcopy(batch)
    for i in range(len(masked_cxr_batch['modality_series'])):
        for j in range(len(masked_cxr_batch['modality_series'][i])):
            masked_cxr_batch['modality_series'][i][j]['cxr_tensor'] = torch.zeros(256, device=batch['labels'].device)
    masked_batches.append(masked_cxr_batch)

    # 3. 텍스트 모달리티 마스킹
    masked_text_batch = copy.deepcopy(batch)
    for i in range(len(masked_text_batch['modality_series'])):
        for j in range(len(masked_text_batch['modality_series'][i])):
            masked_text_batch['modality_series'][i][j]['text_content'] = torch.zeros(256, device=batch['labels'].device)
    masked_batches.append(masked_text_batch)

    # 4. 이미지 + 텍스트 마스킹
    masked_cxr_text_batch = copy.deepcopy(batch)
    for i in range(len(masked_cxr_text_batch['modality_series'])):
        for j in range(len(masked_cxr_text_batch['modality_series'][i])):
            masked_cxr_text_batch["modality_series"][i][j]['cxr_tensor'] = torch.zeros(256, device=batch['labels'].device)
            masked_cxr_text_batch["modality_series"][i][j]['text_content'] = torch.zeros(256, device=batch['labels'].device)
    masked_batches.append(masked_cxr_text_batch)

    return masked_batches