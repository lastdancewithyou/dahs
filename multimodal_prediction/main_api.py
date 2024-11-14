# System                                                                                           
import os
import sys

# Base
import pickle
import numpy as np
import pandas as pd
import datetime as dt
from PIL import Image
import math
import warnings
import logging
from tqdm import tqdm
tqdm.pandas()
pd.options.mode.chained_assignment=None
warnings.filterwarnings("ignore")

from scipy.stats import ks_2samp
from scipy.signal import find_peaks

# Deep Learning
import torch
import cv2

# CLIP
import clip

# MIMIC IV Data Location
core_mimiciv_path = "C:/Users/gangmin/dahs/haim/data/"
# MIMIC-CXR Data Location
core_mimiciv_imgcxr_path = 'C:/Users/gangmin/dahs/data/physionet.org/files/mimic-cxr-jpg/2.1.0/'
# Chexpert Label
cxr_labels = [
    'Atelectasis',
    'Cardiomegaly',
    'Consolidation',
    'Edema',
    'Enlarged Cardiomediastinum',
    'Fracture',
    'Lung Lesion',
    'Lung Opacity',
    'No Finding',
    'Pleural Effusion',
    'Pleural Other',
    'Pneumonia',
    'Pneumothorax',
    'Support Devices'
]

logging.basicConfig(
    filename='check_unreadable_cxr_images.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

# load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

def load_mimiciv(core_mimiciv_path):

    ## CORE
    ### data load
    df_admissions = pd.read_parquet(core_mimiciv_path + 'core/admissions.parquet')
    df_patients = pd.read_parquet(core_mimiciv_path + 'core/patients.parquet')  
    df_transfers = pd.read_parquet(core_mimiciv_path + 'core/transfers.parquet')

    ### dtype
    df_admissions = df_admissions.astype({'admission_location': 'object','deathtime': 'object','edouttime': 'object','edregtime': 'object'})
    df_patients = df_patients.astype({'dod': 'object'})
    df_transfers = df_transfers.astype({'careunit': 'object'})

    ## HOSP
    ### data load
    df_d_labitems = pd.read_parquet(core_mimiciv_path + 'hosp/d_labitems.parquet')
    df_d_icd_procedures = pd.read_parquet(core_mimiciv_path + 'hosp/d_icd_procedures.parquet')
    df_d_icd_diagnoses = pd.read_parquet(core_mimiciv_path + 'hosp/d_icd_diagnoses.parquet')
    df_d_hcpcs = pd.read_parquet(core_mimiciv_path + 'hosp/d_hcpcs.parquet')
    df_diagnoses_icd = pd.read_parquet(core_mimiciv_path + 'hosp/diagnoses_icd.parquet')
    df_drgcodes = pd.read_parquet(core_mimiciv_path + 'hosp/drgcodes.parquet') # No need to change dtype 
    df_emar = pd.read_parquet(core_mimiciv_path + 'hosp/emar.parquet')# No need to change dtype
    df_emar_detail = pd.read_parquet(core_mimiciv_path + 'hosp/emar_detail.parquet')
    df_hcpcsevents = pd.read_parquet(core_mimiciv_path + 'hosp/hcpcsevents.parquet')
    df_labevents = pd.read_parquet(core_mimiciv_path + 'hosp/labevents.parquet')
    df_microbiologyevents = pd.read_parquet(core_mimiciv_path + 'hosp/microbiologyevents.parquet')
    df_poe = pd.read_parquet(core_mimiciv_path + 'hosp/poe.parquet')
    df_poe_detail = pd.read_parquet(core_mimiciv_path + 'hosp/poe_detail.parquet') # No need to change dtype
    df_prescriptions = pd.read_parquet(core_mimiciv_path + 'hosp/prescriptions.parquet')
    df_procedures_icd = pd.read_parquet(core_mimiciv_path + 'hosp/procedures_icd.parquet')
    df_services = pd.read_parquet(core_mimiciv_path + 'hosp/services.parquet')

    ### dtype
    df_d_labitems = df_d_labitems.astype({'itemid': 'object'})
    df_d_icd_procedures = df_d_icd_procedures.astype({'icd_code': 'object', 'icd_version': 'object'})
    df_d_icd_diagnoses = df_d_icd_diagnoses.astype({'icd_code': 'object', 'icd_version': 'object'})
    df_d_hcpcs = df_d_hcpcs.astype({'category': 'object'})
    df_diagnoses_icd = df_diagnoses_icd.astype({'icd_code': 'object', 'icd_version': 'object'})
    df_emar_detail = df_emar_detail.astype({'completion_interval': 'object','dose_due': 'object','dose_given': 'object','infusion_complete': 'object','infusion_rate_adjustment': 'object','infusion_rate_unit': 'object','new_iv_bag_hung': 'object','product_description_other': 'object','reason_for_no_barcode': 'object','restart_interval': 'object','route': 'object','side': 'object','site': 'object','continued_infusion_in_other_location': 'object','infusion_rate': 'object','non_formulary_visual_verification': 'object','prior_infusion_rate': 'object','product_amount_given': 'object', 'infusion_rate_adjustment_amount': 'object'})
    df_hcpcsevents = df_hcpcsevents.astype({'hcpcs_cd': 'object'})
    df_labevents = df_labevents.astype({'storetime': 'object', 'value': 'object', 'valueuom': 'object', 'flag': 'object', 'priority': 'object', 'comments': 'object'})
    df_microbiologyevents = df_microbiologyevents.astype({'comments': 'object', 'quantity': 'object'})
    df_poe = df_poe.astype({'discontinue_of_poe_id': 'object','discontinued_by_poe_id': 'object','order_status': 'object'})
    df_prescriptions = df_prescriptions.astype({'form_rx': 'object','gsn': 'object'})
    df_procedures_icd = df_procedures_icd.astype({'icd_code': 'object', 'icd_version': 'object'})
    df_services = df_services.astype({'prev_service': 'object'})

    ## ICU
    ### data load
    df_d_items = pd.read_parquet(core_mimiciv_path + 'icu/d_items.parquet')
    df_procedureevents = pd.read_parquet(core_mimiciv_path + 'icu/procedureevents.parquet')
    df_outputevents = pd.read_parquet(core_mimiciv_path + 'icu/outputevents.parquet')
    df_inputevents = pd.read_parquet(core_mimiciv_path + 'icu/inputevents.parquet')
    df_icustays = pd.read_parquet(core_mimiciv_path + 'icu/icustays.parquet')
    df_datetimeevents = pd.read_parquet(core_mimiciv_path + 'icu/datetimeevents.parquet')
    df_chartevents = pd.read_parquet(core_mimiciv_path + 'icu/chartevents.parquet')

    ### dtype
    df_procedureevents = df_procedureevents.astype({'value': 'object'})
    df_outputevents = df_outputevents.astype({'value': 'object'})
    df_inputevents = df_inputevents.astype({'amount': 'object', 'totalamountuom': 'object'})
    df_datetimeevents = df_datetimeevents.astype({'value': 'object'})
    df_chartevents = df_chartevents.astype({'value': 'object', 'valueuom': 'object'})

    ## CXR
    ### data load
    df_mimic_cxr_split = pd.read_parquet(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.parquet')
    df_mimic_cxr_chexpert = pd.read_parquet(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.parquet')
    df_mimic_cxr_metadata = pd.read_parquet(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.parquet')

    ### dtype
    df_mimic_cxr_metadata = df_mimic_cxr_metadata.astype({'dicom_id': 'object'})    

    ## NOTES (Disable notes_df when using CLIP Model)
    ### data load
    df_radnotes = pd.read_parquet(core_mimiciv_path + "mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-radiology_notes.parquet")

    # Data Preprocessing (Change the date column of each time to datetime)
    ## CORE
    print("MIMIC-IV CORE PREPROCESSING")
    
    ### admissions
    df_admissions['admittime'] = pd.to_datetime(df_admissions['admittime'])
    df_admissions['dischtime'] = pd.to_datetime(df_admissions['dischtime'])
    df_admissions['deathtime'] = pd.to_datetime(df_admissions['deathtime'])
    df_admissions['edregtime'] = pd.to_datetime(df_admissions['edregtime'])
    df_admissions['edouttime'] = pd.to_datetime(df_admissions['edouttime'])

    ### transfers
    df_transfers['intime'] = pd.to_datetime(df_transfers['intime'])
    df_transfers['outtime'] = pd.to_datetime(df_transfers['outtime'])

    ## HOSP
    print("MIMIC-IV HOSP PREPROCESSING")
    df_diagnoses_icd.icd_code = df_diagnoses_icd.icd_code.str.strip()
    df_diagnoses_icd.icd_version = df_diagnoses_icd.icd_version.astype(str).str.strip()
    df_d_icd_diagnoses.icd_code = df_d_icd_diagnoses.icd_code.str.strip()
    df_d_icd_diagnoses.icd_version = df_d_icd_diagnoses.icd_version.astype(str).str.strip()

    df_procedures_icd.icd_code = df_procedures_icd.icd_code.str.strip()
    df_procedures_icd.icd_version = df_procedures_icd.icd_version.astype(str).str.strip()
    df_d_icd_procedures.icd_code = df_d_icd_procedures.icd_code.str.strip()
    df_d_icd_procedures.icd_version = df_d_icd_procedures.icd_version.astype(str).str.strip()

    df_hcpcsevents.hcpcs_cd = df_hcpcsevents.hcpcs_cd.str.strip()
    df_d_hcpcs.code = df_d_hcpcs.code.str.strip()

    df_prescriptions['starttime'] = pd.to_datetime(df_prescriptions['starttime'])
    df_prescriptions['stoptime'] = pd.to_datetime(df_prescriptions['stoptime'])

    df_emar['charttime'] = pd.to_datetime(df_emar['charttime'])
    df_emar['scheduletime'] = pd.to_datetime(df_emar['scheduletime'])
    df_emar['storetime'] = pd.to_datetime(df_emar['storetime'])

    df_labevents['charttime'] = pd.to_datetime(df_labevents['charttime'])
    df_labevents['storetime'] = pd.to_datetime(df_labevents['storetime'])
    
    df_microbiologyevents['chartdate'] = pd.to_datetime(df_microbiologyevents['chartdate'])
    df_microbiologyevents['charttime'] = pd.to_datetime(df_microbiologyevents['charttime'])
    df_microbiologyevents['storedate'] = pd.to_datetime(df_microbiologyevents['storedate'])
    df_microbiologyevents['storetime'] = pd.to_datetime(df_microbiologyevents['storetime'])

    df_poe['ordertime'] = pd.to_datetime(df_poe['ordertime'])
    df_services['transfertime'] = pd.to_datetime(df_services['transfertime'])

    ## ICU
    print("MIMIC-IV ICU PREPROCESSING")
    df_procedureevents['starttime'] = pd.to_datetime(df_procedureevents['starttime'])
    df_procedureevents['endtime'] = pd.to_datetime(df_procedureevents['endtime'])
    df_procedureevents['storetime'] = pd.to_datetime(df_procedureevents['storetime'], errors='coerce')
    
    df_outputevents['charttime'] = pd.to_datetime(df_outputevents['charttime'])
    df_outputevents['storetime'] = pd.to_datetime(df_outputevents['storetime'], errors='coerce')
    
    df_inputevents['starttime'] = pd.to_datetime(df_inputevents['starttime'])
    df_inputevents['endtime'] = pd.to_datetime(df_inputevents['endtime'])
    df_inputevents['storetime'] = pd.to_datetime(df_inputevents['storetime'], errors='coerce')
    
    df_icustays['intime'] = pd.to_datetime(df_icustays['intime'])
    df_icustays['outtime'] = pd.to_datetime(df_icustays['outtime'])
    
    df_datetimeevents['charttime'] = pd.to_datetime(df_datetimeevents['charttime'])
    df_datetimeevents['storetime'] = pd.to_datetime(df_datetimeevents['storetime'], errors='coerce')
    
    df_chartevents['charttime'] = pd.to_datetime(df_chartevents['charttime'])
    df_chartevents['storetime'] = pd.to_datetime(df_chartevents['storetime'], errors='coerce')

    ## CXR
    print("MIMIC-CXR PREPROCESSING")
    if (not 'cxrtime' in df_mimic_cxr_metadata.columns) or (not 'Img_Filename' in df_mimic_cxr_metadata.columns):

        # Create CXRTime column if it does not exist already
        print("Processing CXR-Time stamps")
        df_cxr = df_mimic_cxr_metadata.copy()
        df_cxr['StudyDateForm'] = pd.to_datetime(df_cxr['StudyDate'], format='%Y%m%d')
        df_cxr['StudyTimeForm'] = df_cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
        df_cxr['StudyTimeForm'] = pd.to_datetime(df_cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time
        df_cxr['cxrtime'] = df_cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)

        # Add paths and info to images in cxr
        df_mimic_cxr_jpg = pd.read_parquet(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-jpeg-txt.parquet')
        df_cxr = pd.merge(df_mimic_cxr_jpg, df_cxr, on='dicom_id')

        # Save and Reload
        df_cxr.to_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
        df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
    else: 
        df_mimic_cxr_metadata = pd.read_csv(core_mimiciv_path + 'mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')

    df_mimic_cxr_metadata['cxrtime'] = pd.to_datetime(df_mimic_cxr_metadata['cxrtime'])
    # mimic_cxr(main data of cxr)
    mimic_cxr = df_mimic_cxr_metadata[['dicom_id', 'subject_id', 'study_id', 'ViewPosition', 'cxrtime', 'StudyDate', 'StudyTime']]
    mimic_cxr = mimic_cxr[(mimic_cxr['ViewPosition']=='PA') | (mimic_cxr['ViewPosition']=='AP')]
    image_filenames_path = core_mimiciv_imgcxr_path + 'IMAGE_FILENAMES'

    df_mimic_cxr_image = pd.DataFrame()
    with open(image_filenames_path, 'r') as file: 
        lines = file.readlines()

    pbar = tqdm(total=len(lines), desc="Processing image files")
    rows = []

    for line in lines:
        line = line.strip()
        img_folder, img_filename = line.rsplit('/', 1) 
        dicom_id = img_filename.split('.')[0]

        rows.append({
            'Img_Folder': img_folder, 
            'Img_Filename': img_filename, 
            'dicom_id': dicom_id
        })

        pbar.update(1)

    pbar.close()

    df_mimic_cxr_image = pd.concat([df_mimic_cxr_image, pd.DataFrame(rows)], ignore_index=True)
    df_mimic_cxr = mimic_cxr.merge(df_mimic_cxr_image, how='left', on='dicom_id')
    df_mimic_cxr['hadm_id'] = df_mimic_cxr.apply(lambda row: assign_hadm_id(row, df_transfers), axis=1)
    df_mimic_cxr = df_mimic_cxr.dropna(subset=['hadm_id'])

    ## NOTES
    print("NOTES PREPROCESSING")
    df_radnotes['chartime'] = pd.to_datetime(df_radnotes['charttime'])
    df_radnotes['chartime'] = pd.to_datetime(df_radnotes['storetime'])

    # Data Sorting
    ## CORE
    print("SORTING CORE")
    df_admissions = df_admissions.sort_values(by=['subject_id','hadm_id'])
    df_patients = df_patients.sort_values(by=['subject_id'])
    df_transfers = df_transfers.sort_values(by=['subject_id','hadm_id'])

    ## HOSP
    print("SORTING HOSP")
    df_diagnoses_icd = df_diagnoses_icd.sort_values(by=['subject_id'])
    df_drgcodes = df_drgcodes.sort_values(by=['subject_id','hadm_id'])
    df_emar = df_emar.sort_values(by=['subject_id','hadm_id'])
    df_emar_detail = df_emar_detail.sort_values(by=['subject_id'])
    df_hcpcsevents = df_hcpcsevents.sort_values(by=['subject_id','hadm_id'])
    df_labevents = df_labevents.sort_values(by=['subject_id','hadm_id'])
    df_microbiologyevents = df_microbiologyevents.sort_values(by=['subject_id','hadm_id'])
    df_poe = df_poe.sort_values(by=['subject_id','hadm_id'])
    df_poe_detail = df_poe_detail.sort_values(by=['subject_id'])
    df_prescriptions = df_prescriptions.sort_values(by=['subject_id','hadm_id'])
    df_procedures_icd = df_procedures_icd.sort_values(by=['subject_id','hadm_id'])
    df_services = df_services.sort_values(by=['subject_id','hadm_id'])

    ## ICU
    print("SORTING ICU")
    df_procedureevents = df_procedureevents.sort_values(by=['subject_id','hadm_id','stay_id'])
    df_outputevents = df_outputevents.sort_values(by=['subject_id','hadm_id','stay_id'])
    df_inputevents = df_inputevents.sort_values(by=['subject_id','hadm_id','stay_id'])
    df_icustays = df_icustays.sort_values(by=['subject_id','hadm_id','stay_id'])
    df_datetimeevents = df_datetimeevents.sort_values(by=['subject_id','hadm_id','stay_id'])
    df_chartevents = df_chartevents.sort_values(by=['subject_id','hadm_id','stay_id'])

    ## CXR
    print("SORTING CXR")
    df_mimic_cxr_split = df_mimic_cxr_split.sort_values(by=['subject_id'])
    df_mimic_cxr = df_mimic_cxr.sort_values(by=['subject_id', 'hadm_id'])

    ## NOTES
    df_radnotes = df_radnotes.sort_values(by=['subject_id','hadm_id'])

    print("df_base_core created")
    df_base_core = df_admissions.merge(df_patients, how='left').merge(df_transfers, how='left')
    df_base_core.to_csv(core_mimiciv_path + 'core/core.csv', index=False)
    df_base_core = pd.read_csv(core_mimiciv_path + 'core/core.csv')

    print("Preprocessing process completed")
    dataframes = {
        'df_base_core': df_base_core,
        'df_admissions': df_admissions,
        'df_patients': df_patients,
        'df_transfers': df_transfers,
        'df_diagnoses_icd': df_diagnoses_icd,
        'df_drgcodes': df_drgcodes,
        'df_emar': df_emar,
        'df_emar_detail': df_emar_detail,
        'df_hcpcsevents': df_hcpcsevents,
        'df_labevents': df_labevents,
        'df_microbiologyevents': df_microbiologyevents,
        'df_poe': df_poe,
        'df_poe_detail': df_poe_detail,
        'df_prescriptions': df_prescriptions,
        'df_procedures_icd': df_procedures_icd,
        'df_services': df_services,
        'df_d_icd_diagnoses': df_d_icd_diagnoses,
        'df_d_icd_procedures': df_d_icd_procedures,
        'df_d_hcpcs': df_d_hcpcs,
        'df_d_labitems': df_d_labitems,
        'df_procedureevents': df_procedureevents,
        'df_outputevents': df_outputevents,
        'df_inputevents': df_inputevents,
        'df_icustays': df_icustays,
        'df_datetimeevents': df_datetimeevents,
        'df_chartevents': df_chartevents,
        'df_d_items': df_d_items,
        'df_mimic_cxr_split': df_mimic_cxr_split,
        'df_mimic_cxr': df_mimic_cxr,
        'df_radnotes': df_radnotes
    }

    return dataframes

def assign_hadm_id(row, df_transfers):
    transfer_records = df_transfers[df_transfers['subject_id'] == row['subject_id']]
    
    for _, record in transfer_records.iterrows():
        if record['intime'] <= row['cxrtime'] <= record['outtime']:
            return record['hadm_id']
    return None

class Patient_ICU(object):
    def __init__(self, admissions, demographics, transfers, core,\
        diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents,\
        labevents, microbiologyevents, poe, poe_detail,\
        prescriptions, procedures_icd, services, procedureevents,\
        outputevents, inputevents, icustays, datetimeevents,\
        chartevents, cxr, imcxr, radnotes):
        
        ## CORE
        self.admissions = admissions
        self.demographics = demographics
        self.transfers = transfers
        self.core = core
        ## HOSP
        self.diagnoses_icd = diagnoses_icd
        self.drgcodes = drgcodes
        self.emar = emar
        self.emar_detail = emar_detail
        self.hcpcsevents = hcpcsevents
        self.labevents = labevents
        self.microbiologyevents = microbiologyevents
        self.poe = poe
        self.poe_detail = poe_detail
        self.prescriptions = prescriptions
        self.procedures_icd = procedures_icd
        self.services = services
        ## ICU
        self.procedureevents = procedureevents
        self.outputevents = outputevents
        self.inputevents = inputevents
        self.icustays = icustays
        self.datetimeevents = datetimeevents
        self.chartevents = chartevents
        ## CXR
        self.cxr = cxr
        self.imcxr = imcxr
        ## NOTES
        self.radnotes = radnotes


def get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id, mimic_data):

    df_base_core = mimic_data['df_base_core']
    df_admissions = mimic_data['df_admissions']
    df_patients = mimic_data['df_patients']
    df_transfers = mimic_data['df_transfers']
    df_diagnoses_icd = mimic_data['df_diagnoses_icd']
    df_drgcodes = mimic_data['df_drgcodes']
    df_emar = mimic_data['df_emar']
    df_emar_detail = mimic_data['df_emar_detail']
    df_hcpcsevents = mimic_data['df_hcpcsevents']
    df_labevents = mimic_data['df_labevents']
    df_microbiologyevents = mimic_data['df_microbiologyevents']
    df_poe = mimic_data['df_poe']
    df_poe_detail = mimic_data['df_poe_detail']
    df_prescriptions = mimic_data['df_prescriptions']
    df_procedures_icd = mimic_data['df_procedures_icd']
    df_services = mimic_data['df_services']
    df_d_icd_diagnoses = mimic_data['df_d_icd_diagnoses']
    df_d_icd_procedures = mimic_data['df_d_icd_procedures']
    df_d_hcpcs = mimic_data['df_d_hcpcs']
    df_d_labitems = mimic_data['df_d_labitems']
    df_procedureevents = mimic_data['df_procedureevents']
    df_outputevents = mimic_data['df_outputevents']
    df_inputevents = mimic_data['df_inputevents']
    df_icustays = mimic_data['df_icustays']
    df_datetimeevents = mimic_data['df_datetimeevents']
    df_chartevents = mimic_data['df_chartevents']
    df_d_items = mimic_data['df_d_items']
    df_mimic_cxr_split = mimic_data['df_mimic_cxr_split']
    df_mimic_cxr = mimic_data['df_mimic_cxr']
    df_radnotes = mimic_data['df_radnotes']

    # Filter data
    ## CORE
    f_df_base_core = df_base_core[(df_base_core.subject_id == key_subject_id) & (df_base_core.hadm_id == key_hadm_id)]
    f_df_admissions = df_admissions[(df_admissions.subject_id == key_subject_id) & (df_admissions.hadm_id == key_hadm_id)]
    f_df_patients = df_patients[(df_patients.subject_id == key_subject_id)]
    f_df_transfers = df_transfers[(df_transfers.subject_id == key_subject_id) & (df_transfers.hadm_id == key_hadm_id)]

    ##-> Merge data into single patient structure
    f_df_core = f_df_base_core
    f_df_core['admittime'] = pd.to_datetime(f_df_core['admittime'])
    f_df_core['dischtime'] = pd.to_datetime(f_df_core['dischtime'])
    f_df_core['deathtime'] = pd.to_datetime(f_df_core['deathtime'])
    f_df_core['edregtime'] = pd.to_datetime(f_df_core['edregtime'])
    f_df_core['edouttime'] = pd.to_datetime(f_df_core['edouttime'])

    f_df_core = f_df_core.merge(f_df_admissions, how='left')
    f_df_core = f_df_core.merge(f_df_patients, how='left')
    f_df_core['intime'] = pd.to_datetime(f_df_core['intime'])
    f_df_core['outtime'] = pd.to_datetime(f_df_core['outtime'])
    f_df_core = f_df_core.merge(f_df_transfers, how='left')

    ##-> HOSP
    f_df_diagnoses_icd = df_diagnoses_icd[(df_diagnoses_icd.subject_id == key_subject_id)]
    f_df_drgcodes = df_drgcodes[(df_drgcodes.subject_id == key_subject_id) & (df_drgcodes.hadm_id == key_hadm_id)]
    f_df_emar = df_emar[(df_emar.subject_id == key_subject_id) & (df_emar.hadm_id == key_hadm_id)]
    f_df_emar_detail = df_emar_detail[(df_emar_detail.subject_id == key_subject_id)]
    f_df_hcpcsevents = df_hcpcsevents[(df_hcpcsevents.subject_id == key_subject_id) & (df_hcpcsevents.hadm_id == key_hadm_id)]
    f_df_labevents = df_labevents[(df_labevents.subject_id == key_subject_id) & (df_labevents.hadm_id == key_hadm_id)]
    f_df_microbiologyevents = df_microbiologyevents[(df_microbiologyevents.subject_id == key_subject_id) & (df_microbiologyevents.hadm_id == key_hadm_id)]
    f_df_poe = df_poe[(df_poe.subject_id == key_subject_id) & (df_poe.hadm_id == key_hadm_id)]
    f_df_poe_detail = df_poe_detail[(df_poe_detail.subject_id == key_subject_id)]
    f_df_prescriptions = df_prescriptions[(df_prescriptions.subject_id == key_subject_id) & (df_prescriptions.hadm_id == key_hadm_id)]
    f_df_procedures_icd = df_procedures_icd[(df_procedures_icd.subject_id == key_subject_id) & (df_procedures_icd.hadm_id == key_hadm_id)]
    f_df_services = df_services[(df_services.subject_id == key_subject_id) & (df_services.hadm_id == key_hadm_id)]
    ###-> Merge content from dictionaries
    f_df_diagnoses_icd = f_df_diagnoses_icd.merge(df_d_icd_diagnoses, how='left') 
    f_df_procedures_icd = f_df_procedures_icd.merge(df_d_icd_procedures, how='left')
    f_df_hcpcsevents = f_df_hcpcsevents.merge(df_d_hcpcs, how='left')
    f_df_labevents = f_df_labevents.merge(df_d_labitems, how='left')

    ##-> ICU
    f_df_procedureevents = df_procedureevents[(df_procedureevents.subject_id == key_subject_id) & (df_procedureevents.hadm_id == key_hadm_id) & (df_procedureevents.stay_id == key_stay_id)]
    f_df_outputevents = df_outputevents[(df_outputevents.subject_id == key_subject_id) & (df_outputevents.hadm_id == key_hadm_id) & (df_outputevents.stay_id == key_stay_id)]
    f_df_inputevents = df_inputevents[(df_inputevents.subject_id == key_subject_id) & (df_inputevents.hadm_id == key_hadm_id) & (df_inputevents.stay_id == key_stay_id)]
    f_df_icustays = df_icustays[(df_icustays.subject_id == key_subject_id) & (df_icustays.hadm_id == key_hadm_id) & (df_icustays.stay_id == key_stay_id)]
    f_df_datetimeevents = df_datetimeevents[(df_datetimeevents.subject_id == key_subject_id) & (df_datetimeevents.hadm_id == key_hadm_id) & (df_datetimeevents.stay_id == key_stay_id)]
    f_df_chartevents = df_chartevents[(df_chartevents.subject_id == key_subject_id) & (df_chartevents.hadm_id == key_hadm_id) & (df_chartevents.stay_id == key_stay_id)]
    ###-> Merge content from dictionaries
    f_df_procedureevents = f_df_procedureevents.merge(df_d_items, how='left')
    f_df_outputevents = f_df_outputevents.merge(df_d_items, how='left')
    f_df_inputevents = f_df_inputevents.merge(df_d_items, how='left')
    f_df_datetimeevents = f_df_datetimeevents.merge(df_d_items, how='left')
    f_df_chartevents = f_df_chartevents.merge(df_d_items, how='left')
    # print("df_core subject_id", f_df_core['subject_id'].iloc[0])
    # print("df_core hadm_id", f_df_core['hadm_id'].iloc[0])

    ##-> CXR
    df_mimic_cxr['hadm_id'] = df_mimic_cxr['hadm_id'].astype(np.int64)
    f_df_mimic_cxr = df_mimic_cxr[(df_mimic_cxr.subject_id == key_subject_id) & (df_mimic_cxr.hadm_id == key_hadm_id)]
    f_df_mimic_cxr = df_mimic_cxr[(df_mimic_cxr.subject_id == key_subject_id)]
    # print("mimic_cxr subject_id", f_df_transfers['subject_id'].iloc[0])
    # print("mimic_cxr hadm_id", f_df_transfers['hadm_id'].iloc[0])
    f_df_mimic_cxr_split = df_mimic_cxr_split[(df_mimic_cxr_split.subject_id == key_subject_id)]

    ###-> Merge data into single patient structure
    f_df_cxr = f_df_mimic_cxr
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr, how='left')
    f_df_cxr = f_df_cxr.merge(f_df_mimic_cxr_split, how='left')

    empty_image_count = 0
    f_df_imcxr = []
    for img_idx, img_row in f_df_cxr.iterrows():
        img_path = core_mimiciv_imgcxr_path + str(img_row['Img_Folder']) + '/' + str(img_row['Img_Filename'])

        if os.path.exists(img_path): 
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None and img.size>0: 
                # img quality will be controlled later
                img_cxr_shape = [224, 224]
                img_cxr = cv2.resize(img, (img_cxr_shape[0], img_cxr_shape[1]))
                f_df_imcxr.append(np.array(img_cxr))
            else: 
                empty_image_count += 1
        else: 
            empty_image_count += 1

    logging.info(f"Empty image is counted for subject_id {key_subject_id}, hadm_id {key_hadm_id}: {empty_image_count}")

    ##-> NOTES
    f_df_radnotes = df_radnotes[(df_radnotes.subject_id == key_subject_id) & (df_radnotes.hadm_id == key_hadm_id)]

    ## CORE
    admissions = f_df_admissions
    demographics = f_df_patients
    transfers = f_df_transfers
    core = f_df_core

    ## HOSP
    diagnoses_icd = f_df_diagnoses_icd
    drgcodes = f_df_diagnoses_icd
    emar = f_df_emar
    emar_detail = f_df_emar_detail
    hcpcsevents = f_df_hcpcsevents
    labevents = f_df_labevents
    microbiologyevents = f_df_microbiologyevents
    poe = f_df_poe
    poe_detail = f_df_poe_detail
    prescriptions = f_df_prescriptions
    procedures_icd = f_df_procedures_icd
    services = f_df_services

    ## ICU
    procedureevents = f_df_procedureevents
    outputevents = f_df_outputevents
    inputevents = f_df_inputevents
    icustays = f_df_icustays
    datetimeevents = f_df_datetimeevents
    chartevents = f_df_chartevents

    ## CXR
    cxr = f_df_cxr
    imcxr = f_df_imcxr

    ## NOTES
    radnotes = f_df_radnotes

    Patient_ICUstay = Patient_ICU(admissions, demographics, transfers, core, \
                                diagnoses_icd, drgcodes, emar, emar_detail, hcpcsevents, \
                                labevents, microbiologyevents, poe, poe_detail, \
                                prescriptions, procedures_icd, services, procedureevents, \
                                outputevents, inputevents, icustays, datetimeevents, \
                                chartevents, cxr, imcxr, radnotes)
    
    return Patient_ICUstay

# DELTA TIME CALCULATOR FROM TWO TIMESTAMPS
def date_diff_hrs(t1, t0):
    # Inputs:
    #   t1 -> Final timestamp in a patient hospital stay
    #   t0 -> Initial timestamp in a patient hospital stay

    # Outputs:
    #   delta_t -> Patient stay structure bounded by allowed timestamps

    try:
        delta_t = (t1-t0).total_seconds()/3600 # Result in hrs
    except:
        delta_t = math.nan
        
    return delta_t

def get_timebound_patient_icustay(Patient_ICUstay, start_hr = None, end_hr = None):
    ## --> Process Event Structure Calculations
    admittime = Patient_ICUstay.core['admittime'].values[0]
    dischtime = Patient_ICUstay.core['dischtime'].values[0]

    Patient_ICUstay.emar.loc[:, 'deltacharttime'] = Patient_ICUstay.emar.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.labevents.loc[:, 'deltacharttime'] = Patient_ICUstay.labevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.microbiologyevents.loc[:, 'deltacharttime'] = Patient_ICUstay.microbiologyevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.outputevents.loc[:, 'deltacharttime'] = Patient_ICUstay.outputevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.datetimeevents.loc[:, 'deltacharttime'] = Patient_ICUstay.datetimeevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.chartevents.loc[:, 'deltacharttime'] = Patient_ICUstay.chartevents.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)
    Patient_ICUstay.radnotes['deltacharttime'] = Patient_ICUstay.radnotes.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)

    # Re-calculate times of CXR database
    Patient_ICUstay.cxr['StudyDateForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyDate'], format='%Y%m%d')
    Patient_ICUstay.cxr['StudyTimeForm'] = Patient_ICUstay.cxr.apply(lambda x : '%#010.3f' % x['StudyTime'] ,1)
    Patient_ICUstay.cxr['StudyTimeForm'] = pd.to_datetime(Patient_ICUstay.cxr['StudyTimeForm'], format='%H%M%S.%f').dt.time

    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr.apply(lambda r : dt.datetime.combine(r['StudyDateForm'],r['StudyTimeForm']),1)
    Patient_ICUstay.cxr['charttime'] = Patient_ICUstay.cxr['charttime'].dt.floor('Min')
    Patient_ICUstay.cxr.loc[:, 'deltacharttime'] = Patient_ICUstay.cxr.apply(lambda x: date_diff_hrs(x['charttime'],admittime) if not x.empty else None, axis=1)

    
    ## --> Filter by allowable time stamps
    if not (start_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime >= start_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]
    
        
    if not (end_hr == None):
        Patient_ICUstay.emar = Patient_ICUstay.emar[(Patient_ICUstay.emar.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.emar.deltacharttime)]
        Patient_ICUstay.labevents = Patient_ICUstay.labevents[(Patient_ICUstay.labevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.labevents.deltacharttime)]
        Patient_ICUstay.microbiologyevents = Patient_ICUstay.microbiologyevents[(Patient_ICUstay.microbiologyevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.microbiologyevents.deltacharttime)]
        Patient_ICUstay.outputevents = Patient_ICUstay.outputevents[(Patient_ICUstay.outputevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.outputevents.deltacharttime)]
        Patient_ICUstay.datetimeevents = Patient_ICUstay.datetimeevents[(Patient_ICUstay.datetimeevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.datetimeevents.deltacharttime)]
        Patient_ICUstay.chartevents = Patient_ICUstay.chartevents[(Patient_ICUstay.chartevents.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.chartevents.deltacharttime)]
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)]
        Patient_ICUstay.imcxr = [Patient_ICUstay.imcxr[i] for i, x in enumerate((Patient_ICUstay.cxr.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.cxr.deltacharttime)) if x]
        Patient_ICUstay.radnotes = Patient_ICUstay.radnotes[(Patient_ICUstay.radnotes.deltacharttime <= end_hr) | pd.isnull(Patient_ICUstay.radnotes.deltacharttime)]


        # Filter CXR to match allowable patient stay
        Patient_ICUstay.cxr = Patient_ICUstay.cxr[(Patient_ICUstay.cxr.charttime <= dischtime)]

    return Patient_ICUstay

def extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr, mimic_data):
    key_subject_id = df_haim_ids.iloc[haim_patient_idx].subject_id
    key_hadm_id = df_haim_ids.iloc[haim_patient_idx].hadm_id
    key_stay_id = df_haim_ids.iloc[haim_patient_idx].stay_id
    start_hr = start_hr # Select timestamps
    end_hr = end_hr   # Select timestamps
    patient = get_patient_icustay(key_subject_id, key_hadm_id, key_stay_id, mimic_data)
    dt_patient = get_timebound_patient_icustay(patient, start_hr , end_hr)
    
    return key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient

def save_patient_object(obj, filepath):

    directory = core_mimiciv_path + 'pickle/'

    if not os.path.exists(directory): 
        os.makedirs(directory)
    full_filepath = os.path.join(directory, filepath)
    
    with open(full_filepath, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def generate_all_mimiciv_patient_object(df_haim_ids, core_mimiciv_path, mimic_data):
    nfiles = len(df_haim_ids)
    with tqdm(total = nfiles) as pbar:
        #Iterate through all patients
        for haim_patient_idx in range(nfiles):
            # Let's select each single patient and extract patient object
            start_hr = None # Select timestamps
            end_hr = None   # Select timestamps
            key_subject_id, key_hadm_id, key_stay_id, patient, dt_patient = extract_single_patient_records_mimiciv(haim_patient_idx, df_haim_ids, start_hr, end_hr, mimic_data)
            
            # Save
            filename = f"{haim_patient_idx:08d}" + '.pkl'
            save_patient_object(dt_patient, core_mimiciv_path + 'pickle/' + filename)
            # Update process bar
            pbar.update(1)
            
    return nfiles

# load each patient's pickle file
def load_patient_object(filepath):

    with open(filepath, 'rb') as input: 
        return pickle.load(input)

# Embedding
## demographics embedding

def height_handling(x):
    
    return x.loc[cond, 'valuenum'].mean()

def get_demographics(dt_patient):
    dem_info = dt_patient.demographics[['gender', 'anchor_age', 'anchor_year']] 
    dem_info['gender'] = (dem_info['gender'] == 'M').astype(int)
    return dem_info.values[0]

def get_demographic_embeddings(dt_patient, verbose=0):
    
    
    demo_embeddings =  dt_patient.core.loc[0, ['anchor_age', 'gender_int', 'race_int']]
    

    if verbose >= 1:
        print(demo_embeddings)

    demo_embeddings = demo_embeddings.values

    return demo_embeddings

## time-series embedding
def get_ts_embeddings(dt_patient, event_type):
    event_frequencies = pd.read_csv(core_mimiciv_path + 'haim_patients_events.csv')
    event_frequencies = event_frequencies.groupby("event_type", group_keys=False).apply(lambda x: x.nlargest(25, 'count'))

    event_mapping = {
        'procedure': 'procedureevents',
        'chart': 'chartevents',
        'medication': 'inputevents',
        'lab': 'labevents'
    }

    if event_type not in event_mapping: 
        raise ValueError(f'Invalid event type: {event_type}')

    mapped_event_type = event_mapping[event_type]
    df = getattr(dt_patient, mapped_event_type)

    if mapped_event_type in event_frequencies['event_type'].values: 
        event_list = event_frequencies[event_frequencies['event_type']==mapped_event_type]['label'].tolist()
        print(event_list)
    else: 
        raise ValueError(f"No event frequencies provided for event type: {event_type}")
    
    if event_type == 'procedure': 
        df_pivot = pivot_procedureevent(df, event_list)

    elif event_type == 'chart': 
        df_pivot = pivot_chartevent(df, event_list)

    elif event_type == 'medication': 
        df_pivot = pivot_inputevent(df, event_list)

    elif event_type == 'lab': 
        df_pivot = pivot_labevent(df, event_list)

    ts_emb = get_ts_emb(df_pivot, event_list)
    try:
        ts_emb = ts_emb.drop(['subject_id', 'hadm_id']).fillna(value=0)
    except:
        ts_emb = pd.Series(0, index=ts_emb.columns).drop(['subject_id', 'hadm_id']).fillna(value=0)

    return ts_emb

def remove_low_freq(df, columm, percentage):
    threshold = df['count'].quantile(0.05)
    filtered_df = df[df['count']>threshold]
    return filtered_df

def pivot_procedureevent(df, event_list):
    df1 = df[['subject_id', 'hadm_id',  'storetime']] 
    for event in event_list: 
        df1[event] = np.nan
        #search in the label column 
        df1.loc[(df['label']==event), event] = df['value'].astype(float)  #Yu: maybe if not label use abbreviation 
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out

def pivot_chartevent(df, event_list):
    df1 = df[['subject_id', 'hadm_id', 'stay_id', 'charttime']] 
    for event in event_list: 
        df1[event] = np.nan
         #search in the abbreviations column  
        df1.loc[(df['label']==event), event] = df['valuenum'].astype(float)
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out 

def pivot_inputevent(df, event_list): 
    df1 = df[['subject_id', 'hadm_id', 'storetime']]
    for event in event_list:
        df1[event]=np.nan
        df1.loc[(df['label']==event), event] = df['amount'].astype(float)
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out

def pivot_labevent(df, event_list):
    df1 = df[['subject_id', 'hadm_id',  'charttime']] 
    for event in event_list: 
        df1[event] = np.nan
        #search in the label column 
        df1.loc[(df['label']==event), event] = df['valuenum'].astype(float) 
    df_out = df1.dropna(axis=0, how='all', subset=event_list)
    return df_out 

def get_ts_emb(df_pivot, event_list):
    try:
        df_out = df_pivot[['subject_id', 'hadm_id']].iloc[0]
    except:
#         print(df_pivot)
        df_out = pd.DataFrame(columns = ['subject_id', 'hadm_id'])
#         df_out = df_pivot[['subject_id', 'hadm_id']]

    row = pd.Series(0, index=df_pivot.columns)
    df_pivot = pd.concat([df_pivot, pd.DataFrame([row])], ignore_index=True)
    
    #Compute the following features
    for event in event_list:
        series = df_pivot[event].dropna() #dropna rows
        if len(series) >0: #if there is any event
            df_out[event+'_max'] = series.max()
            df_out[event+'_min'] = series.min()
            df_out[event+'_mean'] = series.mean(skipna=True)
            df_out[event+'_variance'] = series.var(skipna=True)
            df_out[event+'_meandiff'] = series.diff().mean() #average change
            df_out[event+'_meanabsdiff'] =series.diff().abs().mean()
            df_out[event+'_maxdiff'] = series.diff().abs().max()
            df_out[event+'_sumabsdiff'] =series.diff().abs().sum()
            df_out[event+'_diff'] = series.iloc[-1]-series.iloc[0]
            #Compute the n_peaks
            peaks,_ = find_peaks(series) #, threshold=series.median()
            df_out[event+'_npeaks'] = len(peaks)
            #Compute the trend (linear slope)
            if len(series)>1:
                df_out[event+'_trend']= np.polyfit(np.arange(len(series)), series, 1)[0] #fit deg-1 poly
            else:
                 df_out[event+'_trend'] = 0
    return df_out

## Notes
from transformers import AutoTokenizer, AutoModel
biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# FOR NOTES OBTAIN FIXED-SIZED BIOBERT EMBEDDINGS FOR ALL NOTE EVENTS OF A SINGLE TIMEBOUND PATIENT ICU STAY
def get_biobert_embedding_from_events_list(full_events_list, event_weights, verbose = 0):
    # Inputs:
    #   full_events_list -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   event_weights ->  Weights for aggregation of features in final embeddings
    #   verbose -> Level of printed output of function
    #
    # Outputs:
    #   aggregated_embeddings -> Biobert event features for all events
    #   full_embeddings -> Biobert event features across each event line
    #   event_weights -> Finally used weights for aggregation of features in final embeddings
  
    # %% EXAMPLE OF USE
    # aggregated_embeddings, full_embeddings, event_weights = get_biobert_embedding_from_events_list(full_events_list, event_weights, verbose=1)
  
    event_weights_exp = []
    for idx, event_string in enumerate(full_events_list):   
        weight = event_weights.values[idx]
        string_list, lengths = split_note_document(event_string)
        for idx_sub, event_string_sub in enumerate(string_list):
            #Extract biobert embedding
            embedding, hidden_embedding = get_biobert_embeddings(event_string_sub)
            #Concatenate
            if (idx==0) & (idx_sub==0):
                full_embedding = embedding
            else: 
                full_embedding = np.concatenate((full_embedding, embedding), axis=0)
            event_weights_exp.append(weight)
          
    # Return the weighted average of ebedding vector across temporal dimension
    try:
        #aggregated_embedding = np.dot(np.transpose(full_embedding), np.array(event_weights_exp))
        aggregated_embedding = np.average(full_embedding, axis=0, weights=np.array(event_weights_exp))
    except:
        aggregated_embedding = np.zeros(768)
      
    return aggregated_embedding, full_embedding, event_weights


# FOR NOTES SPLIT TEXT IF TOO LONG FOR NOTE EMBEDDING EXTRACTION
def split_note_document(text, min_length = 15):
    # Inputs:
    #   text -> String of text to be processed into an embedding. BioBERT can only process a string with â‰¤ 512 tokens. If the 
    #           input text exceeds this token count, we split it based on line breaks (driven from the discharge summary syntax). 
    #   min_length ->  When parsing the text into its subsections, remove text strings below a minimum length. These are generally 
    #                  very short and encode minimal information (e.g. 'Name: ___'). 
    #
    # Outputs:
    #   chunk_parse -> A list of "chunks", i.e. text strings, that breaks up the original text into strings with â‰¤ 512 tokens
    #   chunk_length -> A list of the token counts for each "chunk"
  
    # %% EXAMPLE OF USE
    # chunk_parse, chunk_length = split_note_document(ext, min_length = 15)
  
    tokens_list_0 = biobert_tokenizer.tokenize(text)
  
    if len(tokens_list_0) <= 510:
        return [text], [1]
    #print("Text exceeds 512 tokens - splitting into sections")
  
    chunk_parse = []
    chunk_length = []
    chunk = text
  
    ## Go through text and aggregate in groups up to 510 tokens (+ padding)
    tokens_list = biobert_tokenizer.tokenize(chunk)
    if len(tokens_list) >= 510:
        temp = chunk.split('\n')
        ind_start = 0
        len_sub = 0
        for i in range(len(temp)):
            temp_tk = biobert_tokenizer.tokenize(temp[i])
            if len_sub + len(temp_tk) >  510:
                chunk_parse.append(' '.join(temp[ind_start:i]))
                chunk_length.append(len_sub)
                # reset for next chunk
                ind_start = i
                len_sub = len(temp_tk)
            else: 
                len_sub += len(temp_tk)
    elif len(tokens_list) >= min_length:
        chunk_parse.append(chunk)
        chunk_length.append(len(tokens_list))
    #print("Parsed lengths: ", chunk_length)
      
    return chunk_parse, chunk_length

def get_biobert_embeddings(text):
    # Inputs:
    #   text -> Input text (str)
    #
    # Outputs:
    #   embeddings -> Final Biobert embeddings with vector dimensionality = (1,768)
    #   hidden_embeddings -> Last hidden layer in Biobert model with vector dimensionality = (token_size,768)
  
    # %% EXAMPLE OF USE
    # embeddings, hidden_embeddings = get_biobert_embeddings(text)
  
    tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    outputs = biobert_model(**tokens_pt)
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_embeddings = last_hidden_state.detach().numpy()
    embeddings = pooler_output.detach().numpy()

    return embeddings, hidden_embeddings

def get_notes_biobert_embeddings(dt_patient, note_type):
    # Inputs:
    #   dt_patient -> Timebound ICU patient stay structure filtered by max_time_stamp or min_time_stamp if any
    #   note_type -> Type of note to get
    #
    # Outputs:
    #   aggregated_embeddings -> Biobert event features for selected note
  
    # %% EXAMPLE OF USE
    # aggregated_embeddings = get_notes_biobert_embeddings(dt_patient, note_type = 'ecgnotes')
  
    admittime = dt_patient.core['admittime'].values[0]
    note_table = getattr(dt_patient, note_type).copy()
    note_table['deltacharttime'] = note_table['charttime'].apply(lambda x: (x.replace(tzinfo=None) - admittime).total_seconds()/3600)
    try:
        aggregated_embeddings, __, __ = get_biobert_embedding_from_events_list(note_table['text'], note_table['deltacharttime'])
    except:
        aggregated_embeddings, __, __ = get_biobert_embedding_from_events_list(pd.Series([""]), pd.Series([1]))
  
    return aggregated_embeddings


# CXR-CLIP
import os
from typing import Dict, Union

import albumentations
import albumentations.pytorch.transforms

import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

def load_tokenizer(source, pretrained_model_name_or_path, cache_dir, **kwargs):
    if source == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=os.path.exists(os.path.join(cache_dir, f'models--{pretrained_model_name_or_path.replace("/", "--")}')),
            **kwargs,
        )
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = tokenizer.cls_token_id
    else:
        raise KeyError(f"Not supported tokenizer source: {source}")

    return tokenizer


def load_transform(split: str = "train", transform_config: Dict = None):
    assert split in {"train", "valid", "test", "aug"}

    config = []
    if transform_config:
        if split in transform_config:
            config = transform_config[split]
    image_transforms = []

    for name in config:
        if hasattr(transforms, name):
            tr_ = getattr(transforms, name)
        else:
            tr_ = getattr(albumentations, name)
        tr = tr_(**config[name])
        image_transforms.append(tr)

    return image_transforms


def transform_image(image_transforms, image: Union[Image.Image, np.ndarray], normalize="huggingface"):
    for tr in image_transforms:
        if isinstance(tr, albumentations.BasicTransform):
            image = np.array(image) if not isinstance(image, np.ndarray) else image
            image = tr(image=image)["image"]
        else:
            image = transforms.ToPILImage()(image) if not isinstance(image, Image.Image) else image
            image = tr(image)

    if normalize == "huggingface":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)(image)

    elif normalize == "imagenet":
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

    else:
        raise KeyError(f"Not supported Normalize: {normalize}")

    return image