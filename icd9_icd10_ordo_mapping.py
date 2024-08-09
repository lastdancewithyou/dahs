import pandas as pd
import rdflib
from rdflib.namespace import RDFS
import warnings
import re
from tqdm import tqdm

class ICDProcessor:
    def __init__(self, diagnoses_icd, map_titles, icd9_to_icd10, ordo_owl_path):
        self.diagnoses_icd = diagnoses_icd
        self.map_titles = map_titles
        self.icd9_to_icd10 = icd9_to_icd10
        self.ordo_owl_path = ordo_owl_path
        self.graph = rdflib.Graph()
        self.ordo_df = None # orphanet에서 추출될 데이터를 담을 df

    def icd9_to_icd10_mapping(self):
        df_icd9 = self.diagnoses_icd[self.diagnoses_icd['icd_version'] == 9]
        icd9_unique = df_icd9['icd_code'].unique()
        icd9_unique_df = pd.DataFrame(icd9_unique, columns=['icd9'])
        icd9_unique_df = icd9_unique_df.merge(self.map_titles, left_on='icd9', right_on='icd_code')
        icd9_unique_df.rename(columns={'long_title': 'icd9_long_titles'}, inplace=True)
        icd9_unique_df['icd10-NZ'] = ""
        icd9_unique_df['icd10-NZ'] = icd9_unique_df['icd10-NZ'].apply(list)
        icd9_unique_df['icd10-long-titles'] = ""
        icd9_unique_df['icd10-long-titles'] = icd9_unique_df['icd10-long-titles'].apply(list)

        total_tasks_icd9 = len(icd9_unique_df)
        print("Extracting ICD-9 and mapping it to ICD-10")
        for i, row in tqdm(icd9_unique_df.iterrows(), total=total_tasks_icd9):
            icd9_code = row['icd9']
            icd10_matches = self.icd9_to_icd10[self.icd9_to_icd10['clinical_code_from'] == icd9_code]['clinical_code_to'].tolist()
            if icd10_matches:
                icd9_unique_df.at[i, 'icd10-NZ'] = icd10_matches
                long_titles_icd9 = []
                for icd10_code in icd10_matches:
                    long_title = self.map_titles[self.map_titles['icd_code'] == icd10_code]['long_title'].to_string(index=False).strip()
                    if long_title != 'Series([], )':
                        long_titles_icd9.append(long_title)
                icd9_unique_df.at[i, 'icd10_long_titles'] = long_titles_icd9

        icd9_unique_df = icd9_unique_df[['icd9', 'icd9_long_titles', 'icd10-NZ', 'icd10_long_titles']]
        df_icd9 = df_icd9.merge(icd9_unique_df, left_on='icd_code', right_on='icd9', how='left')
        df_icd9.drop(columns=['icd_version', 'icd_code'], inplace=True)
        return df_icd9

    def load_ordo_graph(self):
        try:
            self.graph.parse(self.ordo_owl_path, format='xml')
        except:
            print("xml file format is not correct.")
            try:
                self.graph.parse(self.ordo_owl_path, format="turtle")
            except:
                print("turtle file format is not correct.")
                try: 
                    self.graph.parse(self.ordo_owl_path, format="json-lod")
                except:
                    print("json-lod file format is not correct")
                    try:
                        self.graph.parse(self.ordo_owl_path, format="rdfa")
                    except: 
                        print('RDFa file format is not correct')

    def extract_icd10cm_triples(self):
        icd10cm_triples = []
        for subj, pred, obj in self.graph:
            if "ICD" in str(subj) or "ICD" in str(pred) or "ICD" in str(obj):
                icd10cm_triples.append((subj, pred, obj))
        return icd10cm_triples

    def process_ordo_triples(self, icd10cm_triples):
        data = []
        print("Extracting ICD-10 code from ontology")
        for subj, pred, obj in tqdm(icd10cm_triples):
            try:
                icd10_code = str(obj).split("ICD-10:")[1].strip()
                data.append([str(subj), icd10_code])
            except:
                pass

        df = pd.DataFrame(data, columns=['ORDO URL', 'ICD-10 Code'])
        df['ICD-10 Code'] = df['ICD-10 Code'].apply(self.remove_dots)
        print("rows before filtering: ", len(df))

        self.ordo_df = df[df['ICD-10 Code'].str.len() < 6]
        print('rows after filtering: ', len(self.ordo_df))

        self.ordo_df = self.ordo_df.copy()
        self.ordo_df.loc[:, 'disease name'] = self.ordo_df['ORDO URL'].apply(self.get_diseases)
        self.ordo_df['Orphanet_ID'] = self.ordo_df['ORDO URL'].apply(self.extract_orphanet_id)

    def merge_icd9_ordo(self, df_icd9):
        matched_rows = []
        tasks = len(df_icd9)

        print("matching ICD-9 code with ORDO using ICD-10")
        for index, row in tqdm(df_icd9.iterrows(), total=tasks):
            icd10_list = row['icd10-NZ']
            for icd10 in icd10_list:
                if icd10 in self.ordo_df['ICD-10 Code'].values:
                    matched_rows.append(row)
                    break

        icd_ordo = pd.DataFrame(matched_rows)
        icd_ordo_df = icd_ordo.explode('icd10-NZ')
        merged_icd9_ordo = icd_ordo_df.merge(self.ordo_df, left_on='icd10-NZ', right_on='ICD-10 Code', how='left')
        merged_icd9_ordo = merged_icd9_ordo[['subject_id', 'hadm_id', 'seq_num', 'ICD-10 Code', 'disease name', 'Orphanet_ID', 'ORDO URL']]
        return merged_icd9_ordo

    def merge_icd10_ordo(self, df_icd10):
        merged_icd10_ordo = df_icd10.merge(self.ordo_df, left_on='icd_code', right_on='ICD-10 Code', how='left')
        merged_icd10_ordo = merged_icd10_ordo[['subject_id', 'hadm_id', 'seq_num', 'ICD-10 Code', 'disease name', 'Orphanet_ID', 'ORDO URL']]
        return merged_icd10_ordo

    def process(self):
        self.load_ordo_graph()
        icd10cm_triples = self.extract_icd10cm_triples()
        self.process_ordo_triples(icd10cm_triples)

        df_icd9 = self.icd9_to_icd10_mapping()
        merged_icd9_ordo = self.merge_icd9_ordo(df_icd9)

        df_icd10 = self.diagnoses_icd[self.diagnoses_icd['icd_version'] == 10]
        merged_icd10_ordo = self.merge_icd10_ordo(df_icd10)

        merged_icd_ordo = pd.concat([merged_icd9_ordo, merged_icd10_ordo])
        filtered_icd_ordo = merged_icd_ordo[merged_icd_ordo['Orphanet_ID'].notnull()]
        return filtered_icd_ordo

    @staticmethod
    def remove_dots(code):
        return code.replace('.', '')

    def get_diseases(self, uri):
        for subj, pred, obj in self.graph.triples((rdflib.URIRef(uri), RDFS.label, None)):
            return str(obj)
        return None

    @staticmethod
    def extract_orphanet_id(url):
        if isinstance(url, str):
            match = re.search(r'Orphanet_(\d+)', url)
            if match:
                return match.group(1)
        return None

# diagnoses_icd = pd.read_parquet("")
# map_titles = pd.read_parquet("")
# icd9_to_icd10 = pd.read_excel("")
# ordo_owl_path = ""

processor = ICDProcessor(diagnoses_icd, map_titles, icd9_to_icd10, ordo_owl_path)
final_df = processor.process()
final_df.to_csv('icd9_icd10_ordo.csv', index=False)
print(final_df.head())