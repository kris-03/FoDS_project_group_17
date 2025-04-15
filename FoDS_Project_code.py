#Hi
import pandas as pd
import numpy as np
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv(
    filepath_or_buffer="oral_cancer_prediction_dataset.csv",
    dtype={
        "Country": "category",
        "Gender": "category",
        "Tobacco Use": "category",
        "Alcohol Consumption": "category",
        "HPV Infection": "category",
        "Betel Quid Use": "category",
        "Chronic Sun Exposure": "category",
        "Poor Oral Hygiene": "category",
        "Diet (Fruits & Vegetables Intake)": "category",
        "Family History of Cancer": "category",
        "Compromised Immune System": "category",
        "Oral Lesions": "category",
        "Unexplained Bleeding": "category",
        "Difficulty Swallowing": "category",
        "White or Red Patches in Mouth": "category",
        "Cancer Stage": "category", 
        "Treatment Type": "category",
        "Early Diagnosis": "category", 
        "Oral Cancer (Diagnosis)": "category"
    }
)
