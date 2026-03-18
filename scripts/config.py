import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Input Data Directory (Read-only) 
RAW_DATA_DIR = ROOT_DIR / "raw_data"

# Processed Data Directory 
PROCESSED_DATA_DIR = ROOT_DIR / "processed_data"

# Model Artifacts and Logs 
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Input Files 
TRAIN_CSV = RAW_DATA_DIR / "appointments_train.csv"
TEST_CSV = RAW_DATA_DIR / "appointments_test.csv"
PATIENTS_CSV = RAW_DATA_DIR / "patients.csv"
CLINICS_CSV = RAW_DATA_DIR / "clinics.csv"
SAMPLE_SUBMISSION = RAW_DATA_DIR / "sample_submission.csv"

# Stage-specific Output Directories 
STAGE1_OUT = PROCESSED_DATA_DIR / "stage1"
STAGE2_OUT = PROCESSED_DATA_DIR / "stage2"
STAGE3_OUT = PROCESSED_DATA_DIR / "stage3"
STAGE4_OUT = PROCESSED_DATA_DIR / "stage4"
STAGE5_OUT = PROCESSED_DATA_DIR / "stage5"
STAGE6_OUT = PROCESSED_DATA_DIR / "stage6"

# Date Constants (Time-Split) 
# Sub-Train: Jan 2025 - Sep 2025
# Sub-Val: Oct 2025
VAL_SPLIT_DATE = "2025-10-01" 

# Global Constants
RANDOM_STATE = 42
TARGET_COL = "label_noshow" 

# Ensure necessary directories exist
DIRECTORIES = [
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR,
    STAGE1_OUT, STAGE2_OUT, STAGE3_OUT, 
    STAGE4_OUT, STAGE5_OUT, STAGE6_OUT
]

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)