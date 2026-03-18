import pandas as pd
import numpy as np

def load_raw_data(file_path):
    """
    Loads a single CSV file from the specified path.
    """
    return pd.read_csv(file_path)

def merge_with_lookups(df, patients_df, clinics_df):
    """
    Joins appointment data with patient and clinic lookup tables.
    Uses left join to preserve all appointment records.
    """
    df = df.merge(patients_df, on='patient_id', how='left')
    df = df.merge(clinics_df, on='clinic_id', how='left', suffixes=('', '_clinic'))
    return df

def convert_to_datetime(df, columns):
    """
    Converts a list of columns to pandas datetime objects.
    """
    for col in columns:
        df[col] = pd.to_datetime(df[col])
    return df

def handle_lead_time_leakage(df):
    """
    Addresses time leakage: booking_datetime cannot be after appointment_datetime.
    Caps lead_time_hours at 0 if leakage is detected.
    """
    mask_leakage = df['booking_datetime'] > df['appointment_datetime']
    if mask_leakage.any():
        df.loc[mask_leakage, 'lead_time_hours'] = 0
    return df

def process_sms_logic(df):
    """
    Handles SMS logic: if sms_sent is 0, sets lead_hours to -1 and flags as missing.
    """
    df['sms_is_missing'] = 0
    df.loc[df['sms_sent'] == 0, 'sms_is_missing'] = 1
    df.loc[df['sms_sent'] == 0, 'sms_lead_hours'] = -1
    return df

def downcast_memory(df):
    """
    Optimizes memory usage by downcasting numeric types to float32 and int32.
    Essential for large-scale feature engineering and GPU training.
    """
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

def check_missing_values(df, name="Dataset"):
    """
    Analyzes missing values in the dataframe and returns a summary.
    Rational approach to identify data quality issues before FE.
    """
    missing_count = df.isnull().sum()
    missing_percent = 100 * df.isnull().sum() / len(df)
    
    missing_table = pd.concat([missing_count, missing_percent], axis=1)
    missing_table.columns = ['Missing Values', '% of Total Values']
    
    # Filter only columns with missing values
    missing_table = missing_table[missing_table['Missing Values'] > 0].sort_values('% of Total Values', ascending=False)
    
    if missing_table.empty:
        print(f"Excellent: No missing values found in {name}.")
    else:
        print(f"Missing values found in {name}:")
        print(missing_table)
    
    return missing_table