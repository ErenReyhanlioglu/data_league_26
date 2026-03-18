import pandas as pd
import numpy as np
import itertools

def create_temporal_features(df):
    """
    Temporal Features: Cyclic encoding for hour and day.
    """
    df['hour'] = df['appointment_datetime'].dt.hour
    df['day_of_week'] = df['appointment_datetime'].dt.dayofweek
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def apply_hierarchical_clinic_encoding(train_df, target_df, target_col='label_noshow'):
    """
    Hierarchical Target Encoding with fallback logic.
    """
    global_mean = train_df[target_col].mean()
    
    area_map = train_df.groupby('area_id')[target_col].mean()
    spec_map = train_df.groupby('specialty')[target_col].mean()
    clinic_map = train_df.groupby('clinic_id')[target_col].mean()
    
    target_df['hier_score'] = target_df['clinic_id'].map(clinic_map)
    target_df['hier_score'] = target_df['hier_score'].fillna(target_df['specialty'].map(spec_map))
    target_df['hier_score'] = target_df['hier_score'].fillna(target_df['area_id'].map(area_map))
    target_df['hier_score'] = target_df['hier_score'].fillna(global_mean)
    
    return target_df

def create_patient_history_aggregations(df):
    """
    Patient-level behavioral history with leakage prevention.
    """
    df = df.sort_values(['patient_id', 'appointment_datetime'])
    group = df.groupby('patient_id')
    
    df['patient_appt_count'] = group.cumcount()
    
    # Using expanding window and shift(1) to avoid target leakage
    df['patient_avg_lead_time'] = group['lead_time_hours'].transform(lambda x: x.shift(1).expanding().mean())
    df['patient_max_lead_time'] = group['lead_time_hours'].transform(lambda x: x.shift(1).expanding().max())
    df['patient_total_sms_received'] = group['sms_sent'].transform(lambda x: x.shift(1).expanding().sum())
    
    cols_to_fill = ['patient_avg_lead_time', 'patient_max_lead_time', 'patient_total_sms_received']
    df[cols_to_fill] = df[cols_to_fill].fillna(0)
    
    return df

def create_clinic_load_aggregations(df):
    """
    Clinic Dynamics using 7-day rolling statistics.
    """
    df['appt_date'] = df['appointment_datetime'].dt.date
    daily_vol = df.groupby(['clinic_id', 'appt_date']).size().reset_index(name='daily_vol')
    
    clinic_group = daily_vol.groupby('clinic_id')['daily_vol']
    daily_vol['clinic_load_avg_7d'] = clinic_group.transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    daily_vol['clinic_load_max_7d'] = clinic_group.transform(lambda x: x.rolling(window=7, min_periods=1).max())
    daily_vol['clinic_load_min_7d'] = clinic_group.transform(lambda x: x.rolling(window=7, min_periods=1).min())
    
    df = df.merge(daily_vol[['clinic_id', 'appt_date', 'clinic_load_avg_7d', 'clinic_load_max_7d', 'clinic_load_min_7d']], 
                  on=['clinic_id', 'appt_date'], how='left')
    
    return df.drop(columns=['appt_date'])

def apply_patient_demographic_proxy(train_df, target_df):
    """
    Demographic Proxy Imputation. 
    FIX: Bins updated to [-1, ...] to include age 0.
    """
    bins = [-1, 18, 30, 45, 60, 200]
    labels = [1, 2, 3, 4, 5]
    
    # Process both dataframes safely
    for d in [train_df, target_df]:
        # include_lowest=True is another way, but -1 bin is safer for outliers
        d['age_bucket'] = pd.cut(d['age'], bins=bins, labels=labels)
        # Fill any potential NaNs (though reports show none, defensive coding is rational)
        d['age_bucket'] = d['age_bucket'].fillna(1).astype(int)
    
    proxy_map = train_df.groupby(['age_bucket', 'area_id', 'ses_score'])['label_noshow'].mean().reset_index()
    proxy_map.rename(columns={'label_noshow': 'demo_proxy_rate'}, inplace=True)
    
    target_df = target_df.merge(proxy_map, on=['age_bucket', 'area_id', 'ses_score'], how='left')
    target_df['demo_proxy_rate'] = target_df['demo_proxy_rate'].fillna(train_df['label_noshow'].mean())
    
    return target_df

def create_advanced_interactions(df):
    """
    Numerical interaction features.
    """
    # Lead time interactions
    df['lead_time_per_age'] = df['lead_time_hours'] / (df['age'] + 1)
    df['lead_time_per_distance'] = df['lead_time_hours'] / (df['distance_km'] + 0.1)
    
    # Distance and Socio-economics
    df['dist_ses_interaction'] = df['distance_km'] * df['ses_score']
    df['distance_log'] = np.log1p(df['distance_km'])
    df['lead_time_log'] = np.log1p(df['lead_time_hours'])
    
    return df

def apply_multi_level_target_encoding(train_df, target_df, target_col='label_noshow'):
    """
    Combined categorical target encoding.
    Creates new features by combining pairs of categories.
    """
    combinations = [
        ['specialty', 'day_of_week'],
        ['area_id', 'day_of_week'],
        ['specialty', 'hour'],
        ['booking_channel', 'appointment_type'],
        ['sex', 'specialty']
    ]
    
    global_mean = train_df[target_col].mean()
    
    for combo in combinations:
        col_name = f"te_{'_'.join(combo)}"
        # Group by the combination and calculate mean on train
        agg = train_df.groupby(combo)[target_col].mean().reset_index()
        agg.rename(columns={target_col: col_name}, inplace=True)
        
        # Merge back to target_df
        target_df = target_df.merge(agg, on=combo, how='left')
        target_df[col_name] = target_df[col_name].fillna(global_mean)
        
    return target_df

def create_deep_aggregations(df):
    """
    High-level statistical aggregations (STD, MAX, SKEW).
    Functions: AVG, SUM, MAX, MIN, STD
    """
    # Clinic Level Dynamics
    clinic_aggs = df.groupby('clinic_id').agg({
        'lead_time_hours': ['mean', 'std', 'max'],
        'age': ['mean', 'std'],
        'distance_km': ['mean']
    })
    
    # Flatten columns: 'clinic_lead_time_hours_std' etc.
    clinic_aggs.columns = [f"clinic_{'_'.join(col).strip()}" for col in clinic_aggs.columns.values]
    df = df.merge(clinic_aggs, on='clinic_id', how='left')
    
    # Patient Level Diversity
    # How many different specialties or clinics has the patient visited?
    patient_diversity = df.groupby('patient_id').agg({
        'specialty': 'nunique',
        'clinic_id': 'nunique',
        'area_id': 'nunique'
    }).reset_index()
    patient_diversity.columns = ['patient_id', 'pt_spec_nunique', 'pt_clinic_nunique', 'pt_area_nunique']
    
    df = df.merge(patient_diversity, on='patient_id', how='left')
    
    return df

def generate_bulk_features(df):
    """
    Automated expansion via loops (Count Encoding).
    """
    cols_to_count = ['specialty', 'clinic_id', 'area_id', 'patient_id', 'hour']
    for col in cols_to_count:
        df[f'{col}_count_freq'] = df[col].map(df[col].value_counts())
        
    return df

import pandas as pd
import numpy as np
import itertools

def expand_to_150_plus(train_df, target_df, target_col='label_noshow'):
    """
    Idempotent brute-force expansion to 150+ features.
    Prevents duplicates and handles test set constraints.
    """
    df_result = target_df.copy()
    global_mean = train_df[target_col].mean()
    
    cat_cols = ['specialty', 'area_id', 'booking_channel', 'appointment_type', 'sex', 'day_of_week', 'hour']
    
    for combo in itertools.combinations(cat_cols, 2):
        col_name = f"combo_te_{'_'.join(combo)}"
        if col_name in df_result.columns:
            continue
            
        agg = train_df.groupby(list(combo))[target_col].mean().reset_index()
        agg.rename(columns={target_col: col_name}, inplace=True)
        df_result = df_result.merge(agg, on=list(combo), how='left')
        df_result[col_name] = df_result[col_name].fillna(global_mean)

    nums = ['lead_time_hours', 'distance_km', 'age', 'wait_mins_est']
    cats = ['specialty', 'area_id', 'booking_channel', 'clinic_id']
    
    for c in cats:
        for n in nums:
            new_cols = [f"stat_{c}_{n}_{s}" for s in ['mean', 'std', 'median', 'var']]
            if any(col in df_result.columns for col in new_cols):
                continue
                
            stats = train_df.groupby(c)[n].agg(['mean', 'std', 'median', 'var']).reset_index()
            stats.columns = [c] + new_cols
            df_result = df_result.merge(stats, on=c, how='left')
            
    if 'prev_noshow_1' not in df_result.columns:
        if target_col in df_result.columns:
            df_result = df_result.sort_values(['patient_id', 'appointment_datetime'])
            df_result['prev_noshow_1'] = df_result.groupby('patient_id')[target_col].shift(1).fillna(-1)
            df_result['prev_noshow_2'] = df_result.groupby('patient_id')[target_col].shift(2).fillna(-1)
        else:
            df_result['prev_noshow_1'] = -1
            df_result['prev_noshow_2'] = -1

    return df_result.fillna(0)