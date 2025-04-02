import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_mimic_tables(mimic_path):
    """
    Load relevant MIMIC-IV tables for mortality prediction
    """
    # Core tables
    patients = pd.read_csv(f"{mimic_path}/core/patients.csv")
    admissions = pd.read_csv(f"{mimic_path}/core/admissions.csv")
    icustays = pd.read_csv(f"{mimic_path}/icu/icustays.csv")
    
    # Clinical data
    labevents = pd.read_csv(f"{mimic_path}/hosp/labevents.csv")
    chartevents = pd.read_csv(f"{mimic_path}/icu/chartevents.csv")
    outputevents = pd.read_csv(f"{mimic_path}/icu/outputevents.csv")
    
    return patients, admissions, icustays, labevents, chartevents, outputevents

def process_demographic_features(patients, admissions):
    """
    Process demographic features including sensitive attributes
    """
    demographics = patients.merge(admissions, on='subject_id', how='inner')
    
    # Extract sensitive attributes
    demographics['age'] = demographics['anchor_age']
    demographics['gender'] = demographics['gender'].map({'F': 0, 'M': 1})
    demographics['insurance'] = pd.get_dummies(demographics['insurance'], prefix='insurance')
    demographics['language'] = pd.get_dummies(demographics['language'], prefix='language')
    demographics['ethnicity'] = pd.get_dummies(demographics['ethnicity'], prefix='ethnicity')
    demographics['marital_status'] = pd.get_dummies(demographics['marital_status'], prefix='marital')
    
    return demographics

def process_clinical_features(icustays, labevents, chartevents, outputevents):
    """
    Process clinical features from ICU stays and events
    """
    # Aggregate lab values
    lab_features = labevents.groupby(['subject_id', 'hadm_id']).agg({
        'valuenum': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Aggregate vital signs
    vital_features = chartevents.groupby(['subject_id', 'hadm_id']).agg({
        'valuenum': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Aggregate output events
    output_features = outputevents.groupby(['subject_id', 'hadm_id']).agg({
        'value': ['sum', 'mean']
    }).reset_index()
    
    # Merge clinical features
    clinical = icustays.merge(lab_features, on=['subject_id', 'hadm_id'], how='left')
    clinical = clinical.merge(vital_features, on=['subject_id', 'hadm_id'], how='left')
    clinical = clinical.merge(output_features, on=['subject_id', 'hadm_id'], how='left')
    
    return clinical

def create_mortality_label(admissions, icustays):
    """
    Create mortality label (death during ICU stay)
    """
    mortality = admissions[['subject_id', 'hadm_id', 'deathtime']].merge(
        icustays[['subject_id', 'hadm_id', 'outtime']], 
        on=['subject_id', 'hadm_id']
    )
    mortality['mortality'] = (~pd.isna(mortality['deathtime'])) & (mortality['deathtime'] <= mortality['outtime'])
    return mortality[['subject_id', 'hadm_id', 'mortality']]

def prepare_final_dataset(mimic_path, random_state=42):
    """
    Prepare final dataset for mortality prediction models
    """
    # Load tables
    patients, admissions, icustays, labevents, chartevents, outputevents = load_mimic_tables(mimic_path)
    
    # Process features
    demographics = process_demographic_features(patients, admissions)
    clinical = process_clinical_features(icustays, labevents, chartevents, outputevents)
    mortality = create_mortality_label(admissions, icustays)
    
    # Merge all features
    final_df = demographics.merge(clinical, on=['subject_id', 'hadm_id'], how='inner')
    final_df = final_df.merge(mortality, on=['subject_id', 'hadm_id'], how='inner')
    
    # Handle missing values
    final_df = final_df.fillna(final_df.mean())
    
    # Split features and target
    X = final_df.drop(['mortality', 'subject_id', 'hadm_id'], axis=1)
    y = final_df['mortality']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

# Example usage:
# mimic_path = "/path/to/mimic/data"
# X_train, X_test, y_train, y_test, feature_names = prepare_final_dataset(mimic_path)
