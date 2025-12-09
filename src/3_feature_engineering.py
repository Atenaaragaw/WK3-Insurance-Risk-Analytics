import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# --- Configuration and Data Loading ---
DATA_PATH = 'data/MachineLearningRating_v3.txt'
RANDOM_STATE = 42

def load_and_clean_data(path):
    """Loads the data and performs essential cleaning and type conversion."""
    df = pd.read_csv(path, sep='|', low_memory=False)
    
    # 1. Type Conversion
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0)
    df['RegistrationYear'] = pd.to_numeric(df['RegistrationYear'], errors='coerce')
    
    # Drop rows where TotalPremium is zero to avoid errors
    df = df[df['TotalPremium'] > 0].reset_index(drop=True)
    
    # 2. Derive Target Variables (Frequency and Severity)
    # Frequency (Claim Flag): Binary variable (1 if a claim occurred, 0 otherwise)
    df['Claim_Frequency'] = np.where(df['TotalClaims'] > 0, 1, 0)
    
    # Severity: Average cost per policy (TotalClaims / TotalPremium)
    # Note: For modeling, we often model TotalClaims and use TotalPremium as exposure.
    # We will define Severity simply as TotalClaims for now.
    
    return df

def perform_feature_engineering(df):
    """Encodes categorical variables and creates new numerical features."""
    
    # --- 1. Identify Key Features ---
    # Use only the features confirmed as high-impact from EDA and the new target
    features_to_use = [
        'TotalPremium', 
        'SumInsured', 
        'RegistrationYear', 
        'Province', 
        'Gender', 
        'make',
        'Claim_Frequency', # Target
        'TotalClaims'      # Target/Severity proxy
    ]
    df_model = df[features_to_use]
    
    # --- 2. Feature Creation (Tenure/Vehicle Age) ---
    CURRENT_YEAR = 2015 # Based on the latest data entry in EDA
    df_model['VehicleAge'] = CURRENT_YEAR - df_model['RegistrationYear']
    df_model['VehicleAge'] = df_model['VehicleAge'].clip(lower=0) # Handle potential negative/large values
    
    # --- 3. Encoding Categorical Variables (One-Hot Encoding) ---
    
    # Selected categorical columns based on EDA importance:
    categorical_cols = ['Province', 'Gender']
    
    # Apply One-Hot Encoding to convert categories into numerical features
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    
    print(f"\nDataFrame size after encoding: {df_encoded.shape}")
    print(f"New Feature Columns (Provinces/Gender): {df_encoded.filter(like='Province_').columns.tolist()}")

    return df_encoded

def prepare_modeling_data(df_encoded):
    """Separates features (X) from the target (Y) and splits the data."""
    
    # 1. Define Target and Features
    # We will use Claim_Frequency (binary) as the primary target for initial modeling
    Y = df_encoded['Claim_Frequency']
    X = df_encoded.drop(columns=['Claim_Frequency', 'TotalClaims']) 
    
    # Remove original (now encoded) columns and other non-feature columns
    
    # 2. Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=Y # Ensure balanced classes in train/test sets
    )
    
    print("\n--- Data Split Summary ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Y_train Claim Frequency: {Y_train.mean():.4f}")
    
    return X_train, X_test, Y_train, Y_test

# --- Execution ---
if __name__ == "__main__":
    print("--- Starting Feature Engineering (Task 4) ---")
    
    # Load and clean
    df_raw = load_and_clean_data(DATA_PATH)
    
    # Feature engineering
    df_transformed = perform_feature_engineering(df_raw)
    
    # Prepare and split data
    X_train, X_test, Y_train, Y_test = prepare_modeling_data(df_transformed)
    
    print("\nFeature Engineering Complete. Ready for Model Training.")