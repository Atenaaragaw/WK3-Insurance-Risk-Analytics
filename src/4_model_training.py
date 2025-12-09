import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = 'data/MachineLearningRating_v3.txt'
RANDOM_STATE = 42

def load_and_clean_data(path):
    df = pd.read_csv(path, sep='|', low_memory=False)
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0)
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0)
    df['RegistrationYear'] = pd.to_numeric(df['RegistrationYear'], errors='coerce')
    df = df[df['TotalPremium'] > 0].reset_index(drop=True)
    df['Claim_Frequency'] = np.where(df['TotalClaims'] > 0, 1, 0)
    return df

def perform_feature_engineering(df):
    features_to_use = ['TotalPremium', 'SumInsured', 'RegistrationYear', 'Province', 'Gender', 'make', 'Claim_Frequency', 'TotalClaims']
    df_model = df[features_to_use].copy()
    CURRENT_YEAR = 2015
    df_model['VehicleAge'] = CURRENT_YEAR - df_model['RegistrationYear']
    df_model['VehicleAge'] = df_model['VehicleAge'].clip(lower=0)
    categorical_cols = ['Province', 'Gender']
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, prefix=categorical_cols)
    return df_encoded

def prepare_modeling_data(df_encoded):
    Y = df_encoded['Claim_Frequency']
    # Drop original targets and complex/high-cardinality columns not encoded yet (like 'make')
    X = df_encoded.drop(columns=['Claim_Frequency', 'TotalClaims', 'make', 'RegistrationYear', 'Province', 'Gender'], errors='ignore')
    
    # Scale numerical features (Crucial for Logistic Regression!)
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=Y
    )
    return X_train, X_test, Y_train, Y_test, scaler, X.columns.tolist() # Return columns for later use

# --- Model Training and Evaluation ---
def train_and_evaluate_model(X_train, X_test, Y_train, Y_test, feature_names):
    """Trains Logistic Regression and prints performance metrics."""
    
    # 1. Train the Model
    # Since the claim rate is very low (imbalanced data), we use class_weight='balanced'
    model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', solver='liblinear', max_iter=100)
    print("Training Logistic Regression Model...")
    model.fit(X_train, Y_train)
    
    # 2. Predict and Evaluate
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Model Performance Summary (Logistic Regression) ---")
    print(classification_report(Y_test, Y_pred, target_names=['No Claim', 'Claim']))
    print(f"F1-Score (Claim Class): {f1_score(Y_test, Y_pred, pos_label=1):.4f}")
    
    # 3. Model Interpretation (Coefficients)
    print("\n--- Model Interpretation (Coefficients) ---")
    # Coefficients reflect the impact of each scaled feature on the log-odds of a claim
    coefficients = pd.Series(model.coef_[0], index=feature_names)
    print("Top 10 Positive Coefficients (Increases Risk):")
    print(coefficients.sort_values(ascending=False).head(10))
    
    # 4. Save Model Artifact (MLOps Requirement)
    joblib.dump(model, 'models/baseline_lr_model.pkl')
    print("\nModel saved to models/baseline_lr_model.pkl")

# --- Execution ---
if __name__ == "__main__":
    print("--- Starting Model Training (Task 5) ---")
    
    # Setup MLOps Folders
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Full Pipeline Execution
    df_raw = load_and_clean_data(DATA_PATH)
    df_transformed = perform_feature_engineering(df_raw)
    X_train, X_test, Y_train, Y_test, scaler, feature_names = prepare_modeling_data(df_transformed)
    
    train_and_evaluate_model(X_train, X_test, Y_train, Y_test, feature_names)