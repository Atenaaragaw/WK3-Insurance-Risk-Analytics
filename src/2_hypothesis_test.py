import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# --- Data Loading and Preprocessing ---
DATA_PATH = 'data/MachineLearningRating_v3.txt'

try:
    # Use DVC to ensure the large file is available
    df = pd.read_csv(DATA_PATH, sep='|', low_memory=False)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please run 'dvc pull'.")
    exit()

# Data Cleaning (Ensure TotalClaims is numeric)
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')

# --- Hypothesis Test Setup ---

# Define the two provinces
province_Gauteng = df[df['Province'] == 'Gauteng'].copy()
province_NC = df[df['Province'] == 'Northern Cape'].copy()

# Add a binary 'Claimed' column: 1 if TotalClaims > 0, 0 otherwise.
# This transforms the data into a binomial distribution required for the Z-test for proportions.
province_Gauteng['Claimed'] = np.where(province_Gauteng['TotalClaims'] > 0, 1, 0)
province_NC['Claimed'] = np.where(province_NC['TotalClaims'] > 0, 1, 0)

# --- Z-Test for Two Population Proportions (Claim Frequency) ---

# Hypotheses:
# H0 (Null): Claim Proportion (Gauteng) <= Claim Proportion (NC)  (p1 <= p2)
# Ha (Alternative): Claim Proportion (Gauteng) > Claim Proportion (NC) (p1 > p2)

# 1. Calculate successes (count) and total observations (nobs)
n_Gauteng = len(province_Gauteng)
successes_Gauteng = province_Gauteng['Claimed'].sum()

n_NC = len(province_NC)
successes_NC = province_NC['Claimed'].sum()

# 2. Prepare inputs for proportions_ztest
count = np.array([successes_Gauteng, successes_NC])
nobs = np.array([n_Gauteng, n_NC])

# 'larger' performs the one-sided test for Ha: p1 > p2
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger') 

# --- Print Results and Conclusion ---
p_Gauteng = successes_Gauteng / n_Gauteng
p_NC = successes_NC / n_NC

print("--- Statistical Hypothesis Test: Gauteng vs Northern Cape Claim Frequency ---")
print(f"H0: Claim Frequency (Gauteng) <= Claim Frequency (NC)")
print(f"Ha: Claim Frequency (Gauteng) > Claim Frequency (NC) (One-Tailed Test)")
print("-" * 75)
print(f"Gauteng Claim Proportion (p1): {p_Gauteng:.6f} (n={n_Gauteng})")
print(f"Northern Cape Claim Proportion (p2): {p_NC:.6f} (n={n_NC})")
print("-" * 75)
print(f"Z-statistic: {z_stat:.4f}")
print(f"P-value: {p_value:.15f}")
print("-" * 75)

# Decision Rule: Reject H0 if P-value < alpha (0.05)
if p_value < 0.05:
    print("Conclusion: Reject H0. The difference in claim frequency is statistically significant.")
    print("We conclude that the Claim Frequency in Gauteng is significantly HIGHER than in Northern Cape.")
else:
    print("Conclusion: Fail to Reject H0. The difference is not statistically significant.")