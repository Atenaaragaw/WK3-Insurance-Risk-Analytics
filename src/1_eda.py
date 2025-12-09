import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os # <-- Added for file system operations

# Set a style for better visualization
sns.set_style('whitegrid')

# --- MLOps: Create visuals directory ---
VISUALS_DIR = 'visuals'
if not os.path.exists(VISUALS_DIR):
    os.makedirs(VISUALS_DIR)
    print(f"Created directory: {VISUALS_DIR}/")


# Load the data using the pipe separator
DATA_PATH = 'data/MachineLearningRating_v3.txt'
try:
    # Use the pipe '|' as the separator, suppressing the low_memory warning for simplicity
    df = pd.read_csv(DATA_PATH, sep='|', low_memory=False)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()

print("--- 1. Data Structure and Quality Check ---")
print(f"Initial Data Size: {len(df)} rows, {len(df.columns)} columns.")

# --- Data Cleaning and Type Correction ---

# We use the existing 'TransactionMonth' column and rename the result to 'TransactionDate'
df['TransactionDate'] = pd.to_datetime(df['TransactionMonth'])

# Coerce to numeric, setting non-convertible characters to NaN
df['CapitalOutstanding'] = pd.to_numeric(df['CapitalOutstanding'], errors='coerce')

# These columns have too many missing values to be useful for initial EDA or modeling.
df = df.drop(columns=['NumberOfVehiclesInFleet', 'CrossBorder'])

# Create a binary Claim Flag for frequency analysis
df['ClaimFlag'] = np.where(df['TotalClaims'] > 0, 1, 0)


# --- 2. Descriptive Statistics (KPIs) ---

print("\n--- 2. Descriptive Statistics for Numerical Features ---")
numerical_features = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate', 'SumInsured']
print(df[numerical_features].describe(percentiles=[0.25, 0.5, 0.75, 0.99]).T)


# --- 3. Bivariate Analysis & Business Metrics (Loss Ratio) ---

# Function to calculate aggregated LR
def calculate_agg_lr(data, segment_col):
    agg = data.groupby(segment_col).agg(
        TotalClaims=('TotalClaims', 'sum'),
        TotalPremium=('TotalPremium', 'sum'),
        PolicyCount=('PolicyID', 'nunique')
    )
    # Filter out segments with zero premium
    agg = agg[agg['TotalPremium'] > 0] 
    agg['LossRatio'] = agg['TotalClaims'] / agg['TotalPremium']
    return agg.sort_values('LossRatio', ascending=False)

# 1. Overall Loss Ratio
overall_lr = df['TotalClaims'].sum() / df['TotalPremium'].sum()
print(f"\n--- 3. Key Business Metrics ---")
print(f"Overall Portfolio Loss Ratio: {overall_lr:.4f}")

# 2. Loss Ratio by Province (Guiding Question)
province_lr = calculate_agg_lr(df, 'Province')
print("\nLoss Ratio by Province (Top & Bottom 3):")
print(province_lr[['LossRatio']].head(3)) 
print(province_lr[['LossRatio']].tail(3))

# 3. Loss Ratio by Gender (Guiding Question)
gender_lr = calculate_agg_lr(df.dropna(subset=['Gender']), 'Gender')
print("\nLoss Ratio by Gender:")
print(gender_lr[['LossRatio']])


# --- 4. Visualization (All Required Plots) ---

# Creative Plot 1: Loss Ratio by Province 
plt.figure(figsize=(10, 6))
sns.barplot(x='Province', y='LossRatio', data=province_lr.reset_index(), palette='viridis')
plt.title('Loss Ratio Variation Across Provinces', fontsize=14)
plt.xlabel('Province')
plt.ylabel('Loss Ratio (Risk Measure)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'lr_by_province.png')) # <-- Saved Plot
plt.show()

# Creative Plot 2: Temporal Trend 
df['TransactionMonth_str'] = df['TransactionDate'].dt.to_period('M').astype(str)
monthly_trends = df.groupby('TransactionMonth_str').agg(
    TotalClaims=('TotalClaims', 'sum'),
    TotalPremium=('TotalPremium', 'sum')
)
# Recalculate LR on the aggregated monthly data
monthly_trends['LossRatio'] = monthly_trends['TotalClaims'] / monthly_trends['TotalPremium']


plt.figure(figsize=(10, 5))
sns.lineplot(x=monthly_trends.index, y='LossRatio', data=monthly_trends, marker='o', color='red')
plt.title('Temporal Trend of Portfolio Loss Ratio (Feb 2014 - Aug 2015)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig(os.path.join(VISUALS_DIR, 'temporal_loss_ratio.png')) # <-- Saved Plot
plt.show()

# Creative Plot 3: Top 5 Vehicle Makes by Average Claim Severity
make_severity = df[df['ClaimFlag'] == 1].groupby('make').agg(
    AvgClaim=('TotalClaims', 'mean'),
    ClaimCount=('PolicyID', 'nunique')
)
# Filter for top 10 most claimed-upon makes, then sort by highest AvgClaim
top_make_severity = make_severity.nlargest(10, 'ClaimCount').nlargest(5, 'AvgClaim')

plt.figure(figsize=(10, 6))
sns.barplot(x=top_make_severity.index, y='AvgClaim', data=top_make_severity.reset_index(), palette='coolwarm')
plt.title('Top 5 Vehicle Makes by Average Claim Severity', fontsize=14)
plt.xlabel('Vehicle Make')
plt.ylabel('Average Claim Amount (R)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(VISUALS_DIR, 'top_makes_severity.png')) # <-- Saved Plot
plt.show()


# --- Missing Standard EDA Elements (Required by Feedback) ---

# A Correlation Matrix (Heatmap)
numerical_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'RegistrationYear']
corr_matrix = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Key Numerical Features')
plt.savefig(os.path.join(VISUALS_DIR, 'correlation_heatmap.png')) # <-- Saved Plot
plt.show()


# B Histogram for TotalClaims
plt.figure(figsize=(10, 5))
sns.histplot(np.log1p(df['TotalClaims']), bins=50, kde=True)
plt.title('Distribution of Log(1 + TotalClaims)')
plt.xlabel('Log(1 + TotalClaims)')
plt.savefig(os.path.join(VISUALS_DIR, 'claims_log_histogram.png')) # <-- Saved Plot
plt.show()

# C Box Plot for TotalPremium by Gender
plt.figure(figsize=(10, 5))
sns.boxplot(x='Gender', y='TotalPremium', data=df)
plt.title('Total Premium Distribution by Gender')
plt.savefig(os.path.join(VISUALS_DIR, 'premium_by_gender_boxplot.png')) # <-- Saved Plot
plt.show()