import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA

# Load the healthcare dataset
file_path = "health_dataset.csv"
df = pd.read_csv(file_path)

# Display first few rows
display(df.head())

# Identify non-numeric columns
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical Columns:", categorical_columns)

# Convert categorical columns using Label Encoding
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Ensure all columns are numeric
df = df.apply(pd.to_numeric, errors='coerce')   # Converts non-numeric values to NaN
df.fillna(0, inplace=True)                      # Replace NaNs with 0

# Separate features and target
target_column = 'Disease Risk Score'

if target_column not in df.columns:
    raise ValueError("Target column not found in dataset")

X = df.drop(columns=[target_column])
y = df[target_column]

# Check for missing values in target and encode if necessary
if y.isnull().sum() > 0:
    y.fillna(y.mode()[0], inplace=True)   # Replace NaN with most frequent value

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Remove constant features
var_thresh = VarianceThreshold(threshold=0)
X = var_thresh.fit_transform(X)

# Feature Selection using ANOVA F-score
selector = SelectKBest(
    score_func=f_classif,
    k=min(8, X.shape[1])   # Ensure k does not exceed feature count
)

X_selected = selector.fit_transform(X, y)

print(X_selected)

selected_features = [
    col
    for col, keep in zip(
        df.drop(columns=[target_column]).columns,
        selector.get_support()
    )
    if keep
]

print("Selected Features:", selected_features)

# Normalization using Min-Max Scaling
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_selected)

# Transformation using Power Transform (Box-Cox or Yeo-Johnson)
power_transformer = PowerTransformer(method='yeo-johnson')   # Use 'box-cox' if no negative values

X_transformed = power_transformer.fit_transform(X_normalized)

# Dimensionality Reduction using PCA
pca_components = min(5, X_transformed.shape[1])   # Ensure PCA components do not exceed feature count

pca = PCA(n_components=pca_components)

X_pca = pca.fit_transform(X_transformed)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Convert processed data back to DataFrame
processed_df = pd.DataFrame(
    X_pca,
    columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]
)

processed_df['Disease Risk Score'] = y.reset_index(drop=True)

# Save processed dataset
processed_df.to_csv("processed_health_dataset.csv", index=False)

print("Processed dataset saved successfully.")