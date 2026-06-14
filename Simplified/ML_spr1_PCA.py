import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.decomposition import PCA

df = pd.read_csv("health_dataset.csv")
display(df.head())

# Encode categoricals
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

TARGET = 'Disease Risk Score'
if TARGET not in df.columns: raise ValueError("Target column not found")

X, y = df.drop(columns=[TARGET]), df[TARGET]
if y.isnull().any(): y.fillna(y.mode()[0], inplace=True)
if y.dtype == 'object': y = LabelEncoder().fit_transform(y)

# Feature selection
X = VarianceThreshold(0).fit_transform(X)
selector = SelectKBest(f_classif, k=min(8, X.shape[1])).fit(X, y)
X_sel = selector.transform(X)
print("Selected Features:", [c for c,k in zip(df.drop(columns=[TARGET]).columns, selector.get_support()) if k])

# Normalize → Transform → PCA
X_pca = PCA(n_components=min(5, X_sel.shape[1])).fit_transform(
         PowerTransformer('yeo-johnson').fit_transform(
         MinMaxScaler().fit_transform(X_sel)))

print("Explained Variance:", PCA(n_components=min(5,X_sel.shape[1])).fit(
      PowerTransformer('yeo-johnson').fit_transform(
      MinMaxScaler().fit_transform(X_sel))).explained_variance_ratio_)

out = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
out[TARGET] = y.reset_index(drop=True) if hasattr(y,'reset_index') else y
out.to_csv("processed_health_dataset.csv", index=False)
print("Saved.")