# preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

# データの読み込み
df = pd.read_csv('iris_data.csv')

# 標準化
scaler = StandardScaler()
features = df.drop(columns=['target'])
scaled_features = scaler.fit_transform(features)

# 前処理後のデータを保存
processed_df = pd.DataFrame(scaled_features, columns=features.columns)
processed_df['target'] = df['target']
processed_df.to_csv('processed_iris_data.csv', index=False)
