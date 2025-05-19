import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, explained_variance_score

# === 1. Data Preparation ===
df = pd.read_excel("data.xlsx")
df.rename(columns={
    '碳排放效率': 'CE',
    '人工智能发展水平': 'AI',
    '能源利用效率': 'ENE',
    '人口规模': 'POP',
    '经济发展水平': 'GDP',
    '产业结构': 'IND',
    '政府宏观调控水平': 'GOV',
    '对外开放程度': 'OPEN',
    '年份': 'Year',
    '省份': 'Province'
}, inplace=True)

features = ['AI', 'ENE', 'POP', 'GDP', 'IND', 'GOV', 'OPEN']
target_years = [2021, 2022]

# === 2. Linear Regression to Forecast Feature Values (Based on 2013–2020) ===
predictions = []

for province in df['Province'].unique():
    df_prov = df[df['Province'] == province]
    for feat in features:
        df_feat = df_prov[['Year', feat]]
        df_train = df_feat[df_feat['Year'] <= 2020]
        X_train = df_train['Year'].values.reshape(-1, 1)
        y_train = df_train[feat].values

        if len(X_train) < 2:
            continue

        model = LinearRegression()
        model.fit(X_train, y_train)

        for year in target_years:
            pred_value = model.predict(np.array([[year]]))[0]
            predictions.append({
                "Province": province,
                "Year": year,
                "Feature": feat,
                "Predicted_Value": pred_value
            })

# === 3. Reshape to Wide Format ===
df_pred = pd.DataFrame(predictions)
df_wide = df_pred.pivot(index=['Province', 'Year'], columns='Feature', values='Predicted_Value').reset_index()

# === 4. Load Trained Model and Predict Carbon Efficiency (CE) ===
xgb_model = joblib.load("xgb_model_eval.pkl")

X_pred = df_wide[features]
df_wide["CE_Predicted"] = xgb_model.predict(X_pred)

# === 5. Merge with Actual Values and Evaluate Performance ===
true_ce = df[df['Year'].isin(target_years)][['Province', 'Year', 'CE']]
merged = pd.merge(df_wide, true_ce, on=['Province', 'Year'], how='inner')

y_true = merged['CE']
y_pred = merged['CE_Predicted']

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
evs = explained_variance_score(y_true, y_pred)

print(f"R²     = {r2:.4f}")
print(f"MAE    = {mae:.4f}")
print(f"EVS    = {evs:.4f}")
