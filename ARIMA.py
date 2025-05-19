# =============================================================
# 0. Environment Setup
# -------------------------------------------------------------
import os
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
import statsmodels.api as sm

# Set global style
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid", palette="deep", font_scale=1.2)

# Output directory
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================
# 1. Data Loading and Mapping
# -------------------------------------------------------------
df = pd.read_excel("data.xlsx")

# =============================================================
# 2. Functions: ADF + ARIMA + Evaluation + Forecast
# -------------------------------------------------------------
def adf_test(series, alpha=0.05):
    pval = adfuller(series, autolag='AIC')[1]
    return pval, pval < alpha


def arima_pipeline(province, df_all, forecast_periods=5, save_plots=True):
    ts = df_all[df_all["省份"] == province].sort_values("年份")
    ts = ts.set_index("年份")["碳排放效率"]
    ts.index = pd.PeriodIndex(ts.index, freq='A')

    pval, stationary = adf_test(ts)
    print(f"[{province}] ADF p = {pval:.4f} → {'Stationary' if stationary else 'Non-stationary'}")

    model = pm.auto_arima(ts,
                          start_p=0, start_q=0,
                          max_p=5, max_q=5,
                          d=None, seasonal=False,
                          stepwise=True, information_criterion='aic',
                          suppress_warnings=True)
    order = model.order
    print(f"[{province}] ARIMA{order}")

    sm_model = sm.tsa.ARIMA(ts, order=order).fit()
    converged = sm_model.mle_retvals.get("converged", True)
    print(f"[{province}] {'Converged' if converged else 'Not converged'}")

    # Fit evaluation
    fitted = sm_model.fittedvalues
    mae = mean_absolute_error(ts, fitted)
    rmse = mean_squared_error(ts, fitted, squared=False)
    mape = np.mean(np.abs((ts - fitted) / ts)) * 100
    r2 = r2_score(ts, fitted)
    evs = explained_variance_score(ts, fitted)

    # Residual diagnostics plot
    if save_plots:
        lags = min(max(3, len(ts) - 2), 8)
        fig = sm_model.plot_diagnostics(figsize=(12, 8), lags=lags)
        fig.suptitle(f"Diagnostics of ARIMA{order} – {province}", fontsize=15)
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/{province}_diagnostics.png", dpi=440)
        plt.close()

    # Future forecasting
    pred = sm_model.get_forecast(steps=forecast_periods)
    pred_df = pred.summary_frame(alpha=0.05).rename(columns={
        'mean': 'Forecast_CE',
        'mean_ci_lower': 'CI_low',
        'mean_ci_upper': 'CI_high'
    })
    pred_df['Province'] = province
    pred_df['Year'] = pred_df.index.year
    pred_df.reset_index(drop=True, inplace=True)
    pred_df = pred_df[['Province', 'Year', 'Forecast_CE', 'CI_low', 'CI_high']]

    # Add error metrics
    for k, v in zip(['MAE', 'RMSE', 'MAPE', 'R2', 'Converged', 'EVS'],
                    [mae, rmse, mape, r2, converged, evs]):
        pred_df[k] = v

    # Forecast plot
    if save_plots:
        plt.figure(figsize=(9, 5))
        plt.plot(ts.index.year, ts.values, label="Observed", marker='o')
        plt.plot(pred_df['Year'], pred_df['Forecast_CE'], label="Forecast", marker='o')
        plt.fill_between(pred_df['Year'], pred_df['CI_low'], pred_df['CI_high'],
                         color='gray', alpha=0.3, label="95% CI")
        plt.title(f"{province}: ARIMA{order} 5-Year Forecast", fontsize=14)
        plt.xlabel("Year")
        plt.ylabel("Carbon Emission Efficiency")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{OUT_DIR}/{province}_forecast.png", dpi=440)
        plt.close()

    return pred_df

# =============================================================
# 3. Run and Save Results
# -------------------------------------------------------------
all_results = []
for prov in df["省份"].unique():
    pred_df = arima_pipeline(prov, df, forecast_periods=5, save_plots=False)
    all_results.append(pred_df)

results_df = pd.concat(all_results, ignore_index=True)
results_df.to_excel(f"{OUT_DIR}/ARIMA_5Year_Forecast.xlsx", index=False)

# Statistics
summary_df = results_df.drop_duplicates(subset="Province")
convergence_counts = summary_df["Converged"].value_counts()

print("\n====== Convergence Statistics ======")
print(f"Number of converged provinces: {convergence_counts.get(True, 0)}")
print(f"Number of non-converged provinces: {convergence_counts.get(False, 0)}")

print(f"\nAverage R² for all provinces: {summary_df['R2'].mean():.4f}")
print(f"Average EVS across all provinces: {summary_df['EVS'].mean():.4f}")
print(f"Average MAE across all provinces: {summary_df['MAE'].mean():.4f}")
