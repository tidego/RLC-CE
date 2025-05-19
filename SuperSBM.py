import pandas as pd
import numpy as np
from scipy.optimize import linprog

# ========== 1. Data Loading and Cleaning ==========
file_path = "SuperSBM_input.xlsx"

# Model variable selection:
# Inputs (X):
# - Fixed capital stock (Kt)
# - Labor input (10,000 people) (Lt)
# - Energy input (10,000 tons of standard coal) (Et)
# Desired Output (Yd): Regional GDP (GDPt)
# Undesired Output (Yu): CO₂ emissions (CO₂t)
input_vars = ["固定资本存量", "劳动力投入（万人）", "能源投入（万吨标准煤）"]
desired_output = ["地区生产总值（亿元）"]
undesired_output = ["二氧化碳排放总量(万吨)"]

# Read file
df = pd.read_excel(file_path)

# Standardize variable formats and remove whitespace
df["年份"] = df["年份"].astype(int)
df["省份"] = df["省份"].astype(str).str.strip().str.replace(r"\s+", "", regex=True)

# Construct Decision Making Units (DMUs)
df["单位"] = df["省份"] + "-" + df["年份"].astype(str)

# ========== 2. Extract Required Fields for the Model ==========
cols = ["单位"] + input_vars + desired_output + undesired_output
sbm_df = df[cols].dropna().reset_index(drop=True)
sbm_df.set_index("单位", inplace=True)

# ========== 3. Build SBM Model Data Structures ==========
X = sbm_df[input_vars].values         # Input matrix
Yd = sbm_df[desired_output].values    # Desired outputs
Yu = sbm_df[undesired_output].values  # Undesired outputs

# Merge into a single DataFrame (optional for inspection)
sbm_matrix = pd.DataFrame(
    np.hstack([X, Yd, Yu]),
    columns=input_vars + desired_output + undesired_output,
    index=sbm_df.index
)

# Matrix dimensions
x = X.T            # shape: (m, n)
y_g = Yd.T         # shape: (s1, n)
y_b = Yu.T         # shape: (s2, n)

m, n = x.shape
s1 = y_g.shape[0]
s2 = y_b.shape[0]

# ========== SBM Efficiency Calculation ==========
sbm_eff = []
for i in range(n):
    try:
        f = np.concatenate([
            np.zeros(n),
            -1 / (m * x[:, i]),
            np.zeros(s1 + s2),
            np.array([1])
        ])
        Aeq1 = np.hstack([x,
                          np.identity(m),
                          np.zeros((m, s1 + s2)),
                          -x[:, i, None]])
        Aeq2 = np.hstack([y_g,
                          np.zeros((s1, m)),
                          -np.identity(s1),
                          np.zeros((s1, s2)),
                          -y_g[:, i, None]])
        Aeq3 = np.hstack([y_b,
                          np.zeros((s2, m)),
                          np.zeros((s2, s1)),
                          np.identity(s2),
                          -y_b[:, i, None]])
        Aeq4 = np.hstack([np.zeros(n),
                          np.zeros(m),
                          1 / ((s1 + s2) * y_g[:, i]),
                          1 / ((s1 + s2) * y_b[:, i]),
                          np.array([1])]).reshape(1, -1)
        Aeq = np.vstack([Aeq1, Aeq2, Aeq3, Aeq4])
        beq = np.concatenate([np.zeros(m + s1 + s2), np.array([1])])
        bounds = [(0, None)] * (n + m + s1 + s2 + 1)

        res = linprog(c=f, A_eq=Aeq, b_eq=beq, bounds=bounds, method='highs')
        sbm_eff.append(res.fun if res.success else np.nan)
    except:
        sbm_eff.append(np.nan)

# ========== 4. Super-SBM Efficiency Calculation ==========
super_eff = [np.nan] * n
for i in range(n):
    if not np.isclose(sbm_eff[i], 1, atol=1e-4):
        continue
    try:
        # Exclude the evaluated DMU
        x_ref = np.delete(x, i, axis=1)
        y_g_ref = np.delete(y_g, i, axis=1)
        y_b_ref = np.delete(y_b, i, axis=1)
        n_ref = x_ref.shape[1]

        x_i = x[:, i]
        y_g_i = y_g[:, i]
        y_b_i = y_b[:, i]

        f = np.concatenate([
            np.zeros(n_ref),
            1 / (m * x_i),
            np.zeros(s1 + s2),
            np.array([1])
        ])
        Aeq = np.hstack([
            np.zeros(n_ref),
            np.zeros(m),
            -1 / ((s1 + s2) * y_g_i),
            -1 / ((s1 + s2) * y_b_i),
            np.array([1])
        ]).reshape(1, -1)
        beq = np.array([1])

        Aub1 = np.hstack([
            x_ref,
            -np.identity(m),
            np.zeros((m, s1 + s2)),
            -x_i.reshape(-1, 1)
        ])
        Aub2 = np.hstack([
            -y_g_ref,
            np.zeros((s1, m)),
            -np.identity(s1),
            np.zeros((s1, s2)),
            y_g_i.reshape(-1, 1)
        ])
        Aub3 = np.hstack([
            y_b_ref,
            np.zeros((s2, m)),
            np.zeros((s2, s1)),
            -np.identity(s2),
            -y_b_i.reshape(-1, 1)
        ])
        Aub = np.vstack([Aub1, Aub2, Aub3])
        bub = np.zeros(m + s1 + s2)
        bounds = [(0, None)] * (n_ref + m + s1 + s2 + 1)

        res = linprog(c=f, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                      bounds=bounds, method='highs')
        super_eff[i] = res.fun if res.success else 1.0
    except:
        super_eff[i] = 1.0

# Combine SBM and Super-SBM results
final_eff = [sup if sup > 1 else sbm for sbm, sup in zip(sbm_eff, super_eff)]

# Construct result DataFrame
eff_df = pd.DataFrame({
    "单位": sbm_df.index,
    "碳排放效率 (ρ*)": final_eff
})

# Merge efficiency back into original df (by "单位")
df["单位"] = df["省份"].astype(str).str.strip().str.replace(r"\s+", "", regex=True) + "-" + df["年份"].astype(str)
df = df.merge(eff_df, on="单位", how="left")

# Save to file
df.to_excel("./SuperSBM_output.xlsx", index=False)
print("SuperSBM efficiency calculation completed.")