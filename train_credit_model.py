import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# 1. Load Data & Define Features
df = pd.read_csv("finnable_credit_risk.csv")
num_cols = [
    "cibil_score", "foir_pct", "cc_utilization_pct", "inquiries_last_3m",
    "nach_bounce_12m", "salary_credit_cv", "avg_eod_balance_ratio",
    "emp_tenure_months", "monthly_salary_inr", "loan_amount_inr"
]
cat_cols = ["city_tier", "employer_category"]

X = df[num_cols + cat_cols]
y = df["default_90dpd"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 2. Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
feature_names = num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols))

# 3. Benchmark 1: Logistic Regression Credit Scorecard
lr_model = LogisticRegression(max_iter=1000, class_weight="balanced")
lr_model.fit(X_train_proc, y_train)
lr_preds = lr_model.predict_proba(X_test_proc)[:, 1]

# 4. Champion Model: XGBoost PD Classifier
xgb_model = xgb.XGBClassifier(
    n_estimators=280,
    max_depth=4,
    learning_rate=0.045,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    random_state=42
)
xgb_model.fit(X_train_proc, y_train)
xgb_preds = xgb_model.predict_proba(X_test_proc)[:, 1]

# 5. KS-Statistic & Decile Lift Table Function
def compute_ks_table(y_true, y_prob):
    res = pd.DataFrame({"actual": y_true, "pd": y_prob})
    res["decile"] = pd.qcut(res["pd"], 10, labels=False)
    res["decile"] = 10 - res["decile"]  # Decile 1 = Highest Risk
    table = res.groupby("decile").agg(
        loans=("actual", "count"),
        bads=("actual", "sum"),
        min_pd=("pd", "min"),
        max_pd=("pd", "max"),
        avg_pd=("pd", "mean")
    ).reset_index()
    table["goods"] = table["loans"] - table["bads"]
    table["bad_rate_pct"] = (table["bads"] / table["loans"] * 100).round(2)
    table["cum_bad_pct"] = (table["bads"].cumsum() / table["bads"].sum() * 100).round(2)
    table["cum_good_pct"] = (table["goods"].cumsum() / table["goods"].sum() * 100).round(2)
    table["ks_stat"] = (table["cum_bad_pct"] - table["cum_good_pct"]).round(2)
    return table

ks_table = compute_ks_table(y_test, xgb_preds)
print(f"Logistic Regression ROC-AUC: {roc_auc_score(y_test, lr_preds):.3f}")
print(f"XGBoost Champion ROC-AUC:    {roc_auc_score(y_test, xgb_preds):.3f}")
print(f"XGBoost Max KS-Statistic:    {ks_table['ks_stat'].max():.2f}%")
print(ks_table[["decile", "loans", "bads", "bad_rate_pct", "cum_bad_pct", "ks_stat"]])

# 6. SHAP Explainability for Adverse Action Reasons
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_proc[:500])

# Map Predicted PD to Finnable 300-900 Internal Risk Score & Grades A-E
df_all_proc = preprocessor.transform(X)
df["predicted_pd"] = xgb_model.predict_proba(df_all_proc)[:, 1].round(4)
df["finnable_risk_score"] = (650 - 75 * np.log(df["predicted_pd"] / (1 - df["predicted_pd"]))).astype(int).clip(300, 900)
df["risk_grade"] = pd.cut(
    df["predicted_pd"],
    bins=[0.0, 0.015, 0.035, 0.070, 0.140, 1.0],
    labels=["Grade A (Super-Prime)", "Grade B (Prime)", "Grade C (Near-Prime)", "Grade D (Review)", "Grade E (Auto-Reject)"]
)
df.to_csv("finnable_scored_portfolio.csv", index=False)