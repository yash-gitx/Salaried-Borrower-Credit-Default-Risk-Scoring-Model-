import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Finnable Credit Risk & Underwriting Engine", layout="wide")

@st.cache_data
def load_scored_data():
    return pd.read_csv("finnable_scored_portfolio.csv")

df = load_scored_data()

st.title("Finnable — Salaried Borrower Credit Risk & PD Underwriting Simulator")
st.caption("Machine Learning Credit Scorecard (XGBoost + SHAP) | N = 60,000 Salaried Loan Cohort")

# Sidebar: Dynamic Policy Cutoff Controls
st.sidebar.header("Underwriting Policy Controls")
max_pd_cutoff = st.sidebar.slider("Maximum Auto-Approval PD Cutoff (%)", 2.0, 20.0, 9.5, 0.5) / 100.0
grade_c_haircut = st.sidebar.slider("Principal Haircut for Grade C Borrowers (%)", 0, 50, 25, 5)
lgd_assumption = st.sidebar.slider("Loss Given Default (LGD %)", 40, 90, 65, 5) / 100.0

# Evaluate Legacy Policy vs ML Policy
legacy_approved = df[(df["cibil_score"] >= 700) & (df["foir_pct"] <= 55.0)]
ml_approved = df[df["predicted_pd"] <= max_pd_cutoff].copy()
ml_approved["sanctioned_inr"] = np.where(
    ml_approved["risk_grade"] == "Grade C (Near-Prime)",
    ml_approved["loan_amount_inr"] * (1 - grade_c_haircut / 100.0),
    ml_approved["loan_amount_inr"]
)

leg_app_rate = len(legacy_approved) / len(df) * 100
ml_app_rate = len(ml_approved) / len(df) * 100

leg_bad_rate = legacy_approved["default_90dpd"].mean() * 100
ml_bad_rate = ml_approved["default_90dpd"].mean() * 100

leg_loss_cr = (legacy_approved["loan_amount_inr"] * legacy_approved["default_90dpd"] * lgd_assumption).sum() / 1e7
ml_loss_cr = (ml_approved["sanctioned_inr"] * ml_approved["default_90dpd"] * lgd_assumption).sum() / 1e7

k1, k2, k3, k4 = st.columns(4)
k1.metric("ML Approval Rate", f"{ml_app_rate:.1f}%", f"{ml_app_rate - leg_app_rate:+.1f}% vs Legacy")
k2.metric("Portfolio 90+ DPD Rate", f"{ml_bad_rate:.2f}%", f"{ml_bad_rate - leg_bad_rate:+.2f}% vs Legacy", delta_color="inverse")
k3.metric("Expected Credit Loss (ECL)", f"₹{ml_loss_cr:.2f} Cr", f"₹{ml_loss_cr - leg_loss_cr:+.2f} Cr vs Legacy", delta_color="inverse")
k4.metric("Sanctioned Disbursal Book", f"₹{ml_approved['sanctioned_inr'].sum()/1e7:.2f} Cr")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Default Rate & Volume Across Risk Grades")
    grade_summary = df.groupby("risk_grade", observed=False).agg(
        loans=("loan_id", "count"),
        bad_rate=("default_90dpd", lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    fig_grade = px.bar(
        grade_summary, x="risk_grade", y="bad_rate", text="bad_rate",
        color="risk_grade", title="Empirical 90+ DPD Rate (%) by Finnable Risk Grade"
    )
    st.plotly_chart(fig_grade, use_container_width=True)

with col2:
    st.subheader("2. Swap-Set Matrix: Legacy Policy vs ML Scorecard")
    df["legacy_decision"] = np.where((df["cibil_score"] >= 700) & (df["foir_pct"] <= 55.0), "Legacy Approve", "Legacy Reject")
    df["ml_decision"] = np.where(df["predicted_pd"] <= max_pd_cutoff, "ML Approve", "ML Reject")
    swap_df = df.groupby(["legacy_decision", "ml_decision"]).agg(
        borrowers=("loan_id", "count"),
        bad_rate_pct=("default_90dpd", lambda x: round(x.mean() * 100, 2)),
        volume_cr=("loan_amount_inr", lambda x: round(x.sum() / 1e7, 2))
    ).reset_index()
    st.dataframe(swap_df, use_container_width=True)
    st.caption("**Swap-In (Legacy Reject -> ML Approve):** Low-default thin-file borrowers unlocked. **Swap-Out (Legacy Approve -> ML Reject):** High-default credit-hungry borrowers blocked.")