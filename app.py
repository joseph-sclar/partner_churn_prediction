import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance


st.set_page_config(page_title="Partner Churn – Overview · EDA · Training · Predict", page_icon="📉", layout="wide")



# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
PROB_MIN, PROB_MAX = 0.001, 0.999  # clamp displayed probabilities
FEATURE_COLS = [
    "uptime_minutes_30d","uptime_diff","gmv_30d","gmv_diff",
    "api_events_30d","avg_wall_position_30d","orders_30d","cancel_rate_30d"
]
EXTRA_COLS = [
    "support_contacts_30d","avg_prep_time_sec_30d","menu_updates_30d","delivery_issues_30d"
]
ALL_NUMERIC_FOR_EDA = FEATURE_COLS + EXTRA_COLS

# -------------------------------------------------------------------
# Data generator (kept inside the app; do not expose in UI)
# -------------------------------------------------------------------
def _jitter(rng, x, sigma):
    return x + rng.normal(0, sigma, size=x.shape)

@st.cache_data
def make_train_dataset(n_partners=3000, seed=42) -> pd.DataFrame:
    """
    Partner-level snapshot (last 30 days -> churn next 30 days).
    Churn prevalence ~16.03% (e.g., 481/3000).
    """
    rng = np.random.default_rng(seed)
    partner_ids = np.arange(100001, 100001 + n_partners)

    # Base (mixed) distributions
    uptime_minutes_30d = rng.normal(20000, 4200, size=n_partners).clip(0, None)
    uptime_diff        = rng.normal(0, 2000, size=n_partners)
    gmv_30d            = rng.normal(20000, 8000, size=n_partners).clip(0, None)
    gmv_diff           = rng.normal(0, 5000, size=n_partners)
    api_events_30d     = rng.poisson(lam=50, size=n_partners).astype(float)
    avg_wall_position  = rng.uniform(5, 40, size=n_partners)
    orders_30d         = (gmv_30d / rng.uniform(15, 25, size=n_partners)).astype(float).clip(0, None)
    cancel_rate_30d    = rng.uniform(0.03, 0.22, size=n_partners)

    # A few extra ops signals (not used in model, available for EDA)
    support_contacts_30d   = rng.poisson(lam=2, size=n_partners).astype(float)
    avg_prep_time_sec_30d  = rng.normal(900, 180, size=n_partners).clip(300, 1800)   # ~5–30 min
    menu_updates_30d       = rng.poisson(lam=3, size=n_partners).astype(float)
    delivery_issues_30d    = rng.poisson(lam=4, size=n_partners).astype(float)

    # Assign churners (~16.03%)
    churn_count = int(round(n_partners * 0.1603))
    churn_idx = rng.choice(n_partners, size=churn_count, replace=False)

    # Shift churners (keep overlap)
    uptime_minutes_30d[churn_idx] -= rng.normal(2500, 2500, size=churn_count)
    uptime_diff[churn_idx]        -= rng.normal(1500, 1200, size=churn_count)
    gmv_30d[churn_idx]            -= rng.normal(5000, 4000, size=churn_count)
    gmv_diff[churn_idx]           -= rng.normal(3500, 2500, size=churn_count)
    api_events_30d[churn_idx]     -= rng.poisson(lam=12, size=churn_count)
    avg_wall_position[churn_idx]  += rng.uniform(6, 12, size=churn_count)
    orders_30d[churn_idx]          = (gmv_30d[churn_idx] / rng.uniform(17, 27, size=churn_count)).clip(0, None)
    cancel_rate_30d[churn_idx]    += rng.uniform(0.06, 0.12, size=churn_count)

    # Mild signal in extra columns (still messy)
    support_contacts_30d[churn_idx]  += rng.poisson(lam=1, size=churn_count)
    avg_prep_time_sec_30d[churn_idx] += rng.normal(60, 60, size=churn_count)  # slightly higher
    menu_updates_30d[churn_idx]      += rng.normal(-0.5, 1.0, size=churn_count)  # slightly fewer
    delivery_issues_30d[churn_idx]   += rng.poisson(lam=2, size=churn_count)

    # Global jitter to avoid perfect patterns
    uptime_minutes_30d = np.clip(_jitter(rng, uptime_minutes_30d, 500), 0, None)
    uptime_diff        = _jitter(rng, uptime_diff, 300)
    gmv_30d            = np.clip(_jitter(rng, gmv_30d, 1200), 0, None)
    gmv_diff           = _jitter(rng, gmv_diff, 800)
    api_events_30d     = np.clip(np.round(_jitter(rng, api_events_30d, 4)).astype(int), 0, None).astype(float)
    avg_wall_position  = np.clip(_jitter(rng, avg_wall_position, 1.5), 1, 50)
    orders_30d         = np.clip(np.round(_jitter(rng, orders_30d, 5)).astype(int), 0, None).astype(float)
    cancel_rate_30d    = np.clip(_jitter(rng, cancel_rate_30d, 0.01), 0, 1)

    support_contacts_30d  = np.clip(np.round(_jitter(rng, support_contacts_30d, 0.8)), 0, None)
    avg_prep_time_sec_30d = np.clip(_jitter(rng, avg_prep_time_sec_30d, 20), 300, 2400)
    menu_updates_30d      = np.clip(np.round(_jitter(rng, menu_updates_30d, 0.7)), 0, None)
    delivery_issues_30d   = np.clip(np.round(_jitter(rng, delivery_issues_30d, 0.9)), 0, None)

    churn_next_30d = np.zeros(n_partners, dtype=int)
    churn_next_30d[churn_idx] = 1

    df = pd.DataFrame({
        "partner_id": partner_ids,
        "uptime_minutes_30d": np.round(uptime_minutes_30d).astype(int),
        "uptime_diff": np.round(uptime_diff).astype(int),
        "gmv_30d": np.round(gmv_30d, 2),
        "gmv_diff": np.round(gmv_diff, 2),
        "api_events_30d": api_events_30d.astype(int),
        "avg_wall_position_30d": np.round(avg_wall_position, 1),
        "orders_30d": orders_30d.astype(int),
        "cancel_rate_30d": np.round(cancel_rate_30d, 3),
        # extra (for EDA only)
        "support_contacts_30d": support_contacts_30d.astype(int),
        "avg_prep_time_sec_30d": np.round(avg_prep_time_sec_30d).astype(int),
        "menu_updates_30d": menu_updates_30d.astype(int),
        "delivery_issues_30d": delivery_issues_30d.astype(int),
        "churn_next_30d": churn_next_30d
    })
    return df

@st.cache_data
def make_prod_dataset(n_partners=1200, seed=777) -> pd.DataFrame:
    df = make_train_dataset(n_partners=n_partners, seed=seed).copy()
    return df.drop(columns=["churn_next_30d"], errors="ignore")

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def clamp_probs(p: np.ndarray, lo=PROB_MIN, hi=PROB_MAX) -> np.ndarray:
    return np.clip(p, lo, hi)

def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, calibrate=True):
    X = df[FEATURE_COLS].copy()
    y = df["churn_next_30d"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                        test_size=test_size, random_state=random_state)
    base = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced_subsample", random_state=random_state, n_jobs=-1
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3) if calibrate else base
    model.fit(X_train, y_train)

    proba_test = clamp_probs(model.predict_proba(X_test)[:, 1])
    roc = roc_auc_score(y_test, proba_test)
    pr, rc, th = precision_recall_curve(y_test, proba_test)
    pr_auc = auc(rc, pr)
    return model, (X_train, X_test, y_train, y_test), (proba_test, roc, (pr, rc, th, pr_auc))

def confusion_and_report(y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    rep = classification_report(y_true, y_pred, target_names=["kept","churned"], digits=3, zero_division=0)
    return cm, rep

def risk_band(p): return "High" if p>=0.7 else ("Medium" if p>=0.4 else "Low")

# -------------------------------------------------------------------
# Session state (fixed data)
# -------------------------------------------------------------------
if "df_train" not in st.session_state:
    st.session_state.df_train = make_train_dataset()
if "df_prod" not in st.session_state:
    st.session_state.df_prod = make_prod_dataset()
if "model_pack" not in st.session_state:
    st.session_state.model_pack = None
if "scores_prod" not in st.session_state:
    st.session_state.scores_prod = None


if "model_pack" not in st.session_state or st.session_state.model_pack is None:
    model, splits, evals = train_model(st.session_state.df_train, test_size=0.2, calibrate=True)
    st.session_state.model_pack = (model, splits, evals)

df_train = st.session_state.df_train
df_prod  = st.session_state.df_prod

# -------------------------------------------------------------------
# Sidebar – only navigation
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Go to", ["0) Overview", "1) EDA", "2) Training", "3) Predict"], index=0)

# -------------------------------------------------------------------
# PAGE 0 — Overview
# -------------------------------------------------------------------
if page.startswith("0"):
    st.title("Overview")
    st.write(
        """
        **Goal**  
        Estimate the probability that a partner will churn in the next 30 days using their last 30 days of activity.

        **How it works**  
        1) **EDA:** explore partner activity and see how behavior differs between churned vs kept.  
        2) **Training:** fit the model, check metrics, tune the alert threshold.  
        3) **Predict:** score the live partner list, review top risk, and drill into a single partner.

        **Signals used (last 30 days)**  
        - `uptime_minutes_30d` – minutes online  
        - `uptime_diff` – change vs previous 30 days  
        - `gmv_30d` – gross merchandise value  
        - `gmv_diff` – change vs previous 30 days  
        - `api_events_30d` – app/API actions  
        - `avg_wall_position_30d` – average ranking/visibility (lower is better)  
        - `orders_30d` – completed orders  
        - `cancel_rate_30d` – cancellation rate

        **Model**  
        Random Forest with probability calibration. Threshold is adjustable to balance recall and precision based on business needs.

        **Risk bands**  
        - Low: < 40%  
        - Medium: 40–69%  
        - High: ≥ 70%
        """
    )

    # Small snapshot
    c1, c2, c3 = st.columns(3)
    c1.metric("Partners (train)", f"{len(df_train):,}")
    c2.metric("Churn rate (train)", f"{df_train['churn_next_30d'].mean():.2%}")
    c3.metric("Partners (predict)", f"{len(df_prod):,}")

# -------------------------------------------------------------------
# PAGE 1 — EDA
# -------------------------------------------------------------------
elif page.startswith("1"):
    st.title("📊 EDA — Explore training data")
    st.caption("Compare distributions for churned vs kept partners.")

    # Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Partners (train)", f"{len(df_train):,}")
    c2.metric("Churn rate", f"{df_train['churn_next_30d'].mean():.2%}")
    c3.metric("Median GMV (30d)", f"{df_train['gmv_30d'].median():.0f}")
    c4.metric("Median orders (30d)", f"{df_train['orders_30d'].median():.0f}")

    st.subheader("Distributions by churn status")
    var = st.selectbox("Pick a variable", ALL_NUMERIC_FOR_EDA, index=0)
    kind = st.radio("Plot type", ["Histogram", "Box"], horizontal=True)

    if kind == "Histogram":
        fig = px.histogram(
            df_train, x=var, color=df_train["churn_next_30d"].map({0:"kept",1:"churned"}),
            nbins=40, barmode="overlay", opacity=0.6,
            title=f"{var} by churn status", labels={"color":"status"}
        )
    else:
        fig = px.box(
            df_train, x=df_train["churn_next_30d"].map({0:"kept",1:"churned"}),
            y=var, points="all", title=f"{var} by churn status", labels={"x":"status"}
        )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation with churn (quick view)")
    cors = []
    y = df_train["churn_next_30d"].astype(int)
    for col in ALL_NUMERIC_FOR_EDA:
        s = df_train[col].astype(float)
        if s.std() == 0:
            corr = 0.0
        else:
            corr = np.corrcoef((s - s.mean()) / (s.std() + 1e-9), y)[0,1]
        cors.append({"feature": col, "corr_with_churn": corr})
    st.dataframe(pd.DataFrame(cors).sort_values("corr_with_churn", ascending=False), use_container_width=True)

    st.subheader("Raw preview")
    st.dataframe(df_train.head(25), use_container_width=True, height=320)

# -------------------------------------------------------------------
# PAGE 2 — Training
# -------------------------------------------------------------------
elif page.startswith("2"):
    st.title("🧠 Training — Fit & evaluate")
    st.caption("Adjust the decision threshold to see the confusion matrix update.")

    left, right = st.columns([1, 1])
    with left:
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
        threshold = st.slider("Decision threshold (flag as churn if ≥)", 0.05, 0.9, 0.4, 0.05)
        train_btn = st.button("Train / Retrain")

    if (st.session_state.model_pack is None) or train_btn:
        model, splits, evals = train_model(df_train, test_size=test_size, calibrate=True)
        st.session_state.model_pack = (model, splits, evals)

    model, (X_train, X_test, y_train, y_test), (proba_test, roc, (pr, rc, th, pr_auc)) = st.session_state.model_pack

    m1, m2 = right.columns(2)
    m1.metric("ROC-AUC", f"{roc:.3f}")
    m2.metric("PR-AUC", f"{pr_auc:.3f}")

    fig_pr = px.area(x=rc, y=pr, title="Precision–Recall curve", labels={"x":"Recall","y":"Precision"})
    st.plotly_chart(fig_pr, use_container_width=True)

    cm, rep = confusion_and_report(y_test, proba_test, threshold)
    c0, c1 = st.columns(2)
    with c0:
        st.markdown("**Confusion matrix (test)** — rows=actual, cols=predicted\n`[[TN FP],[FN TP]]`")
        st.code(cm)
    with c1:
        st.markdown("**Classification report (test)**")
        st.code(rep)

    # Feature importances
    try:
        pi = permutation_importance(
            model, X_test, y_test,
            scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1
        )
        pi_mean = pd.Series(pi.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)
        fig_imp = px.bar(pi_mean, title="Permutation importance (AUC drop when shuffled)")
        st.plotly_chart(fig_imp, use_container_width=True)

    except Exception:
        st.info("Feature importances not available.")

# -------------------------------------------------------------------
# PAGE 3 — Predict
# -------------------------------------------------------------------
else:
    st.title("🔮 Predict — Score partners")
    st.caption("Score the partner list, review top risk, and drill into a single partner.")

    if st.session_state.model_pack is None:
        st.warning("Please train a model first (go to the Training page).")
        st.stop()
    model, _, _ = st.session_state.model_pack

    Xp = df_prod[FEATURE_COLS].copy()
    proba_raw = model.predict_proba(Xp)[:, 1]
    proba = clamp_probs(proba_raw)
    scored = df_prod[["partner_id"] + FEATURE_COLS].copy()
    scored["churn_score"] = proba
    scored["risk_band"] = scored["churn_score"].apply(risk_band)
    st.session_state.scores_prod = scored

    c1, c2, c3 = st.columns(3)
    c1.metric("Partners (predict)", f"{len(scored):,}")
    c2.metric("Mean churn prob.", f"{scored['churn_score'].mean():.1%}")
    c3.metric("High risk (≥70%)", f"{(scored['churn_score']>=0.7).mean():.1%}")

    st.subheader("Top at-risk partners")
    topk = st.slider("Show top K", 10, 300, 50, 10)
    st.dataframe(scored.sort_values("churn_score", ascending=False).head(topk), use_container_width=True, height=360)

    # Download
    csv = scored[["partner_id","churn_score","risk_band"]].sort_values("churn_score", ascending=False).to_csv(index=False).encode()
    st.download_button("⬇️ Download scores (CSV)", data=csv, file_name="partner_churn_scores.csv", mime="text/csv")

    st.subheader("Partner lookup & comparison")
    colA, colB = st.columns([1,2])
    with colA:
        pid = st.text_input("Enter partner_id", value=str(int(scored['partner_id'].iloc[0])))
        threshold = st.slider("Decision threshold (flag as churn if ≥)", 0.05, 0.9, 0.4, 0.05, key="thr_pred")
        go = st.button("Check")

    with colB:
        if go:
            try:
                pid_val = int(pid)
                row = scored.loc[scored["partner_id"] == pid_val]
                if row.empty:
                    st.error("Partner not found.")
                else:
                    prob = float(np.clip(row["churn_score"].iloc[0], PROB_MIN, PROB_MAX))
                    band = row["risk_band"].iloc[0]
                    st.metric(label=f"Partner {pid_val} — churn probability (next 30d)", value=f"{prob:.2%}")
                    st.write(f"**Risk band:** {band} | **Flagged?** {'Yes' if prob>=threshold else 'No'}")

                    # Compare against TRAIN medians for context
                    key_feats = FEATURE_COLS
                    comp = pd.DataFrame({
                        "feature": key_feats,
                        "partner_value": [float(row[f].iloc[0]) for f in key_feats],
                        "train_median": [float(df_train[f].median()) for f in key_feats]
                    })
                    fig_cmp = px.bar(comp, x="feature", y=["partner_value","train_median"],
                                     barmode="group", title="Partner vs training median (key features)")
                    st.plotly_chart(fig_cmp, use_container_width=True)

                    if prob >= 0.7:
                        st.warning("Suggested action: AM outreach + temporary boost (promo/fee relief) + SLA check.")
                    elif prob >= 0.4:
                        st.info("Suggested action: engagement nudge (visibility slot, reminder), review cancellations.")
                    else:
                        st.success("Suggested action: loyalty nudge / keep warm.")
            except ValueError:
                st.error("Invalid partner_id (must be an integer).")
