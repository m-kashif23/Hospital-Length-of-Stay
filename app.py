"""
Hospital Length of Stay (LoS) Predictor — Streamlit dashboard
-------------------------------------------------------------
Two models, NY SPARCS 2021:
  • XGBoost regressor      (native categorical features)
  • Random Forest regressor (one-hot encoded features)

Both predict log1p(LoS); the app converts back to days.

Run:  streamlit run app.py

Both trained models are downloaded from one Hugging Face model repo at runtime
(they are too big for GitHub). Only app.py + requirements.txt need to live in
your GitHub repo.  >>> SET HF_REPO_ID below to your NEW repo. <<<
"""

import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import joblib
from huggingface_hub import hf_hub_download

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Hospital LoS Predictor", page_icon="🏥", layout="wide")

# >>> EDIT THIS to your NEW Hugging Face repo that holds BOTH .pkl files <<<
HF_REPO_ID   = "Mkashif23/los-models"
XGB_FILENAME = "XGboost.pkl"
RF_FILENAME  = "random_forest.pkl"

# Human-readable labels for the integer-coded features
AGE_LABELS       = {0: "0 to 17", 1: "18 to 29", 2: "30 to 49", 3: "50 to 69", 4: "70 or Older"}
SEVERITY_LABELS  = {0: "0 — Unassigned", 1: "1 — Minor", 2: "2 — Moderate", 3: "3 — Major", 4: "4 — Extreme"}
MORTALITY_LABELS = {0: "Minor", 1: "Moderate", 2: "Major", 3: "Extreme"}

# How the Random Forest was encoded in the notebook (must match training)
RF_NOMINAL_COLS = [
    "Hospital Service Area", "Hospital County", "Facility Name",
    "Gender", "Race", "Ethnicity", "Type of Admission",
    "CCSR Diagnosis Description", "CCSR Procedure Description",
    "APR DRG Description", "APR MDC Description", "APR Medical Surgical Description",
    "Emergency Department Indicator",
]
RF_NUMERIC_COLS = [
    "Age Group", "APR Severity of Illness Code", "APR Risk of Mortality", "Coverage_Count",
]

# A ready-to-go example patient (70+ emergency heart-failure admission)
DEFAULTS = {
    "Hospital Service Area": "New York City",
    "Hospital County": "Manhattan",
    "Facility Name": "Mount Sinai Hospital",
    "Age Group": 4,
    "Gender": "M",
    "Race": "White",
    "Ethnicity": "Not Span/Hispanic",
    "Type of Admission": "Emergency",
    "CCSR Diagnosis Description": "Heart failure",
    "CCSR Procedure Description": "No Procedure",
    "APR DRG Description": "HEART FAILURE",
    "APR MDC Description": "DISEASES AND DISORDERS OF THE CIRCULATORY SYSTEM",
    "APR Severity of Illness Code": 3,
    "APR Risk of Mortality": 2,
    "APR Medical Surgical Description": "Medical",
    "Emergency Department Indicator": "Y",
    "Coverage_Count": 2,
}


# --------------------------------------------------------------------------- #
# Load BOTH models (from one Hugging Face repo)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Downloading & loading models…")
def load_models():
    # ---- XGBoost (also supplies the dropdown vocabulary for both models) ----
    xgb_path = hf_hub_download(repo_id=HF_REPO_ID, filename=XGB_FILENAME)
    xgb_model = joblib.load(xgb_path)

    booster = xgb_model.get_booster()
    names = list(booster.feature_names)
    ftypes = list(booster.feature_types)
    best_it = int(getattr(xgb_model, "best_iteration", booster.num_boosted_rounds() - 1))

    # Pull category lists out of the saved model (XGBoost >= 3.0)
    tmp = tempfile.mktemp(suffix=".json")
    booster.save_model(tmp)
    with open(tmp) as f:
        mj = json.load(f)
    try:
        enc = mj["learner"]["gradient_booster"]["model"]["cats"]["enc"]
    except KeyError as e:
        raise RuntimeError(
            "Could not read category info from the XGBoost model. This app needs "
            "xgboost>=3.0 to load this .pkl. Please upgrade xgboost."
        ) from e

    def decode(entry):
        if "offsets" in entry:                        # string categorical
            offs, vals = entry["offsets"], entry["values"]
            out = []
            for i in range(len(offs) - 1):
                out.append(bytes(vals[offs[i]:offs[i + 1]]).decode("utf-8"))
            if offs:
                out.append(bytes(vals[offs[-1]:]).decode("utf-8"))
            return out
        return list(entry["values"])                  # numeric categorical

    cats = {names[i]: {"type": ftypes[i], "categories": decode(enc[i])}
            for i in range(len(names))}

    # ---- Random Forest (one-hot feature names live on the model itself) ----
    rf_path = hf_hub_download(repo_id=HF_REPO_ID, filename=RF_FILENAME)
    rf_model = joblib.load(rf_path)
    rf_features = list(getattr(rf_model, "feature_names_in_", []))

    return {
        "xgb": xgb_model, "booster": booster, "best_it": best_it,
        "features": names, "cats": cats,
        "rf": rf_model, "rf_features": rf_features,
    }


M = load_models()
FEATURES = M["features"]      # XGBoost feature order
CATS     = M["cats"]          # XGBoost category vocabulary (drives the dropdowns)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def opts(feature):
    """Selectable string options (drop the blank/NaN placeholder category)."""
    return [c for c in CATS[feature]["categories"] if c != ""]


def idx(feature, choices):
    """Index of the default value within a choice list (0 if missing)."""
    d = DEFAULTS.get(feature)
    return choices.index(d) if d in choices else 0


def display_value(feature, raw):
    """Human-readable value (labels for the coded features)."""
    v = raw[feature]
    if feature == "Age Group":
        return AGE_LABELS.get(v, v)
    if feature == "APR Severity of Illness Code":
        return SEVERITY_LABELS.get(v, v)
    if feature == "APR Risk of Mortality":
        return MORTALITY_LABELS.get(v, v)
    return str(v)


def rf_source_feature(col):
    """Map a one-hot column back to its original feature (for grouping importances)."""
    if col in RF_NUMERIC_COLS:
        return col
    for nc in RF_NOMINAL_COLS:
        if col.startswith(nc + "_"):
            return nc
    return col


# ---- XGBoost feature row (native categoricals) ----
def build_X_xgb(raw):
    X = pd.DataFrame([raw])[FEATURES]
    for f in FEATURES:
        if CATS[f]["type"] == "int":
            X[f] = X[f].astype("int64")
        else:
            X[f] = pd.Categorical(X[f], categories=CATS[f]["categories"])
    return X


# ---- Random Forest feature row (one-hot, rebuilt to the trained columns) ----
def build_X_rf(raw):
    rf_features = M["rf_features"]
    row = {f: 0 for f in rf_features}
    # numeric / ordinal passthrough columns
    for col in RF_NUMERIC_COLS:
        if col in row:
            row[col] = raw[col]
    # one-hot nominal columns (rare values fall back to the "_Other" bucket)
    for col in RF_NOMINAL_COLS:
        dummy = f"{col}_{raw[col]}"
        if dummy in row:
            row[dummy] = 1
        else:
            other = f"{col}_Other"
            if other in row:
                row[other] = 1
    return pd.DataFrame([row], columns=rf_features)


def predict_los(model_key, raw):
    """Return (predicted_days, feature_frame)."""
    if model_key == "xgb":
        X = build_X_xgb(raw)
        dm = xgb.DMatrix(X, enable_categorical=True)
        pred_log = float(M["booster"].predict(dm, iteration_range=(0, M["best_it"] + 1))[0])
    else:
        X = build_X_rf(raw)
        pred_log = float(M["rf"].predict(X)[0])
    return float(np.clip(np.expm1(pred_log), 0, 120)), X


# ---- XGBoost per-prediction explanation (exact TreeSHAP) ----
def shap_contributions(X):
    dm = xgb.DMatrix(X, enable_categorical=True)
    contribs = M["booster"].predict(dm, pred_contribs=True,
                                    iteration_range=(0, M["best_it"] + 1))[0]
    return float(contribs[-1]), np.asarray(contribs[:-1], dtype=float)


def shap_figure(raw, shap_log):
    order = np.argsort(np.abs(shap_log))
    vals = shap_log[order]
    labels = [f"{FEATURES[i]}  =  {display_value(FEATURES[i], raw)}" for i in order]
    colors = ["#d9534f" if v > 0 else "#3f8fd0" for v in vals]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("SHAP value  —  impact on model output (log scale of length of stay)")
    ax.set_title("Why this prediction?  Feature contributions (TreeSHAP)", fontsize=12, pad=10)
    for y, v in enumerate(vals):
        ax.text(v + (0.004 if v >= 0 else -0.004), y, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=7, color="#222")
    pad = max(0.02, np.abs(vals).max() * 0.25)
    ax.set_xlim(vals.min() - pad, vals.max() + pad)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---- Random Forest explanation (model-level feature importance) ----
def rf_importance_figure():
    rf_features = M["rf_features"]
    imp = pd.Series(M["rf"].feature_importances_, index=rf_features)
    grouped = imp.groupby([rf_source_feature(c) for c in rf_features]).sum()
    grouped = grouped.sort_values().tail(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2ecc71" if v > 0.1 else "#3498db" for v in grouped.values]
    ax.barh(range(len(grouped)), grouped.values, color=colors)
    ax.set_yticks(range(len(grouped)))
    ax.set_yticklabels(grouped.index, fontsize=9)
    for y, v in enumerate(grouped.values):
        ax.text(v, y, f" {v:.3f}", va="center", ha="left", fontsize=8, color="#222")
    ax.set_xlim(0, grouped.max() * 1.15)
    ax.set_xlabel("Importance (Gini, summed over one-hot columns)")
    ax.set_title("Random Forest — overall feature importance", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Sidebar — navigation
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🏥 LoS Predictor")
    page = st.radio("Navigate", ["🏠 Home", "🔮 Predictor"], label_visibility="collapsed")
    st.divider()
    st.caption(
        f"XGBoost features: {len(FEATURES)}  ·  trees: {M['best_it'] + 1}\n\n"
        f"Random Forest inputs: {len(M['rf_features'])} (one-hot)  ·  "
        f"trees: {getattr(M['rf'], 'n_estimators', '?')}"
    )


# --------------------------------------------------------------------------- #
# HOME PAGE
# --------------------------------------------------------------------------- #
def render_home():
    st.title("🏥 Hospital Length of Stay Predictor")
    st.subheader("Estimate inpatient Length of Stay from a single admission record")
    st.write(
        "This tool predicts how many days a hospital inpatient is likely to stay, using models "
        "trained on **New York SPARCS 2021** discharge records (~2.1 million admissions). "
        "Pick a model, fill in the patient details, and get a predicted Length of Stay along with "
        "an explanation of what drove the estimate."
    )

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ⚡ XGBoost")
        st.write(
            "Gradient-boosted trees with **native categorical** features. Generally the stronger, "
            "more accurate model. Comes with **per-prediction SHAP** explanations that show exactly "
            "how each field pushed this specific estimate up or down."
        )
    with c2:
        st.markdown("#### 🌲 Random Forest")
        st.write(
            "An ensemble of decision trees on **one-hot encoded** features. A robust, easy-to-interpret "
            "baseline for comparison. Explained through **overall feature importance** "
            "(which fields matter most across all patients)."
        )

    st.divider()
    st.markdown("#### How to use")
    st.markdown(
        "1. Open **🔮 Predictor** from the sidebar.\n"
        "2. Choose **XGBoost** or **Random Forest** at the top.\n"
        "3. Set the patient's demographics, admission, severity and clinical coding.\n"
        "4. Click **Predict** to get the estimated stay and the explanation."
    )

    st.info(
        "Predictions are statistical estimates for planning and education only — they are **not** "
        "medical advice and should not drive individual clinical decisions.",
        icon="ℹ️",
    )


# --------------------------------------------------------------------------- #
# PREDICTOR PAGE
# --------------------------------------------------------------------------- #
def render_predictor():
    st.title("🔮 Predict Length of Stay")

    model_choice = st.radio(
        "Model", ["⚡ XGBoost", "🌲 Random Forest"], horizontal=True,
        help="XGBoost is usually more accurate and gives per-case SHAP; Random Forest is a one-hot baseline.",
    )
    model_key = "xgb" if "XGBoost" in model_choice else "rf"
    st.caption("Values are pre-filled with an example case. Adjust any field, then click **Predict**.")

    # Pre-computed ordinal choice lists
    age_choices  = [AGE_LABELS[k]       for k in sorted(AGE_LABELS)]
    sev_choices  = [SEVERITY_LABELS[k]  for k in sorted(SEVERITY_LABELS)]
    mort_choices = [MORTALITY_LABELS[k] for k in sorted(MORTALITY_LABELS)]
    age_back  = {v: k for k, v in AGE_LABELS.items()}
    sev_back  = {v: k for k, v in SEVERITY_LABELS.items()}
    mort_back = {v: k for k, v in MORTALITY_LABELS.items()}

    with st.form("los_form"):
        st.subheader("Demographics")
        c1, c2, c3, c4 = st.columns(4)
        age       = c1.selectbox("Age Group", age_choices,
                                 index=age_choices.index(AGE_LABELS[DEFAULTS["Age Group"]]))
        gender    = c2.selectbox("Gender", opts("Gender"), index=idx("Gender", opts("Gender")))
        race      = c3.selectbox("Race", opts("Race"), index=idx("Race", opts("Race")))
        ethnicity = c4.selectbox("Ethnicity", opts("Ethnicity"), index=idx("Ethnicity", opts("Ethnicity")))

        st.subheader("Admission")
        a1, a2, a3, a4 = st.columns(4)
        adm     = a1.selectbox("Type of Admission", opts("Type of Admission"),
                               index=idx("Type of Admission", opts("Type of Admission")))
        ed      = a2.selectbox("Emergency Dept. Indicator", opts("Emergency Department Indicator"),
                               index=idx("Emergency Department Indicator", opts("Emergency Department Indicator")))
        cov     = a3.number_input("Insurance Coverage Count", min_value=0, max_value=3,
                                  value=int(DEFAULTS["Coverage_Count"]), step=1,
                                  help="Number of payment typologies that are not Self-Pay (0–3).")
        medsurg = a4.selectbox("Medical / Surgical", opts("APR Medical Surgical Description"),
                               index=idx("APR Medical Surgical Description", opts("APR Medical Surgical Description")))

        st.subheader("Severity")
        s1, s2 = st.columns(2)
        sev  = s1.selectbox("APR Severity of Illness", sev_choices,
                            index=sev_choices.index(SEVERITY_LABELS[DEFAULTS["APR Severity of Illness Code"]]))
        mort = s2.selectbox("APR Risk of Mortality", mort_choices,
                            index=mort_choices.index(MORTALITY_LABELS[DEFAULTS["APR Risk of Mortality"]]))

        st.subheader("Clinical coding")
        cl1, cl2 = st.columns(2)
        mdc      = cl1.selectbox("APR MDC (major diagnostic category)", opts("APR MDC Description"),
                                 index=idx("APR MDC Description", opts("APR MDC Description")))
        drg      = cl1.selectbox("APR DRG", opts("APR DRG Description"),
                                 index=idx("APR DRG Description", opts("APR DRG Description")))
        ccsr_dx  = cl2.selectbox("CCSR Diagnosis", opts("CCSR Diagnosis Description"),
                                 index=idx("CCSR Diagnosis Description", opts("CCSR Diagnosis Description")))
        ccsr_pr  = cl2.selectbox("CCSR Procedure", opts("CCSR Procedure Description"),
                                 index=idx("CCSR Procedure Description", opts("CCSR Procedure Description")))

        st.subheader("Hospital / location")
        h1, h2, h3 = st.columns(3)
        area     = h1.selectbox("Hospital Service Area", opts("Hospital Service Area"),
                                index=idx("Hospital Service Area", opts("Hospital Service Area")))
        county   = h2.selectbox("Hospital County", opts("Hospital County"),
                                index=idx("Hospital County", opts("Hospital County")))
        facility = h3.selectbox("Facility Name", opts("Facility Name"),
                                index=idx("Facility Name", opts("Facility Name")))

        submitted = st.form_submit_button(f"🔮 Predict with {model_choice}",
                                          type="primary", use_container_width=True)

    if not submitted:
        return

    raw = {
        "Hospital Service Area": area,
        "Hospital County": county,
        "Facility Name": facility,
        "Age Group": age_back[age],
        "Gender": gender,
        "Race": race,
        "Ethnicity": ethnicity,
        "Type of Admission": adm,
        "CCSR Diagnosis Description": ccsr_dx,
        "CCSR Procedure Description": ccsr_pr,
        "APR DRG Description": drg,
        "APR MDC Description": mdc,
        "APR Severity of Illness Code": sev_back[sev],
        "APR Risk of Mortality": mort_back[mort],
        "APR Medical Surgical Description": medsurg,
        "Emergency Department Indicator": ed,
        "Coverage_Count": int(cov),
    }

    pred_days, X = predict_los(model_key, raw)

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric(f"Predicted Length of Stay · {model_choice}", f"{pred_days:.1f} days")
        st.caption(f"Plan for roughly **{int(np.ceil(pred_days))} day(s)**.")
        band = ("Short stay", "🟢") if pred_days < 3 else \
               ("Medium stay", "🟠") if pred_days <= 7 else ("Extended stay", "🔴")
        st.write(f"{band[1]} {band[0]}")

    # ---- Model-specific explanation ----
    if model_key == "xgb":
        base_log, shap_log = shap_contributions(X)
        base_days = float(np.expm1(base_log))
        with r2:
            top = int(np.argmax(np.abs(shap_log)))
            direction = "increased" if shap_log[top] > 0 else "reduced"
            st.markdown(
                f"The average expected stay across patients is **{base_days:.1f} days**. "
                f"For this case the model predicts **{pred_days:.1f} days**. "
                f"The single biggest driver is **{FEATURES[top]}** "
                f"(*{display_value(FEATURES[top], raw)}*), which {direction} the estimate."
            )
        st.subheader("Feature contributions for this prediction (SHAP)")
        st.pyplot(shap_figure(raw, shap_log))

        with st.expander("See exact SHAP values"):
            tbl = pd.DataFrame({
                "Feature": FEATURES,
                "Value": [display_value(f, raw) for f in FEATURES],
                "SHAP (log)": shap_log.round(4),
                "Direction": ["↑ longer" if v > 0 else "↓ shorter" for v in shap_log],
            })
            tbl["abs"] = tbl["SHAP (log)"].abs()
            tbl = tbl.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
            st.dataframe(tbl, use_container_width=True, hide_index=True)
    else:
        with r2:
            st.markdown(
                f"The Random Forest predicts **{pred_days:.1f} days** for this admission. "
                "Random Forest does not provide exact per-case contributions, so the chart below shows "
                "the model's **overall feature importance** — which fields matter most across all patients."
            )
            st.caption(
                "Tip: switch to **XGBoost** above for a per-case SHAP breakdown of this exact prediction."
            )
        st.subheader("Overall feature importance (Random Forest)")
        st.pyplot(rf_importance_figure())


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #
if page.startswith("🏠"):
    render_home()
else:
    render_predictor()
