"""
Hospital Length of Stay (LoS) Predictor — Streamlit dashboard
-------------------------------------------------------------
Model : XGBoost regressor (predicts log1p of LoS), NY SPARCS 2021.
Run   : streamlit run app.py

Place the trained model file `XGboost.pkl` in the SAME folder as this file.
The 17 input features and every dropdown option are read straight from the
model, so the app stays in sync with whatever model you ship.
"""

import os
import json
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Hospital LoS Predictor", page_icon="🏥", layout="wide")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "XGboost.pkl")

# Human-readable labels for the three integer-coded features
AGE_LABELS       = {0: "0 to 17", 1: "18 to 29", 2: "30 to 49", 3: "50 to 69", 4: "70 or Older"}
SEVERITY_LABELS  = {0: "0 — Unassigned", 1: "1 — Minor", 2: "2 — Moderate", 3: "3 — Major", 4: "4 — Extreme"}
MORTALITY_LABELS = {0: "Minor", 1: "Moderate", 2: "Major", 3: "Extreme"}

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
# Load model + decode the categorical encoder it carries
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    model = joblib.load(MODEL_PATH)
    booster = model.get_booster()
    names = list(booster.feature_names)
    ftypes = list(booster.feature_types)
    best_it = int(getattr(model, "best_iteration", booster.num_boosted_rounds() - 1))

    # Pull the category lists out of the saved model (XGBoost >= 3.0)
    tmp = tempfile.mktemp(suffix=".json")
    booster.save_model(tmp)
    with open(tmp) as f:
        mj = json.load(f)
    try:
        enc = mj["learner"]["gradient_booster"]["model"]["cats"]["enc"]
    except KeyError as e:
        raise RuntimeError(
            "Could not read category info from the model. This app needs "
            "xgboost>=3.0 to load this .pkl. Please upgrade xgboost."
        ) from e

    def decode(entry):
        if "offsets" in entry:                       # string categorical
            offs, vals = entry["offsets"], entry["values"]
            out = []
            for i in range(len(offs) - 1):
                out.append(bytes(vals[offs[i]:offs[i + 1]]).decode("utf-8"))
            if offs:
                out.append(bytes(vals[offs[-1]:]).decode("utf-8"))
            return out
        return list(entry["values"])                 # numeric categorical

    cats = {names[i]: {"type": ftypes[i], "categories": decode(enc[i])}
            for i in range(len(names))}
    return model, booster, names, cats, best_it


model, booster, FEATURES, CATS, BEST_IT = load_model()


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
    """Show the human-readable value (labels for the coded features)."""
    v = raw[feature]
    if feature == "Age Group":
        return AGE_LABELS.get(v, v)
    if feature == "APR Severity of Illness Code":
        return SEVERITY_LABELS.get(v, v)
    if feature == "APR Risk of Mortality":
        return MORTALITY_LABELS.get(v, v)
    return str(v)


def build_X(raw):
    """One-row dataframe with the exact dtypes the model was trained on."""
    X = pd.DataFrame([raw])[FEATURES]
    for f in FEATURES:
        if CATS[f]["type"] == "int":
            X[f] = X[f].astype("int64")
        else:
            X[f] = pd.Categorical(X[f], categories=CATS[f]["categories"])
    return X


def predict_los(X):
    pred_log = float(model.predict(X)[0])
    return float(np.clip(np.expm1(pred_log), 0, 120))


def shap_contributions(X):
    """Exact TreeSHAP values (log space). Returns (base_log, shap_log array)."""
    dm = xgb.DMatrix(X, enable_categorical=True)
    contribs = booster.predict(dm, pred_contribs=True,
                               iteration_range=(0, BEST_IT + 1))[0]
    return float(contribs[-1]), np.asarray(contribs[:-1], dtype=float)


def shap_figure(raw, shap_log):
    """Horizontal bar chart of every feature's contribution to this prediction."""
    order = np.argsort(np.abs(shap_log))            # ascending -> biggest on top
    vals = shap_log[order]
    labels = [f"{FEATURES[i]}  =  {display_value(FEATURES[i], raw)}" for i in order]
    colors = ["#d9534f" if v > 0 else "#3f8fd0" for v in vals]   # red up, blue down

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


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("About")
    st.write(
        "Predicts a hospital inpatient's **Length of Stay (days)** from an "
        "XGBoost model trained on NY SPARCS 2021 discharge records."
    )
    st.caption(
        "Red bars push the stay **longer**, blue bars push it **shorter**. "
        "Contributions are exact TreeSHAP values in the model's log-output space; "
        "they add up from a baseline to the final prediction."
    )
    st.divider()
    st.caption(f"Model features: {len(FEATURES)}  ·  trees used: {BEST_IT + 1}")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🏥 Hospital Length of Stay Predictor")
st.write("Set the patient details below and click **Predict**. Values are pre-filled with an example case.")

# Pre-computed ordinal choice lists
age_choices  = [AGE_LABELS[k]       for k in sorted(AGE_LABELS)]
sev_choices  = [SEVERITY_LABELS[k]  for k in sorted(SEVERITY_LABELS)]
mort_choices = [MORTALITY_LABELS[k] for k in sorted(MORTALITY_LABELS)]
age_back  = {v: k for k, v in AGE_LABELS.items()}
sev_back  = {v: k for k, v in SEVERITY_LABELS.items()}
mort_back = {v: k for k, v in MORTALITY_LABELS.items()}


# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #
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

    submitted = st.form_submit_button("🔮 Predict Length of Stay", type="primary",
                                      use_container_width=True)


# --------------------------------------------------------------------------- #
# Predict + explain
# --------------------------------------------------------------------------- #
if submitted:
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

    X = build_X(raw)
    pred_days = predict_los(X)
    base_log, shap_log = shap_contributions(X)
    base_days = float(np.expm1(base_log))

    st.divider()
    r1, r2 = st.columns([1, 2])
    with r1:
        st.metric("Predicted Length of Stay", f"{pred_days:.1f} days")
        st.caption(f"Plan for roughly **{int(np.ceil(pred_days))} day(s)**.")
        band = ("Short stay", "🟢") if pred_days < 3 else \
               ("Medium stay", "🟠") if pred_days <= 7 else ("Extended stay", "🔴")
        st.write(f"{band[1]} {band[0]}")
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
