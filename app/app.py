from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from joblib import load

# Create .streamlit/config.toml (light theme) in the folder you launch Streamlit from, if missing.
_cfg = Path.cwd() / ".streamlit" / "config.toml"
if not _cfg.exists():
    try:
        _cfg.parent.mkdir(exist_ok=True)
        _cfg.write_text(
            '[theme]\nbase = "light"\nprimaryColor = "#12A89D"\nbackgroundColor = "#F4F7FB"\n'
            'secondaryBackgroundColor = "#FFFFFF"\ntextColor = "#243B53"\nfont = "sans serif"\n'
        )
        _CONFIG_JUST_CREATED = True
    except OSError:
        _CONFIG_JUST_CREATED = False
else:
    _CONFIG_JUST_CREATED = False

st.set_page_config(
    page_title="ChurnIQ | Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# THEME (navy, teal, soft grey) - forced light so it never clashes with dark mode
# Also add .streamlit/config.toml with base = "light" (see instructions)
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --navy: #102A43; --teal: #12A89D; --teal-light: #E5F7F5;
        --text: #243B53; --muted: #627D98; --border: #D9E2EC; --card: #FFFFFF;
    }

    :root, html, body, .stApp { color-scheme: light !important; }

    /* Tabs: strongest selectors, in case Streamlit's markup differs by version */
    [role="tablist"] button, [role="tablist"] button *, [data-baseweb="tab-list"] button *,
    [data-testid="stTabs"] button[role="tab"] *, button[data-baseweb="tab"], button[data-baseweb="tab"] * {
        color: #0B6F67 !important; -webkit-text-fill-color: #0B6F67 !important;
        opacity: 1 !important; filter: none !important; font-weight: 700 !important;
    }
    [role="tablist"] button[aria-selected="true"], [role="tablist"] button[aria-selected="true"] *,
    [data-testid="stTabs"] button[aria-selected="true"] * {
        color: #0B6F67 !important; -webkit-text-fill-color: #0B6F67 !important;
    }
    [role="tablist"] button[aria-selected="true"] { border-bottom: 3px solid #12A89D !important; }
    [role="tablist"] [data-baseweb="tab-highlight"], [data-baseweb="tab-highlight"],
    [role="tablist"] > div[aria-hidden="true"], [role="tablist"] > div:last-child:not([role]) {
        background-color: #12A89D !important; background: #12A89D !important;
    }
    [role="tablist"] [data-baseweb="tab-border"], [data-baseweb="tab-border"] { background-color: #D9E2EC !important; }

    /* Background */
    html, body, [class*="css"], .stApp { font-family: 'DM Sans', sans-serif; color: var(--text); }
    .stApp {
        background-color: #F4F7FB !important;
        background-image:
            radial-gradient(circle at 8% 0%, rgba(18,168,157,0.18), transparent 40%),
            radial-gradient(circle at 95% 8%, rgba(21,94,117,0.15), transparent 38%),
            radial-gradient(rgba(16,42,67,0.07) 1px, transparent 1px),
            linear-gradient(180deg, #F4F7FB 0%, #E8EFF7 100%) !important;
        background-size: auto, auto, 24px 24px, auto !important;
        background-attachment: fixed !important;
    }
    .block-container { max-width: 1320px; padding-top: 1.7rem; padding-bottom: 3rem; }
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent !important; }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.035em; font-weight: 700; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] > div { padding-top: 1.5rem; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #334E68 !important; opacity: 1 !important; }

    /* Hero and cards */
    .hero {
        background: linear-gradient(120deg, #102A43 0%, #155E75 65%, #12A89D 100%);
        padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(16, 42, 67, 0.13);
    }
    .hero-kicker { color: #A8E7E1; font-size: 0.78rem; font-weight: 700; letter-spacing: 0.08em; margin-bottom: 0.55rem; }
    .hero-title { color: #FFFFFF; font-size: 2.15rem; line-height: 1.15; font-weight: 800; }
    .hero-subtitle { color: #D9F1F0; font-size: 1rem; margin-top: 0.6rem; max-width: 720px; }
    .section-note { color: var(--muted); margin-top: -0.5rem; margin-bottom: 1rem; }
    .mini-label { color: var(--muted); font-size: 0.78rem; font-weight: 700; letter-spacing: 0.05em; }
    .metric-card {
        background: var(--card); border: 1px solid var(--border); border-radius: 15px;
        padding: 1rem 1.15rem; min-height: 105px; box-shadow: 0 3px 12px rgba(16, 42, 67, 0.05);
    }
    .metric-value { color: var(--navy); font-size: 1.55rem; font-weight: 800; margin-top: 0.2rem; }
    .metric-caption { color: var(--muted); font-size: 0.82rem; }
    .result-card {
        background: #FFFFFF; border: 1px solid var(--border); border-top: 5px solid var(--accent);
        border-radius: 18px; padding: 1.5rem; box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08);
    }
    .result-label { color: var(--muted); font-size: 0.9rem; font-weight: 600; }
    .result-number { color: var(--accent); font-size: 3.4rem; line-height: 1.12; font-weight: 800; margin: 0.35rem 0; }
    .result-risk { color: var(--accent); font-weight: 800; font-size: 1.05rem; }
    .result-advice { color: var(--muted); font-size: 0.9rem; margin-top: 1rem; line-height: 1.55; }
    .progress-track { height: 10px; background: #EAF0F6; border-radius: 20px; overflow: hidden; margin-top: 0.9rem; }
    .progress-fill { height: 100%; background: var(--accent); border-radius: 20px; }
    .info-card { background: #FFFFFF; border: 1px solid var(--border); border-radius: 15px; padding: 1.25rem 1.4rem; height: 100%; }
    .info-card h4 { color: var(--navy); margin: 0 0 0.6rem 0; font-size: 1rem; }
    .info-card p, .info-card li { color: var(--muted); font-size: 0.9rem; line-height: 1.6; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0.7rem; border-bottom: 1px solid var(--border); }
    .stTabs [data-baseweb="tab"] { height: 3.2rem; font-weight: 700; background: transparent !important; }
    .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] *,
    .stTabs button[role="tab"] p, .stTabs button[role="tab"] div {
        color: #486581 !important; opacity: 1 !important;
    }
    .stTabs [data-baseweb="tab"]:hover, .stTabs [data-baseweb="tab"]:hover * { color: var(--navy) !important; }
    .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] *,
    .stTabs button[aria-selected="true"] p { color: var(--teal) !important; }
    .stTabs [data-baseweb="tab-highlight"], div[data-baseweb="tab-highlight"] {
        background-color: var(--teal) !important; background: var(--teal) !important; height: 3px !important;
    }
    .stTabs [data-baseweb="tab-border"], div[data-baseweb="tab-border"] { background-color: var(--border) !important; }

    /* Expanders */
    div[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
    div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary { background-color: #FFFFFF !important; }
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * {
        color: var(--navy) !important; fill: var(--navy) !important; font-weight: 700;
    }

    /* Labels */
    [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label { color: var(--text) !important; font-weight: 600 !important; }
    [data-testid="stCaptionContainer"] { color: var(--muted); }

    /* Dropdowns */
    [data-baseweb="select"] div { background-color: #FFFFFF !important; color: var(--text) !important; }
    [data-baseweb="select"] > div { border: 1px solid var(--border) !important; border-radius: 9px !important; }
    [data-baseweb="select"] svg { fill: var(--muted) !important; }
    [data-baseweb="popover"] ul, [data-baseweb="popover"] li,
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] [role="option"] {
        background-color: #FFFFFF !important; color: var(--text) !important;
    }
    [data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] li:hover { background-color: var(--teal-light) !important; }

    /* Number inputs */
    [data-testid="stNumberInput"] input {
        color: var(--navy) !important; -webkit-text-fill-color: var(--navy) !important;
        background-color: #FFFFFF !important; font-weight: 600 !important;
    }
    [data-testid="stNumberInput"] [data-baseweb="input"], [data-testid="stNumberInput"] [data-baseweb="base-input"] {
        background-color: #FFFFFF !important; border: 1px solid var(--border) !important; border-radius: 9px !important;
    }
    [data-testid="stNumberInput"] button { background-color: var(--teal-light) !important; color: var(--navy) !important; border: none !important; }
    [data-testid="stNumberInput"] button:hover { background-color: var(--teal) !important; color: #FFFFFF !important; }
    [data-testid="stNumberInput"] button svg { fill: currentColor !important; }

    /* Slider */
    [data-baseweb="slider"] div[role="slider"] { background-color: var(--teal) !important; }
    [data-testid="stSliderThumbValue"] { color: var(--teal) !important; }

    /* Buttons */
    .stButton > button {
        background: var(--teal); border: 0; border-radius: 10px; min-height: 3rem;
        font-weight: 800; transition: all 0.2s ease;
    }
    .stButton > button, .stButton > button * { color: #FFFFFF !important; }
    .stButton > button:hover { background: #0D887F; border: 0; transform: translateY(-1px); }
    .stButton > button:focus { box-shadow: 0 0 0 3px rgba(18, 168, 157, 0.22); }

    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
    hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODELS (looks in app/models, then project-root/models)
# =========================================================
@st.cache_resource
def load_models():
    app_dir = Path(__file__).resolve().parent
    models_dir = next((d for d in (app_dir / "models", app_dir.parent / "models") if d.is_dir()), None)
    if models_dir is None:
        raise FileNotFoundError("Could not find a 'models' folder beside app.py or one level above it.")
    names = ["logistic_model.pkl", "rf_model.pkl", "xgb_model.pkl"]
    missing = [n for n in names if not (models_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing model file(s) in {models_dir}: {', '.join(missing)}")
    return tuple(load(models_dir / n) for n in names)


try:
    logistic_model, rf_model, xgb_model = load_models()
except Exception as exc:
    st.error(f"Model loading error: {exc}")
    st.stop()

try:
    preprocessor = rf_model.named_steps["preprocessor"]
    rf_classifier = rf_model.named_steps["model"]
except (AttributeError, KeyError):
    st.error("The Random Forest file must be a pipeline with steps named 'preprocessor' and 'model'.")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.2rem 0 1rem 0;">
            <div style="font-size:1.55rem;font-weight:800;color:#102A43;">Churn<span style="color:#12A89D;">IQ</span></div>
            <div style="color:#627D98;font-size:0.82rem;">Customer retention analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### ⚙️ Prediction settings")
    model_choice = st.selectbox("Choose machine learning model", ["Logistic Regression", "Random Forest", "XGBoost"])
    st.markdown("---")
    st.markdown("**Risk classification**")
    st.markdown("🟢 Low: below 30%")
    st.markdown("🟠 Medium: 30% to 60%")
    st.markdown("🔴 High: above 60%")
    st.markdown("---")
    st.caption("Model output is an estimate based on the customer information entered.")


model = {"Logistic Regression": logistic_model, "Random Forest": rf_model, "XGBoost": xgb_model}[model_choice]


# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Customer analytics and predictive intelligence</div>
        <div class="hero-title">Customer Churn Predictor</div>
        <div class="hero-subtitle">Understand retention risk, identify customers who may leave, and support smarter retention decisions.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3 = st.columns(3)
for col, label, value, caption in [
    (m1, "Prediction type", "Churn risk", "Customer-level classification"),
    (m2, "Available models", "3", "Logistic Regression, Random Forest, XGBoost"),
    (m3, "Risk levels", "3", "Low, Medium, High"),
]:
    col.markdown(
        f'<div class="metric-card"><div class="mini-label">{label}</div>'
        f'<div class="metric-value">{value}</div><div class="metric-caption">{caption}</div></div>',
        unsafe_allow_html=True,
    )
st.write("")

prediction_tab, insights_tab = st.tabs(["  🔎  Customer prediction  ", "  📈  Model insights  "])


# =========================================================
# CUSTOMER PREDICTION
# =========================================================
with prediction_tab:
    form_col, result_col = st.columns([1.55, 1], gap="large")
    YNN = ["Yes", "No", "No internet service"]

    with form_col:
        st.subheader("Customer information")
        st.markdown('<div class="section-note">Enter the customer profile, subscribed services and billing details.</div>', unsafe_allow_html=True)

        with st.expander("👤  Customer profile", expanded=True):
            c1, c2 = st.columns(2)
            gender = c1.selectbox("Gender", ["Male", "Female"])
            senior = c2.selectbox("Senior citizen", [0, 1], format_func=lambda v: "Yes" if v else "No")
            partner = c1.selectbox("Partner", ["Yes", "No"])
            dependents = c2.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with st.expander("📡  Services subscribed", expanded=True):
            c1, c2 = st.columns(2)
            phoneservice = c1.selectbox("Phone service", ["Yes", "No"])
            multiplelines = c2.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
            internet = c1.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            onlinesecurity = c2.selectbox("Online security", YNN)
            onlinebackup = c1.selectbox("Online backup", YNN)
            deviceprotection = c2.selectbox("Device protection", YNN)
            techsupport = c1.selectbox("Tech support", YNN)
            streamingtv = c2.selectbox("Streaming TV", YNN)
            streamingmovies = c1.selectbox("Streaming movies", YNN)

        with st.expander("💳  Contract and billing", expanded=True):
            c1, c2 = st.columns(2)
            contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = c2.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            c1, c2 = st.columns(2)
            monthly = c1.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 1.0)
            total = c2.number_input("Total charges ($)", 0.0, 10000.0, 1000.0, 10.0)

    customer_data = pd.DataFrame({
        "gender": [gender], "SeniorCitizen": [senior], "Partner": [partner],
        "Dependents": [dependents], "tenure": [tenure], "PhoneService": [phoneservice],
        "MultipleLines": [multiplelines], "InternetService": [internet],
        "OnlineSecurity": [onlinesecurity], "OnlineBackup": [onlinebackup],
        "DeviceProtection": [deviceprotection], "TechSupport": [techsupport],
        "StreamingTV": [streamingtv], "StreamingMovies": [streamingmovies],
        "Contract": [contract], "PaperlessBilling": [paperless], "PaymentMethod": [payment],
        "MonthlyCharges": [monthly], "TotalCharges": [total],
    })

    with result_col:
        st.subheader("Prediction result")
        st.markdown('<div class="section-note">Your selected model will estimate the probability of churn.</div>', unsafe_allow_html=True)

        predict_clicked = st.button("🚀  Predict customer churn", use_container_width=True, type="primary")

        if predict_clicked:
            try:
                p = float(np.clip(model.predict_proba(customer_data)[0][1], 0, 1))
                if p > 0.60:
                    risk, accent = "High churn risk", "#D64545"
                    advice = "Prioritise this customer for a retention conversation. Review service issues and consider a relevant retention offer."
                elif p > 0.30:
                    risk, accent = "Medium churn risk", "#D99A24"
                    advice = "Monitor this customer and review satisfaction, billing concerns and opportunities for a loyalty incentive."
                else:
                    risk, accent = "Low churn risk", "#239B70"
                    advice = "The model estimates a lower likelihood of churn. Continue regular customer engagement."
                st.session_state["last_prediction"] = {
                    "probability": p, "risk": risk, "accent": accent, "advice": advice,
                    "model_choice": model_choice, "customer_data": customer_data.copy(),
                }
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        pred = st.session_state.get("last_prediction")
        if pred:
            st.markdown(
                f"""
                <div class="result-card" style="--accent:{pred['accent']};">
                    <div class="result-label">Estimated churn probability</div>
                    <div class="result-number">{pred['probability'] * 100:.1f}%</div>
                    <div class="result-risk">{pred['risk']}</div>
                    <div class="progress-track"><div class="progress-fill" style="width:{pred['probability'] * 100:.1f}%;"></div></div>
                    <div class="result-advice">{pred['advice']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Prediction model: {pred['model_choice']}")
        else:
            st.markdown(
                '<div class="info-card"><h4>Ready when you are</h4><p>Complete the customer details and click '
                '<b>Predict customer churn</b>. The result shows the estimated churn probability and a suggested next action.</p></div>',
                unsafe_allow_html=True,
            )

    if predict_clicked and st.session_state.get("last_prediction"):
        st.divider()
        st.subheader("🔍 What influenced this prediction?")
        st.caption("SHAP explanation for the Random Forest model. It is shown even if another model is selected above.")
        try:
            latest = st.session_state["last_prediction"]["customer_data"]
            transformed_df = pd.DataFrame(preprocessor.transform(latest), columns=preprocessor.get_feature_names_out())
            shap_values = shap.Explainer(rf_classifier)(transformed_df)
            fig, ax = plt.subplots(figsize=(10, 5))
            plt.sca(ax)
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        except Exception as exc:
            st.warning(f"Could not create the SHAP explanation: {exc}")


# =========================================================
# MODEL INSIGHTS
# =========================================================
with insights_tab:
    st.subheader("Model performance comparison")
    st.markdown('<div class="section-note">Evaluation metrics recorded for the three trained models.</div>', unsafe_allow_html=True)
    performance_df = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
        "ROC AUC": [0.86, 0.85, 0.85],
        "F1 Score": [0.64, 0.65, 0.63],
        "Precision": [0.52, 0.56, 0.55],
        "Recall": [0.84, 0.78, 0.75],
    })
    st.dataframe(
        performance_df, hide_index=True, use_container_width=True,
        column_config={c: st.column_config.NumberColumn(format="%.2f") for c in ["ROC AUC", "F1 Score", "Precision", "Recall"]},
    )

    st.divider()
    st.subheader("Top churn drivers (Random Forest)")
    feature_names = preprocessor.get_feature_names_out()
    top = pd.DataFrame({"Feature": feature_names, "Importance": rf_classifier.feature_importances_}) \
        .sort_values("Importance", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Feature"], top["Importance"], color="#12A89D")
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title("15 most influential features", loc="left", fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.divider()
    st.subheader("SHAP model explainability")
    st.caption("Generates an illustrative SHAP overview from a synthetic sample based on the form inputs, not your full dataset.")

    if st.button("Generate SHAP overview"):
        try:
            base_row = st.session_state.get("last_prediction", {}).get("customer_data", customer_data)
            sample = pd.concat([base_row] * 50, ignore_index=True)
            rng = np.random.default_rng(42)
            sample["tenure"] = rng.integers(0, 73, size=50)
            sample["MonthlyCharges"] = rng.uniform(20, 120, size=50)
            sample["TotalCharges"] = rng.uniform(100, 5000, size=50)
            sample_df = pd.DataFrame(preprocessor.transform(sample), columns=feature_names)
            values = shap.Explainer(rf_classifier)(sample_df)

            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**SHAP summary**")
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.sca(ax)
                shap.plots.beeswarm(values[:, :, 1], max_display=15, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with right:
                st.markdown("**SHAP feature importance**")
                fig, ax = plt.subplots(figsize=(8, 6))
                plt.sca(ax)
                shap.plots.bar(values[:, :, 1], max_display=15, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except Exception as exc:
            st.error(f"Could not generate SHAP overview: {exc}")

    st.divider()
    st.subheader("Business interpretation")
    left, right = st.columns(2, gap="large")
    left.markdown(
        """
        <div class="info-card"><h4>📌 Key drivers of churn</h4><ul>
            <li><b>Tenure:</b> newer customers are more likely to leave.</li>
            <li><b>Contract type:</b> month-to-month customers have less commitment.</li>
            <li><b>Monthly charges:</b> higher bills increase price sensitivity.</li>
            <li><b>Internet service:</b> fiber optic customers tend to churn more than DSL.</li>
            <li><b>Add-on services:</b> support and security options relate to retention.</li>
        </ul></div>
        """,
        unsafe_allow_html=True,
    )
    right.markdown(
        """
        <div class="info-card"><h4>💡 Recommended business actions</h4><ul>
            <li>Offer long-term contract benefits.</li>
            <li>Review service quality and support for at-risk customers.</li>
            <li>Identify customers who may need a billing or plan review.</li>
            <li>Strengthen onboarding for customers with low tenure.</li>
            <li>Track retention outcomes to evaluate campaigns.</li>
        </ul></div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div style="text-align:center;color:#829AB1;font-size:0.78rem;padding:2rem 0 0.5rem;">'
    "ChurnIQ · Customer Churn Prediction · Built with Python and Streamlit</div>",
    unsafe_allow_html=True,
)