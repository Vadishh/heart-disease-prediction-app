import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .main {
        background-color: #f8fafc;
    }

    .title {
        font-size: 42px;
        font-weight: 800;
        color: #dc2626;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #475569;
        text-align: center;
        margin-bottom: 30px;
    }

    .card {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .result-risk {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        border: 2px solid #ef4444;
    }

    .result-safe {
        background-color: #dcfce7;
        color: #166534;
        padding: 25px;
        border-radius: 18px;
        text-align: center;
        font-size: 28px;
        font-weight: 800;
        border: 2px solid #22c55e;
    }

    .small-text {
        color: #64748b;
        font-size: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------
@st.cache_resource
def load_files():
    model = joblib.load("KNN_heart.pkl")       # change name if your model file has different name
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns

try:
    model, scaler, columns = load_files()
except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.stop()

# ---------------- HEADER ----------------
st.markdown('<div class="title">❤️ Heart Disease Prediction App</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter patient health details and get an ML-based prediction</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("About App")
st.sidebar.info(
    "This app uses a trained Machine Learning model to predict whether a person may have heart disease."
)

st.sidebar.warning(
    "This is only a student ML project. It is not medical advice."
)

st.sidebar.markdown("### Model Files Used")
st.sidebar.write("✅ KNN_heart.pkl")
st.sidebar.write("✅ scaler.pkl")
st.sidebar.write("✅ columns.pkl")

# ---------------- INPUT SECTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=1, max_value=120, value=45)

    Sex_label = st.selectbox("Sex", ["Male", "Female"])
    Sex = "M" if Sex_label == "Male" else "F"

    ChestPainType_label = st.selectbox(
        "Chest Pain Type",
        [
            "ASY - Asymptomatic",
            "ATA - Atypical Angina",
            "NAP - Non-Anginal Pain",
            "TA - Typical Angina"
        ]
    )
    ChestPainType = ChestPainType_label.split(" - ")[0]

    RestingBP = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        max_value=250,
        value=120
    )

with col2:
    Cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=200
    )

    FastingBS_label = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl?",
        ["No", "Yes"]
    )
    FastingBS = 1 if FastingBS_label == "Yes" else 0

    RestingECG_label = st.selectbox(
        "Resting ECG",
        [
            "Normal",
            "ST - ST-T Wave Abnormality",
            "LVH - Left Ventricular Hypertrophy"
        ]
    )
    RestingECG = RestingECG_label.split(" - ")[0]

    MaxHR = st.number_input(
        "Maximum Heart Rate",
        min_value=0,
        max_value=250,
        value=150
    )

with col3:
    ExerciseAngina_label = st.selectbox(
        "Exercise Induced Angina?",
        ["No", "Yes"]
    )
    ExerciseAngina = "Y" if ExerciseAngina_label == "Yes" else "N"

    Oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    ST_Slope_label = st.selectbox(
        "ST Slope",
        [
            "Up",
            "Flat",
            "Down"
        ]
    )
    ST_Slope = ST_Slope_label

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- DATA PREPARATION ----------------
input_data = pd.DataFrame({
    "Age": [Age],
    "Sex": [Sex],
    "ChestPainType": [ChestPainType],
    "RestingBP": [RestingBP],
    "Cholesterol": [Cholesterol],
    "FastingBS": [FastingBS],
    "RestingECG": [RestingECG],
    "MaxHR": [MaxHR],
    "ExerciseAngina": [ExerciseAngina],
    "Oldpeak": [Oldpeak],
    "ST_Slope": [ST_Slope]
})

input_encoded = pd.get_dummies(input_data, drop_first=True)

input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

numerical_columns = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

input_encoded[numerical_columns] = scaler.transform(input_encoded[numerical_columns])

# ---------------- PREDICTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Prediction Result")

predict_button = st.button("Predict Heart Disease", use_container_width=True)

if predict_button:
    prediction = model.predict(input_encoded)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_encoded)[0]
        risk_probability = probability[1] * 100
    else:
        risk_probability = None

    if prediction[0] == 1:
        st.markdown(
            '<div class="result-risk">⚠️ Heart Disease Detected</div>',
            unsafe_allow_html=True
        )

        if risk_probability is not None:
            st.metric("Estimated Risk Probability", f"{risk_probability:.2f}%")

        st.warning("The model predicts that the patient may have heart disease.")

    else:
        st.markdown(
            '<div class="result-safe">✅ No Heart Disease Detected</div>',
            unsafe_allow_html=True
        )

        if risk_probability is not None:
            st.metric("Estimated Risk Probability", f"{risk_probability:.2f}%")

        st.success("The model predicts that the patient is less likely to have heart disease.")

    with st.expander("See input data used by model"):
        st.dataframe(input_data)

    with st.expander("See encoded data sent to model"):
        st.dataframe(input_encoded)

else:
    st.info("Fill the details above and click Predict.")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown(
    '<p class="small-text">Made using Python, Streamlit, Scikit-learn and Machine Learning</p>',
    unsafe_allow_html=True
)