import streamlit as st
import pandas as pd
import joblib
from utils import explain_score_model
import numpy as np

# Load model components
weights = joblib.load("model_weights.pkl")
intercept = joblib.load("model_intercept.pkl")
feature_order = joblib.load("X_train_columns.pkl")
scaler = joblib.load("scaler.pkl")

# Updated Mapping dictionaries with unique float values
airway_map = {"Ventilation": 10, "Intubation": 9.25, "Low oxygen": 7.52, "Stable": 0}
renal_map = {"Dialysis": 10, "AKI": 0.67, "Normal": 0}
hematological_map = {"Need for blood products": 10, "Bleeding from the skin or mucosa": 0.53, "Normal": 0}
gi_map = {"Liver failure": 10, "Decreased intestine movement": 0.4991, "Jaundice": 0.2894, "Normal": 0}
co_morb_map = {
    "Systemic Hypertension": 0,
    "Smoking": 0,
    "Chronic Obstructive Pulmonary Disease": 0,
    "Alcohol dependence": 0,
    "Asthma": 0,
    "Diabetes Mellitus": 0.3214,
    "Hypertension, Diabetes Mellitus": 0.3214,
    "Hypertension, Ischemic Heart Disease (IHD)": 0.3214,
    "Hypertension, CKI (Chronic Kidney Insufficiency)": 0.3214,
    "Diabetes Mellitus, Hypertension, Chronic Liver Disease": 0.3214,
    "Diabetes Mellitus, IHD, Hypothyroidism": 0.3214,
    "Diabetes Mellitus, Hypertension, IHD": 0.3214,
    "Hypothyroidism, Hypertension, Diabetes Mellitus, IHD": 0.3214,
    "Diabetes Mellitus, Portal Hypertension": 0.3214,
    "Hypertension, IHD, Diabetes Mellitus": 0.3214,
    "Chronic Liver Disease": 0.3214,
    "Chronic Liver Disease, Hypertension": 0.3214,
    "Chronic Liver Disease, Hypertension, Chronic Pulmonary Disease": 0.3214,
    "Diabetes Mellitus, Systemic Hypertension": 0.3214,
    "Systemic Hypertension, IHD, Diabetes Mellitus": 0.3214,
    "Diabetes Mellitus, Hypertension, Left Lower Lobe Pneumonia": 0.3214,
    "Hypertension, Diabetes Mellitus, IHD, Chronic Kidney Disease (CKD)": 0.3214,
    "CKD": 10,
    "MODS (Multiple Organ Dysfunction Syndrome)": 10,
    "MODS, IHD": 10,
    "CKD, Chronic Liver Disease": 10,
    "CKD, Diabetes Mellitus": 10,
    "Systemic Hypertension, CKD, IHD": 10
}

complications_map = {
    "Myalgia": 0,
    "Reduced Appetite": 0,
    "Abdominal Pain": 0.32154,
    "Right Lower Limb Swelling": 0.32154,
    "Left Lower Limb Cellulitis": 0.32154,
    "Abdominal Distension": 0.32154,
    "Systemic Hypertension": 0.32154,
    "Chest Pain, Dyspnea, Tachypnea": 0.32154,
    "AKI": 10,
    "ARDS": 10,
    "Chronic Liver Disease": 10,
    "Chronic Obstructive Pulmonary Disease": 10,
    "CKD": 10,
    "Chronic Pulmonary Disease": 10,
    "Hepatitis": 10
}

echo_map = {
    "Normal": 0,
    "Adequate LV systolic function": 0,
    "Mild LV systolic dysfunction": 0,
    "Adequate RV function": 0,
    "Mild concentric LVH": 0,
    "Right ventricular dysfunction": 0,
    "Poor Echo Window": 0,

    "Moderate LV Systolic function": 8.57,
    "Moderate LV Dysfunction": 8.57,
    "Bilateral pleural effusion": 8.57,
    "Bilateral Moderate pleural effusion": 8.57,
    "Modified left-sided pleural effusion, Mild LV Systolic function": 8.57,

    "Severe LV Dysfunction": 10,
    "Severe LV systolic dysfunction": 10,
    "Severe LV systolic dysfunction, Mild AR, Severe TR": 10,
    "Massive pericardial effusion": 10,
    "LV dysfunction, Pericardial effusion": 10,
    "LV dysfunction and right ventricular dysfunction": 10,
    "LV dysfunction": 10,
    "Dilated left and right ventricular": 10,
    "Pulmonary hypertension": 10,
    "Multiorgan dysfunction": 10,
    "Reduced RV function": 10,
    "Severe TR": 10,
    "Liver tachycardia": 10,
    "Degenerative aortic valve disease": 10,
    "Concentric LVH": 10
}


# Streamlit page layout
st.set_page_config(page_title="Mortality Risk Predictor", layout="wide")
st.title("🧠 Mortality Risk Prediction for AFI Patients with Thrombocytopenia")
st.markdown("Enter patient details to assess mortality risk and get SHAP-based explanation.")

if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False

# User input form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.number_input("Age", 0, 120, 45)
        Gender = st.selectbox("Sex", ["Male", "Female"])
        Duration_illness = st.number_input("Duration of illness at admission (days)", 0, 60, 3)
        Duration_stay = st.number_input("Duration of hospital stay (days)", 0, 60, 5)
        ICU = st.selectbox("Requirement for ICU at admission", ["No", "Yes"])
        Airway = st.selectbox("Airway & Breathing", list(airway_map.keys()))
        Circulation = st.selectbox("Circulation", ["Stable", "Unstable"])
        Resp = st.selectbox("Respiratory system", ["Normal", "Abnormal"])
        CVS = st.selectbox("Cardiovascular system", ["Normal", "Abnormal"])
        Renal = st.selectbox("Renal", list(renal_map.keys()))
        Hemato = st.selectbox("Hematological", list(hematological_map.keys()))
    with col2:
        GI = st.selectbox("Gastrointestinal and Hepatic", list(gi_map.keys()))
        Neuro = st.selectbox("Nervous System", ["Normal", "Abnormal"])
        Comorb = st.selectbox("Co -Morbidity", list(co_morb_map.keys()))
        Comp = st.selectbox("Complications", list(complications_map.keys()))
        GRBS = st.number_input("GRBS (mg/dL)", 0.0, 500.0, 120.0)
        Total_Protein = st.number_input("Total Protein (g/dl)", 0.0, 10.0, 6.5)
        Albumin = st.number_input("Serum Albumin (g/dl)", 0.0, 5.0, 3.5)
        PT = st.number_input("Prothrombin Time (PT)", 5.0, 50.0, 12.5)
        APTT = st.number_input("APTT", 5.0, 80.0, 30.0)
        Urea = st.number_input("Urea (mg/dL)", 0.0, 200.0, 30.0)
        Sodium = st.number_input("Sodium (mM/l)", 100.0, 180.0, 135.0)
    with col3:
        Potassium = st.number_input("Potassium (mM/l)", 2.0, 6.0, 4.0)
        pH = st.number_input("pH", 6.5, 8.0, 7.4)
        PO2 = st.number_input("PO2 (mmHg)", 0.0, 200.0, 90.0)
        PCO2 = st.number_input("PCO2 (mmHg)", 10.0, 80.0, 40.0)
        Bicarb = st.number_input("Bicarbonate (mmol/l)", 5.0, 40.0, 22.0)
        Lactate = st.number_input("Lactate (mol/L)", 0.0, 15.0, 1.2)
        CPK = st.number_input("CPK (U/L)", 0.0, 10000.0, 150.0)
        CPK_MB = st.number_input("CPK-MB (U/L)", 0.0, 1000.0, 25.0)
        CRP = st.number_input("CRP (mg/L)", 0.0, 400.0, 10.0)
        PCT = st.number_input("Procalcitonin (ng/ml)", 0.0, 100.0, 0.5)
        Ferritin = st.number_input("Serum Ferritin (ng/ml)", 0.0, 5000.0, 200.0)

    # Final section
    LDH = st.number_input("LDH (U/L)", 0.0, 2000.0, 300.0)
    D_Dimer = st.number_input("D -Dimer (mcg/ml)", 0.0, 50.0, 0.8)
    USG = st.selectbox("USG Abdomen", ["Normal", "Abnormal"])
    ECHO = st.selectbox("ECHO", list(echo_map.keys()))
    Hb = st.number_input("Hemoglobin (g/dL)", 0.0, 20.0, 12.0)
    Platelet = st.number_input("Platelet (/mm3)", 0.0, 600000.0, 140000.0)
    WBC = st.number_input("Total Leukocyte Count", 0.0, 30000.0, 8000.0)
    TB = st.number_input("Total Bilirubin", 0.0, 20.0, 1.0)
    DB = st.number_input("Direct Bilirubin", 0.0, 10.0, 0.5)
    AST = st.number_input("AST (U/L)", 0.0, 1000.0, 50.0)
    ALT = st.number_input("ALT (U/L)", 0.0, 1000.0, 50.0)
    Creatinine = st.number_input("Creatinine", 0.0, 20.0, 1.0)

    submitted = st.form_submit_button("🧾 Predict and Explain")
    if submitted:
        st.session_state["form_submitted"] = True
if st.session_state["form_submitted"]:
    view_mode = st.radio(
        "📋 Select Report Type:",
        ["Model View", "Clinical View"],
        index=0,
        key="report_type_toggle"
    )

    # Prepare input dicts
    original_input_text = {
        "Airway & breathing": Airway,
        "Renal": Renal,
        "Hematological": Hemato,
        "Gastrointestinal and Hepatic": GI,
        "CO-MORBIDITY": Comorb,
        "Complications": Comp,
        "ECHO": ECHO,
        "Circulation": Circulation,
        "Respiratory system": Resp,
        "Cardiovascular system": CVS,
        "Nervous system": Neuro,
        "USG abdomen": USG,
        "Requirement for ICU at admission": ICU,
        "Sex": Gender
    }

    input_dict = {
        "Age": Age,
        "Sex": 1 if Gender == "Male" else 0,
        "Duration of illness at the time of admission (days)": Duration_illness,
        "Duration of hospital stay (days)": Duration_stay,
        "Requirement for ICU at admission": 1 if ICU == "Yes" else 0,
        "Airway & breathing": airway_map[Airway],
        "Circulation": 0 if Circulation == "Stable" else 1,
        "Respiratory system": 0 if Resp == "Normal" else 1,
        "Cardiovascular system": 0 if CVS == "Normal" else 1,
        "Renal": renal_map[Renal],
        "Hematological": hematological_map[Hemato],
        "Gastrointestinal and Hepatic": gi_map[GI],
        "Nervous system": 0 if Neuro == "Normal" else 1,
        "CO-MORBIDITY": co_morb_map[Comorb],
        "Complications": complications_map[Comp],
        "GRBS/ random blood sugar (mg/dL)": GRBS,
        "Total protein (g/dl)": Total_Protein,
        "Serum albumin (g/dl)": Albumin,
        "Prothrombin time (PT)": PT,
        "Activated partial thromboplastin time (APTT)": APTT,
        "Urea (mg/dL)": Urea,
        "Sodium (mM/l)": Sodium,
        "Potassium (mM/l)": Potassium,
        "pH": pH,
        "PO2 (mmHg)": PO2,
        "PCO2 (mmHg)": PCO2,
        "Bicarb (mmol/l)": Bicarb,
        "Lactate (mol/L)": Lactate,
        "CPK (U/L)": CPK,
        "CPK-MB (U/L)": CPK_MB,
        "CRP (mg/L)": CRP,
        "Procalcitonin (ng/ml)": PCT,
        "Serum ferritin (ng/ml)": Ferritin,
        "LDH (U/L)": LDH,
        "d Dimer (mcg/ml)": D_Dimer,
        "USG abdomen": 0 if USG == "Normal" else 1,
        "ECHO": echo_map[ECHO],
        "Hemoglobin": Hb,
        "Platelet": Platelet,
        "Total leukocyte count": WBC,
        "Total bilirubin": TB,
        "Direct bilirubin": DB,
        "AST": AST,
        "ALT": ALT,
        "Creatinine": Creatinine
    }

    df_new = pd.DataFrame([input_dict])
    try:
        scaled_columns = scaler.transform(df_new[scaler.feature_names_in_])
        scaled_df = pd.DataFrame(scaled_columns, columns=scaler.feature_names_in_)
        df_combined = pd.concat([scaled_df, df_new.drop(columns=scaler.feature_names_in_)], axis=1)
        df_score_input = df_combined.reindex(columns=feature_order)
        X_input = df_score_input.values.astype(float)
        score = float(np.dot(X_input, weights) + intercept)
        prob = float(1 / (1 + np.exp(-score)))

        if prob >= 0.7:
            outcome = "🟢 Low Risk of Mortality"
        elif prob >= 0.4:
            outcome = "🟡 Moderate Risk of Mortality"
        else:
            outcome = "🔴 High Risk of Mortality"

        st.subheader("🧮 Score-Based Prediction Result")
        st.metric("Predicted Survival Probability", f"{prob * 100:.2f}%")
        st.markdown(f"**Outcome:** {outcome}")

        shap_df = explain_score_model(
            score_weights=weights,
            input_df=df_score_input,
            reference_order=feature_order,
            original_input_df=df_new,
            original_text_map=original_input_text,
            mode=view_mode
        )

    except Exception as ex:
        st.error(f"⚠️ Error during prediction: {ex}")

# Optional: Reset button
if st.session_state["form_submitted"]:
    if st.button("🔁 Reset Form"):
        st.session_state["form_submitted"] = False
        st.experimental_rerun()