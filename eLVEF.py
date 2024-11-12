import streamlit as st
import numpy as np

# Define the function to calculate the logistic regression prediction
def calculate_probability(male, index_dx_out, age, dx_defibrillator, hosp_chf, 
                          rx_ace, rx_antagonist, rx_bblocker, rx_digoxin, 
                          rx_loop_diuretic, rx_nitrates, rx_thiazide, dx_afib, 
                          dx_anemia, dx_cabg, dx_cardiomyopathy, dx_copd, 
                          dx_depression, dx_htn_nephropathy, dx_hyperlipidemia, 
                          dx_hypertension, dx_hypotension, dx_mi, dx_obesity, 
                          dx_oth_dysrhythmia, dx_psychosis, dx_rheumatic_heart, 
                          dx_sleep_apnea, dx_stable_angina, dx_valve_disorder, hf_type):
    
    # Intercept
    intercept = -1.37218706653502
    LP0 = intercept
    
    # Linear predictor
    LP0 += 0.323651 * male
    LP0 += -0.187191 * index_dx_out
    LP0 += -0.005747 * age
    LP0 += 0.275032 * dx_defibrillator
    LP0 += 0.346289 * hosp_chf
    LP0 += 0.221748 * rx_ace
    LP0 += 0.166008 * rx_antagonist
    LP0 += 0.087257 * rx_bblocker
    LP0 += 0.163224 * rx_digoxin
    LP0 += 0.084251 * rx_loop_diuretic
    LP0 += 0.129225 * rx_nitrates
    LP0 += -0.160819 * rx_thiazide
    LP0 += -0.002267 * dx_afib
    LP0 += -0.165353 * dx_anemia
    LP0 += -0.040175 * dx_cabg
    LP0 += 1.415113 * dx_cardiomyopathy
    LP0 += -0.037023 * dx_copd
    LP0 += -0.033829 * dx_depression
    LP0 += -0.033830 * dx_htn_nephropathy
    LP0 += -0.001805 * dx_hyperlipidemia
    LP0 += -0.098539 * dx_hypertension
    LP0 += -0.017282 * dx_hypotension
    LP0 += 0.651778 * dx_mi
    LP0 += -0.141956 * dx_obesity
    LP0 += 0.116652 * dx_oth_dysrhythmia
    LP0 += -0.068198 * dx_psychosis
    LP0 += -0.073889 * dx_rheumatic_heart
    LP0 += -0.035560 * dx_sleep_apnea
    LP0 += -0.015657 * dx_stable_angina
    LP0 += -0.163684 * dx_valve_disorder

    # Adding the appropriate value for the selected heart failure type
    if hf_type == "Systolic":
        LP0 += 0.754954
    elif hf_type == "Diastolic":
        LP0 += -0.950856
    elif hf_type == "Left":
        LP0 += 0.766415
    elif hf_type == "Unspecified":
        LP0 += -0.577221

    # Logistic function
    probability = 1 / (1 + np.exp(-LP0))
    
    return probability

# Streamlit UI
st.title("Claims–Based Model to Predict Reduced Left Ventricular Ejection Fraction")

# Diagnosis section
st.header("Diagnosis")
col1, col2 = st.columns(2)

# Define tooltips for diagnosis variables
diagnosis_tooltips = {
    "dx_defibrillator": "ICD10-CM: Z45.02, Z95.810, CPT: 33215-33218, 33220, 33223, 33226, 33230-33231, 33240-33241, 33244, 33249, 33262-33264, 93307-93308, 93640-93642, CCSPCS: 7.6.2",
    "hosp_chf": "Count of CHF as primary diagnosis in inpatient stay - 6 months prior to index date",
    "dx_afib": "ICD10-CM: I48 - 6 months prior to index date",
    "dx_anemia": "CCS: 4.1.3 - 6 months prior to index date",
    "dx_cabg": "CCSPCS: 7.2, CPT: 33510-33519, 33521-33523, 33530, 33533-33536, 33545, 33572 - 6 months prior to index date",
    "dx_cardiomyopathy": "ICD10-CM: I42.x - 6 months prior to index date",
    "dx_copd": "ICD10-CM: I43, I44 - 6 months prior to index date",
    "dx_depression": "CCS: 5.6.2 - 6 months prior to index date",
    "dx_valve_disorder": "CCS: 7.2.1, CPT: 33660, 33665, 33400-33401, 33403, 33420-33427, 33430, 33460-33468, 33475, 0257T-0259T, 0262T, PCS10-CM: Various codes - 6 months prior to index date"
}

with col1:
    dx_defibrillator = st.checkbox("Defibrillator Diagnosis", help=diagnosis_tooltips["dx_defibrillator"])
    hosp_chf = st.checkbox("Hospitalized for CHF", help=diagnosis_tooltips["hosp_chf"])
    dx_afib = st.checkbox("Atrial Fibrillation", help=diagnosis_tooltips["dx_afib"])
    dx_anemia = st.checkbox("Anemia", help=diagnosis_tooltips["dx_anemia"])
    dx_cabg = st.checkbox("CABG", help=diagnosis_tooltips["dx_cabg"])
    dx_cardiomyopathy = st.checkbox("Cardiomyopathy", help=diagnosis_tooltips["dx_cardiomyopathy"])
    dx_copd = st.checkbox("COPD", help=diagnosis_tooltips["dx_copd"])
    dx_depression = st.checkbox("Depression", help=diagnosis_tooltips["dx_depression"])
    dx_htn_nephropathy = st.checkbox("HTN Nephropathy", help="ICD10-CM: I12, I13.10, I13.11 - 6 months prior to index date")
    dx_hyperlipidemia = st.checkbox("Hyperlipidemia", help="ICD10-CM: I78 - 6 months prior to index date")

with col2:
    dx_hypertension = st.checkbox("Hypertension", help="CCS: 7.1 - 6 months prior to index date")
    dx_hypotension = st.checkbox("Hypotension", help="ICD10-CM: I95 - 6 months prior to index date")
    dx_mi = st.checkbox("Myocardial Infarction", help="ICD10-CM: I21 - 6 months prior to index date")
    dx_obesity = st.checkbox("Obesity", help="CCS: 3.11.2, CPT: 43842-43843, 43846-43847, HCPCS: G0443, G0447 - 6 months prior to index date")
    dx_oth_dysrhythmia = st.checkbox("Other Dysrhythmia", help="CCS: 7.2.9 - 6 months prior to index date")
    dx_psychosis = st.checkbox("Psychosis", help="CCS: 5.10 - 6 months prior to index date")
    dx_rheumatic_heart = st.checkbox("Rheumatic Heart Disease", help="ICD10-CM: I05.x - I09.89 - 6 months prior to index date")
    dx_sleep_apnea = st.checkbox("Sleep Apnea", help="ICD10-CM: G47.3 - 6 months prior to index date")
    dx_stable_angina = st.checkbox("Stable Angina", help="ICD10-CM: I20 - 6 months prior to index date")
    dx_valve_disorder = st.checkbox("Valve Disorder", help=diagnosis_tooltips["dx_valve_disorder"])

# Prescriptions section
st.header("Prescriptions")
col3, col4 = st.columns(2)

# Define updated tooltips for prescription variables
prescription_tooltips = {
    "rx_ace": "GPI: 3610 - ACE Inhibitors (e.g., Benazepril, Lisinopril) - 6 months prior to index date",
    "rx_antagonist": "GPI: 3625, 3750 - Mineralocorticoid receptor antagonists (e.g., Eplerenone, Spironolactone) - 6 months prior to index date",
    "rx_bblocker": "GPI: 33 - Beta Blockers (e.g., Carvedilol, Metoprolol) - 6 months prior to index date",
    "rx_digoxin": "GPI: 3120 - Digoxin - 6 months prior to index date",
    "rx_loop_diuretic": "GPI: 3720 - Loop Diuretics (e.g., Furosemide) - 6 months prior to index date",
    "rx_nitrates": "GPI: 3210 - Nitrates (e.g., Nitroglycerin) - 6 months prior to index date",
    "rx_thiazide": "GPI: 3760 - Thiazide Diuretics (e.g., Hydrochlorothiazide) - 6 months prior to index date"
}

with col3:
    rx_ace = st.checkbox("ACE Inhibitor", help=prescription_tooltips["rx_ace"])
    rx_antagonist = st.checkbox("Antagonist", help=prescription_tooltips["rx_antagonist"])
    rx_bblocker = st.checkbox("Beta Blocker", help=prescription_tooltips["rx_bblocker"])
    rx_digoxin = st.checkbox("Digoxin", help=prescription_tooltips["rx_digoxin"])

with col4:
    rx_loop_diuretic = st.checkbox("Loop Diuretic", help=prescription_tooltips["rx_loop_diuretic"])
    rx_nitrates = st.checkbox("Nitrates", help=prescription_tooltips["rx_nitrates"])
    rx_thiazide = st.checkbox("Thiazide", help=prescription_tooltips["rx_thiazide"])

# Other parameters section
st.header("Other Parameters")
col5, col6 = st.columns(2)

with col5:
    age = st.number_input("Age", min_value=0, max_value=120, value=77, help="Enter the patient's age.")
    male = st.checkbox("Male", help="Binary (1: Male, 0: Female) - On index date")
    index_dx_out = st.checkbox("Recent HF-related Outpatient Visit", help="Most Recent HF diagnosis was in an outpatient setting")

with col6:
    hf_type = st.radio("Select Heart Failure Type", 
                       ("Systolic", "Diastolic", "Left", "Unspecified"), help="Select the type of heart failure.")

# Calculate and display the logistic regression probability
if st.button("Calculate Probability"):
    probability = calculate_probability(male, index_dx_out, age, dx_defibrillator, hosp_chf, 
                                        rx_ace, rx_antagonist, rx_bblocker, rx_digoxin, 
                                        rx_loop_diuretic, rx_nitrates, rx_thiazide, 
                                        dx_afib, dx_anemia, dx_cabg, dx_cardiomyopathy, 
                                        dx_copd, dx_depression, dx_htn_nephropathy, 
                                        dx_hyperlipidemia, dx_hypertension, dx_hypotension, 
                                        dx_mi, dx_obesity, dx_oth_dysrhythmia, dx_psychosis, 
                                        dx_rheumatic_heart, dx_sleep_apnea, dx_stable_angina, 
                                        dx_valve_disorder, hf_type)
    if probability > 0.4678:
        st.write(f"eLVEF Classification: Reduced Ejection Fraction")
    else:
        st.write(f"eLVEF Classification: Preserved Ejection Fraction")
