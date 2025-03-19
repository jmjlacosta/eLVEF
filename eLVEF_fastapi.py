from typing import Annotated, Literal
from fastapi import FastAPI, Form
import numpy as np

"""
Medicare Claims-Based EF Model
This repository contains an implementation of a logistic regression model that predicts Left Ventricular Ejection Fraction (EF) class in patients with heart failure (HF) using Medicare claims data. The model is based on the paper: "Development and Preliminary Validation of a Medicare Claims–Based Model to Predict Left Ventricular Ejection Fraction Class in Patients With Heart Failure" by Desai et al.

The model aims to differentiate patients into two EF classes: reduced EF (<0.45) and preserved EF (≥0.45), leveraging administrative claims data in scenarios where EF measurements are unavailable.

Input: Patient Data Input: Users manually input key patient data, including demographics, diagnoses, and treatment history.

Prediction: The model processes the inputs and returns the predicted EF class (reduced or preserved).

EF Class Prediction: The model uses logistic regression to classify patients as having either reduced EF or preserved EF.

Model Details
The logistic regression model was trained and validated using data from two academic medical centers:

Training Sample: The model was developed using data from one center, with 35 predictors chosen from 57 potential candidate variables. Testing Sample: The model was validated using data from a second center, with an accuracy of 83%, sensitivity of 0.97 for preserved EF, and a positive predictive value of 0.73 for reduced EF.

Summary
This application provides a practical tool for healthcare researchers and professionals to estimate EF class in heart failure patients using Medicare claims data, helping to bridge the gap when EF measurements are not available. It can be used in studies evaluating health outcomes, healthcare utilization, and cost among heart failure patients.
"""

app = FastAPI(
    title="Estimation of Reduced LVEF",
    description="Predicts the likelihood of Reduced Left Ventricular Ejection Fraction based on diagnosis and medication inputs.",
    version="1.0.0",
)

@app.post("/calculate_probability")
def calculate_probability(
    male: Annotated[Literal["True", "False"], Form(...)] = "False",
    index_dx_out: Annotated[Literal["True", "False"], Form(...)] = "False",
    age: Annotated[int, Form(...)] = 0,
    dx_defibrillator: Annotated[Literal["True", "False"], Form(...)] = "False",
    hosp_chf: Annotated[Literal["True", "False"], Form(...)] = "False",
    rx_ace: Annotated[Literal["True", "False"], Form(...)] = "False",
    rx_antagonist: Annotated[Literal["True", "False"], Form(...)] = "False",
    rx_bblocker: Annotated[Literal["True", "False"], Form(...)] = "False",
    rx_digoxin: Annotated[Literal["True", "False"], Form(...)] = "False",
    rx_loop_diuretic: Annotated[Literal["True", "False"], Form(...)] = "False",
    rx_nitrates: Annotated[Literal["True", "False"], Form(...)] = "False",
    rx_thiazide: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_afib: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_anemia: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_cabg: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_cardiomyopathy: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_copd: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_depression: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_htn_nephropathy: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_hyperlipidemia: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_hypertension: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_hypotension: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_mi: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_obesity: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_oth_dysrhythmia: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_psychosis: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_rheumatic_heart: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_sleep_apnea: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_stable_angina: Annotated[Literal["True", "False"], Form(...)] = "False",
    dx_valve_disorder: Annotated[Literal["True", "False"], Form(...)] = "False",
    hf_type: Annotated[Literal["Systolic", "Diastolic", "Left", "Unspecified"], Form(...)] = "Unspecified",
):
    intercept = -1.37218706653502
    LP0 = intercept

    # Convert "True"/"False" strings to 1/0
    def to_int(value: str) -> int:
        return 1 if value == "True" else 0

    male = to_int(male)
    index_dx_out = to_int(index_dx_out)
    dx_defibrillator = to_int(dx_defibrillator)
    hosp_chf = to_int(hosp_chf)
    rx_ace = to_int(rx_ace)
    rx_antagonist = to_int(rx_antagonist)
    rx_bblocker = to_int(rx_bblocker)
    rx_digoxin = to_int(rx_digoxin)
    rx_loop_diuretic = to_int(rx_loop_diuretic)
    rx_nitrates = to_int(rx_nitrates)
    rx_thiazide = to_int(rx_thiazide)
    dx_afib = to_int(dx_afib)
    dx_anemia = to_int(dx_anemia)
    dx_cabg = to_int(dx_cabg)
    dx_cardiomyopathy = to_int(dx_cardiomyopathy)
    dx_copd = to_int(dx_copd)
    dx_depression = to_int(dx_depression)
    dx_htn_nephropathy = to_int(dx_htn_nephropathy)
    dx_hyperlipidemia = to_int(dx_hyperlipidemia)
    dx_hypertension = to_int(dx_hypertension)
    dx_hypotension = to_int(dx_hypotension)
    dx_mi = to_int(dx_mi)
    dx_obesity = to_int(dx_obesity)
    dx_oth_dysrhythmia = to_int(dx_oth_dysrhythmia)
    dx_psychosis = to_int(dx_psychosis)
    dx_rheumatic_heart = to_int(dx_rheumatic_heart)
    dx_sleep_apnea = to_int(dx_sleep_apnea)
    dx_stable_angina = to_int(dx_stable_angina)
    dx_valve_disorder = to_int(dx_valve_disorder)

    # Linear predictor calculations
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

    # Add coefficient for heart failure type
    hf_type_mapping = {
        "Systolic": 0.754954,
        "Diastolic": -0.950856,
        "Left": 0.766415,
        "Unspecified": -0.577221
    }
    LP0 += hf_type_mapping[hf_type]

    # Logistic function to calculate probability
    probability = 1 / (1 + np.exp(-LP0))

    # Determine classification
    classification = "Reduced Ejection Fraction" if probability > 0.4678 else "Preserved Ejection Fraction"

    return {"probability": round(probability, 4), "classification": classification}
