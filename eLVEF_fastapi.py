from typing import Annotated, Literal
from fastapi import FastAPI, Form, HTTPException
import numpy as np

app = FastAPI(
    title="eLVEF Classification API",
    description="Predicts the likelihood of Reduced Left Ventricular Ejection Fraction based on diagnosis and medication inputs.",
    version="1.0.0",
)

@app.post("/calculate_probability")
def calculate_probability(
    male: Annotated[bool, Form(...)],
    index_dx_out: Annotated[bool, Form(...)],
    age: Annotated[int, Form(..., ge=0, le=120, description="Age of the patient")],
    dx_defibrillator: Annotated[bool, Form(...)],
    hosp_chf: Annotated[bool, Form(...)],
    rx_ace: Annotated[bool, Form(...)],
    rx_antagonist: Annotated[bool, Form(...)],
    rx_bblocker: Annotated[bool, Form(...)],
    rx_digoxin: Annotated[bool, Form(...)],
    rx_loop_diuretic: Annotated[bool, Form(...)],
    rx_nitrates: Annotated[bool, Form(...)],
    rx_thiazide: Annotated[bool, Form(...)],
    dx_afib: Annotated[bool, Form(...)],
    dx_anemia: Annotated[bool, Form(...)],
    dx_cabg: Annotated[bool, Form(...)],
    dx_cardiomyopathy: Annotated[bool, Form(...)],
    dx_copd: Annotated[bool, Form(...)],
    dx_depression: Annotated[bool, Form(...)],
    dx_htn_nephropathy: Annotated[bool, Form(...)],
    dx_hyperlipidemia: Annotated[bool, Form(...)],
    dx_hypertension: Annotated[bool, Form(...)],
    dx_hypotension: Annotated[bool, Form(...)],
    dx_mi: Annotated[bool, Form(...)],
    dx_obesity: Annotated[bool, Form(...)],
    dx_oth_dysrhythmia: Annotated[bool, Form(...)],
    dx_psychosis: Annotated[bool, Form(...)],
    dx_rheumatic_heart: Annotated[bool, Form(...)],
    dx_sleep_apnea: Annotated[bool, Form(...)],
    dx_stable_angina: Annotated[bool, Form(...)],
    dx_valve_disorder: Annotated[bool, Form(...)],
    hf_type: Annotated[Literal["Systolic", "Diastolic", "Left", "Unspecified"], Form(...)],
):
    intercept = -1.37218706653502
    LP0 = intercept

    # Linear predictor calculations based on input values
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

    # Determine classification based on threshold
    classification = "Reduced Ejection Fraction" if probability > 0.4678 else "Preserved Ejection Fraction"

    return {"probability": round(probability, 4), "classification": classification}
