from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(
    title="eLVEF Classification API",
    description="Predicts the likelihood of Reduced Left Ventricular Ejection Fraction based on diagnosis and medication inputs.",
    version="1.0.0",
)

class ProbabilityInput(BaseModel):
    male: Optional[bool] 
    age: int = Field(..., ge=0, le=120, description="Age of the patient")
    dx_defibrillator: Optional[bool]
    hosp_chf: Optional[bool]
    rx_ace: Optional[bool]
    rx_antagonist: Optional[bool]
    rx_bblocker: Optional[bool]
    rx_digoxin: Optional[bool]
    rx_loop_diuretic: Optional[bool]
    rx_nitrates: Optional[bool]
    rx_thiazide: Optional[bool]
    dx_afib: Optional[bool]
    dx_anemia: Optional[bool]
    dx_cabg: Optional[bool]
    dx_cardiomyopathy: Optional[bool]
    dx_copd: Optional[bool]
    dx_depression: Optional[bool]
    dx_htn_nephropathy: Optional[bool]
    dx_hyperlipidemia: Optional[bool]
    dx_hypertension: Optional[bool]
    dx_hypotension: Optional[bool]
    dx_mi: Optional[bool]
    dx_obesity: Optional[bool]
    dx_oth_dysrhythmia: Optional[bool]
    dx_psychosis: Optional[bool]
    dx_rheumatic_heart: Optional[bool]
    dx_sleep_apnea: Optional[bool]
    dx_stable_angina: Optional[bool]
    dx_valve_disorder: Optional[bool]
    hf_type: Optional[Literal["Systolic", "Diastolic", "Left", "Unspecified"]]

class ProbabilityOutput(BaseModel):
    probability: float
    classification: str

@app.post("/calculate_probability", response_model=ProbabilityOutput)
def calculate_probability(data: ProbabilityInput) -> ProbabilityOutput:
    intercept = -1.37218706653502
    LP0 = intercept

    # Linear predictor calculations based on input values
    LP0 += 0.323651 * data.male
    LP0 += -0.187191
    LP0 += -0.005747 * data.age
    LP0 += 0.275032 * data.dx_defibrillator
    LP0 += 0.346289 * data.hosp_chf
    LP0 += 0.221748 * data.rx_ace
    LP0 += 0.166008 * data.rx_antagonist
    LP0 += 0.087257 * data.rx_bblocker
    LP0 += 0.163224 * data.rx_digoxin
    LP0 += 0.084251 * data.rx_loop_diuretic
    LP0 += 0.129225 * data.rx_nitrates
    LP0 += -0.160819 * data.rx_thiazide
    LP0 += -0.002267 * data.dx_afib
    LP0 += -0.165353 * data.dx_anemia
    LP0 += -0.040175 * data.dx_cabg
    LP0 += 1.415113 * data.dx_cardiomyopathy
    LP0 += -0.037023 * data.dx_copd
    LP0 += -0.033829 * data.dx_depression
    LP0 += -0.033830 * data.dx_htn_nephropathy
    LP0 += -0.001805 * data.dx_hyperlipidemia
    LP0 += -0.098539 * data.dx_hypertension
    LP0 += -0.017282 * data.dx_hypotension
    LP0 += 0.651778 * data.dx_mi
    LP0 += -0.141956 * data.dx_obesity
    LP0 += 0.116652 * data.dx_oth_dysrhythmia
    LP0 += -0.068198 * data.dx_psychosis
    LP0 += -0.073889 * data.dx_rheumatic_heart
    LP0 += -0.035560 * data.dx_sleep_apnea
    LP0 += -0.015657 * data.dx_stable_angina
    LP0 += -0.163684 * data.dx_valve_disorder

    # Add coefficient for heart failure type
    hf_type_mapping = {
        "Systolic": 0.754954,
        "Diastolic": -0.950856,
        "Left": 0.766415,
        "Unspecified": -0.577221
    }
    LP0 += hf_type_mapping[data.hf_type]

    # Logistic function to calculate probability
    probability = 1 / (1 + np.exp(-LP0))

    # Determine classification based on threshold
    classification = "Reduced Ejection Fraction" if probability > 0.4678 else "Preserved Ejection Fraction"

    return ProbabilityOutput(probability=round(probability, 4), classification=classification)
