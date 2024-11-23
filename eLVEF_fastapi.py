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
    male: Optional[bool] = None
    age: Optional[int] = Field(None, ge=0, le=120, description="Age of the patient")
    dx_defibrillator: Optional[bool] = None
    hosp_chf: Optional[bool] = None
    rx_ace: Optional[bool] = None
    rx_antagonist: Optional[bool] = None
    rx_bblocker: Optional[bool] = None
    rx_digoxin: Optional[bool] = None
    rx_loop_diuretic: Optional[bool] = None
    rx_nitrates: Optional[bool] = None
    rx_thiazide: Optional[bool] = None
    dx_afib: Optional[bool] = None
    dx_anemia: Optional[bool] = None
    dx_cabg: Optional[bool] = None
    dx_cardiomyopathy: Optional[bool] = None
    dx_copd: Optional[bool] = None
    dx_depression: Optional[bool] = None
    dx_htn_nephropathy: Optional[bool] = None
    dx_hyperlipidemia: Optional[bool] = None
    dx_hypertension: Optional[bool] = None
    dx_hypotension: Optional[bool] = None
    dx_mi: Optional[bool] = None
    dx_obesity: Optional[bool] = None
    dx_oth_dysrhythmia: Optional[bool] = None
    dx_psychosis: Optional[bool] = None
    dx_rheumatic_heart: Optional[bool] = None
    dx_sleep_apnea: Optional[bool] = None
    dx_stable_angina: Optional[bool] = None
    dx_valve_disorder: Optional[bool] = None
    hf_type: Optional[Literal["Systolic", "Diastolic", "Left", "Unspecified"]] = None

class ProbabilityOutput(BaseModel):
    probability: float
    classification: str

@app.post("/calculate_probability", response_model=ProbabilityOutput)
def calculate_probability(data: ProbabilityInput) -> ProbabilityOutput:
    intercept = -1.37218706653502
    LP0 = intercept

    # Handle None values by defaulting to 0 in calculations
    LP0 += 0.323651 * (data.male or 0)
    LP0 += -0.187191
    LP0 += -0.005747 * (data.age or 0)
    LP0 += 0.275032 * (data.dx_defibrillator or 0)
    LP0 += 0.346289 * (data.hosp_chf or 0)
    LP0 += 0.221748 * (data.rx_ace or 0)
    LP0 += 0.166008 * (data.rx_antagonist or 0)
    LP0 += 0.087257 * (data.rx_bblocker or 0)
    LP0 += 0.163224 * (data.rx_digoxin or 0)
    LP0 += 0.084251 * (data.rx_loop_diuretic or 0)
    LP0 += 0.129225 * (data.rx_nitrates or 0)
    LP0 += -0.160819 * (data.rx_thiazide or 0)
    LP0 += -0.002267 * (data.dx_afib or 0)
    LP0 += -0.165353 * (data.dx_anemia or 0)
    LP0 += -0.040175 * (data.dx_cabg or 0)
    LP0 += 1.415113 * (data.dx_cardiomyopathy or 0)
    LP0 += -0.037023 * (data.dx_copd or 0)
    LP0 += -0.033829 * (data.dx_depression or 0)
    LP0 += -0.033830 * (data.dx_htn_nephropathy or 0)
    LP0 += -0.001805 * (data.dx_hyperlipidemia or 0)
    LP0 += -0.098539 * (data.dx_hypertension or 0)
    LP0 += -0.017282 * (data.dx_hypotension or 0)
    LP0 += 0.651778 * (data.dx_mi or 0)
    LP0 += -0.141956 * (data.dx_obesity or 0)
    LP0 += 0.116652 * (data.dx_oth_dysrhythmia or 0)
    LP0 += -0.068198 * (data.dx_psychosis or 0)
    LP0 += -0.073889 * (data.dx_rheumatic_heart or 0)
    LP0 += -0.035560 * (data.dx_sleep_apnea or 0)
    LP0 += -0.015657 * (data.dx_stable_angina or 0)
    LP0 += -0.163684 * (data.dx_valve_disorder or 0)

    # Add coefficient for heart failure type
    hf_type_mapping = {
        "Systolic": 0.754954,
        "Diastolic": -0.950856,
        "Left": 0.766415,
        "Unspecified": -0.577221
    }
    LP0 += hf_type_mapping.get(data.hf_type, 0)

    # Logistic function to calculate probability
    probability = 1 / (1 + np.exp(-LP0))

    # Determine classification based on threshold
    classification = "Reduced Ejection Fraction" if probability > 0.4678 else "Preserved Ejection Fraction"

    return ProbabilityOutput(probability=round(probability, 4), classification=classification)
