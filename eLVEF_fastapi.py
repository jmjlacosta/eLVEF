import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Annotated, Literal

app = FastAPI(
    title="Estimation of Reduced LVEF",
    description="Predicts the likelihood of Reduced Left Ventricular Ejection Fraction based on diagnosis and medication inputs.",
    version="1.0.0",
)

# Enable CORS to allow external clients to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProbabilityResponse(BaseModel):
    """Schema for API response."""
    probability: float = Field(..., description="Predicted probability of reduced ejection fraction")
    classification: str = Field(..., description="Classification based on the probability threshold")

@app.post(
    "/calculate_probability",
    response_model=ProbabilityResponse,
    summary="Calculate Probability of Reduced Ejection Fraction",
    description="Computes the probability and classification of a patient having reduced left ventricular ejection fraction."
)
def calculate_probability(
    request: Request,
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
) -> ProbabilityResponse:
    """Compute eLVEF probability using logistic regression based on patient attributes."""
    
    # Convert "True"/"False" strings to 1/0
    def to_int(value: str) -> int:
        return 1 if value == "True" else 0

    # Convert boolean-like values
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

    # Compute probability using logistic regression
    LP0 = -1.37218706653502
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
    LP0 += 1.415113 * dx_cardiomyopathy
    LP0 += 0.651778 * dx_mi

    hf_type_mapping = {
        "Systolic": 0.754954,
        "Diastolic": -0.950856,
        "Left": 0.766415,
        "Unspecified": -0.577221
    }
    LP0 += hf_type_mapping[hf_type]

    probability = 1 / (1 + np.exp(-LP0))
    classification = "Reduced Ejection Fraction" if probability > 0.4678 else "Preserved Ejection Fraction"

    return ProbabilityResponse(probability=round(probability, 4), classification=classification)
