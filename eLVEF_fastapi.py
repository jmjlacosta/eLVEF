from typing import Annotated, Literal, Optional
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(
    title="eLVEF Classification API",
    description=(
        "Predicts the likelihood of Reduced Left Ventricular Ejection Fraction (eLVEF) "
        "based on patient demographics, diagnoses, and medications."
    ),
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class ELVEFInput(BaseModel):
    """Form-based input schema for eLVEF prediction."""
    male: Optional[bool] = Field(None, description="Gender of the patient (male: True, female: False)")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age of the patient.")
    dx_defibrillator: Optional[bool] = Field(None, description="Has a defibrillator diagnosis.")
    hosp_chf: Optional[bool] = Field(None, description="History of CHF hospitalization.")
    rx_ace: Optional[bool] = Field(None, description="ACE inhibitors medication usage.")
    rx_antagonist: Optional[bool] = Field(None, description="Mineralocorticoid receptor antagonist usage.")
    rx_bblocker: Optional[bool] = Field(None, description="Beta blocker medication usage.")
    rx_digoxin: Optional[bool] = Field(None, description="Digoxin medication usage.")
    rx_loop_diuretic: Optional[bool] = Field(None, description="Loop diuretic medication usage.")
    rx_nitrates: Optional[bool] = Field(None, description="Nitrate medication usage.")
    rx_thiazide: Optional[bool] = Field(None, description="Thiazide medication usage.")
    dx_afib: Optional[bool] = Field(None, description="Atrial fibrillation diagnosis.")
    dx_anemia: Optional[bool] = Field(None, description="Anemia diagnosis.")
    dx_cabg: Optional[bool] = Field(None, description="CABG surgery history.")
    dx_cardiomyopathy: Optional[bool] = Field(None, description="Cardiomyopathy diagnosis.")
    dx_copd: Optional[bool] = Field(None, description="COPD diagnosis.")
    dx_depression: Optional[bool] = Field(None, description="Depression diagnosis.")
    dx_htn_nephropathy: Optional[bool] = Field(None, description="Hypertensive nephropathy diagnosis.")
    dx_hyperlipidemia: Optional[bool] = Field(None, description="Hyperlipidemia diagnosis.")
    dx_hypertension: Optional[bool] = Field(None, description="Hypertension diagnosis.")
    dx_hypotension: Optional[bool] = Field(None, description="Hypotension diagnosis.")
    dx_mi: Optional[bool] = Field(None, description="Myocardial infarction (MI) diagnosis.")
    dx_obesity: Optional[bool] = Field(None, description="Obesity diagnosis.")
    dx_oth_dysrhythmia: Optional[bool] = Field(None, description="Other dysrhythmia diagnosis.")
    dx_psychosis: Optional[bool] = Field(None, description="Psychosis diagnosis.")
    dx_rheumatic_heart: Optional[bool] = Field(None, description="Rheumatic heart disease diagnosis.")
    dx_sleep_apnea: Optional[bool] = Field(None, description="Sleep apnea diagnosis.")
    dx_stable_angina: Optional[bool] = Field(None, description="Stable angina diagnosis.")
    dx_valve_disorder: Optional[bool] = Field(None, description="Valve disorder diagnosis.")
    hf_type: Optional[Literal["Systolic", "Diastolic", "Left", "Unspecified"]] = Field(
        None, description="Type of heart failure diagnosed."
    )

# Output schema
class ELVEFOutput(BaseModel):
    """Output schema for eLVEF prediction results."""
    probability: float = Field(..., description="Probability of reduced ejection fraction.")
    classification: str = Field(..., description="Classification based on the probability.")

def calculate_linear_predictor(data: ELVEFInput) -> float:
    """Helper function to calculate the linear predictor."""
    intercept = -1.37218706653502
    LP0 = intercept

    # Add coefficients for all input features
    LP0 += 0.323651 * (data.male or 0)
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
        "Unspecified": -0.577221,
    }
    LP0 += hf_type_mapping.get(data.hf_type, 0)

    return LP0

@app.post(
    "/calculate_probability/",
    response_model=ELVEFOutput,
    summary="Predict eLVEF",
    description="Calculates the likelihood of reduced ejection fraction and provides classification.",
)
def calculate_probability(
    data: Annotated[ELVEFInput, Form()]
) -> ELVEFOutput:
    """Calculate probability and classify eLVEF."""
    LP0 = calculate_linear_predictor(data)

    # Logistic function to calculate probability
    probability = 1 / (1 + np.exp(-LP0))

    # Determine classification based on threshold
    classification = (
        "Reduced Ejection Fraction" if probability > 0.4678 else "Preserved Ejection Fraction"
    )

    return ELVEFOutput(
        probability=round(probability, 4),
        classification=classification,
    )
