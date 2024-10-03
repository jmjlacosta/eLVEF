# Medicare Claims-Based EF Model
This repository contains an implementation of a logistic regression model that predicts Left Ventricular Ejection Fraction (EF) class in patients with heart failure (HF) using Medicare claims data. The model is based on the paper: "Development and Preliminary Validation of a Medicare Claims–Based Model to Predict Left Ventricular Ejection Fraction Class in Patients With Heart Failure" by Desai et al.

The model aims to differentiate patients into two EF classes: reduced EF (<0.45) and preserved EF (≥0.45), leveraging administrative claims data in scenarios where EF measurements are unavailable.

**Input:** Patient Data Input: Users manually input key patient data, including demographics, diagnoses, and treatment history.

**Prediction:** The model processes the inputs and returns the predicted EF class (reduced or preserved).

**EF Class Prediction:** The model uses logistic regression to classify patients as having either reduced EF or preserved EF.

# Model Details
The logistic regression model was trained and validated using data from two academic medical centers:

Training Sample: The model was developed using data from one center, with 35 predictors chosen from 57 potential candidate variables.
Testing Sample: The model was validated using data from a second center, with an accuracy of 83%, sensitivity of 0.97 for preserved EF, and a positive predictive value of 0.73 for reduced EF.

# Summary
This application provides a practical tool for healthcare researchers and professionals to estimate EF class in heart failure patients using Medicare claims data, helping to bridge the gap when EF measurements are not available. It can be used in studies evaluating health outcomes, healthcare utilization, and cost among heart failure patients.