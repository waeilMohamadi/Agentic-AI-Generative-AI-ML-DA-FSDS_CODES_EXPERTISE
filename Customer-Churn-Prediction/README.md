# Customer Churn Prediction API

A reproducible machine learning project that predicts whether a telecom customer is likely to leave. It rebuilds a course notebook as a training pipeline with a Gradio browser interface and an optional FastAPI service. Both use the same saved preprocessing and model pipeline.

## Goals and scope

- Compare a logistic regression baseline with gradient boosting using five-fold cross-validation on training data only.
- Clean numeric and categorical inputs inside the same scikit-learn pipeline used at prediction time.
- Evaluate the selected model once on an untouched, stratified test split.
- Serve a validated JSON prediction endpoint with probability and a stated classification threshold.
- Explore predictions in a Gradio form with customer and service details.

## Data

The included `data/Telco-Customer-Churn.csv` is the IBM Telco Customer Churn sample dataset (7,043 rows, 21 columns), also available in [IBM's source repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv). `customerID` is excluded from modeling, and `Churn` is the label. Blank `TotalCharges` values become missing numeric values and are imputed using training-fold statistics. This is sample data, not customer information from a real deployment.

## Training and results

The project uses an 80/20 stratified split (5,634 training and 1,409 test rows). Each cross-validation fold fits its own imputation, scaling, and one-hot encoding on that fold's training data. No SMOTE is applied before the split. The model with the higher mean cross-validation average precision is selected and refit on all training rows.

| Model | Mean 5-fold train average precision |
| --- | ---: |
| Logistic regression with balanced class weights | 0.6600 |
| Histogram gradient boosting | **0.6614** |

Selected model: histogram gradient boosting. On the untouched test set: **ROC AUC 0.8436**, **average precision 0.6556**, **accuracy 0.8020**, and **churn recall 0.5187** at the fixed 0.5 threshold. Results are from this run with `random_state=42`; `metrics.json` records the detailed classification report. The small model-selection difference does not establish that boosting is universally better. This is a learning project, not a validated retention policy.

## Tech stack

Python, pandas, scikit-learn (`Pipeline`, `ColumnTransformer`, `OneHotEncoder`, cross-validation), joblib, Gradio, FastAPI, Pydantic, and Uvicorn.

## Run on Windows PowerShell

Open PowerShell in this folder. Use Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe gradio_app.py
```

The training command creates `models/churn_pipeline.joblib` and refreshes `metrics.json`. The model file is ignored by Git and must be trained locally. The Gradio command opens a local browser form for changing customer attributes and viewing the predicted churn probability.

To run the optional JSON API instead, stop Gradio with Ctrl+C and run:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API form. Use the field values in `example_request.json` to try `POST /predict`. `GET /health` reports whether the model file exists. From PowerShell, a direct request is:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType 'application/json' -InFile .\example_request.json
```

Sample response (probability depends on the installed package versions and retraining):

```json
{"churn_probability": 0.6912, "prediction": "churn", "threshold": 0.5}
```

## Project files

- `train.py`: data cleaning, train/test split, cross-validation, training, metrics, and model export.
- `app.py`: validated FastAPI inference endpoint; loads only the locally trained artifact.
- `gradio_app.py`: browser form backed by the same trained pipeline.
- `data/Telco-Customer-Churn.csv`: sample data for reproduction.
- `example_request.json`: one sample input without ID or target.
- `metrics.json`: metrics from the included training run.
- `models/`: generated model artifact (ignored by Git).

## Limitations

The API accepts valid types but does not enforce every category choice. Unseen categories are ignored by the encoder. Probability estimates are not calibrated, and 0.5 is a demonstration threshold rather than one chosen from retention costs. Accuracy alone would hide missed churners. Do not load model files from untrusted sources: joblib uses Python pickle internally.

The original notebook and saved pickle were reviewed as project inputs but are not used for deployment. The notebook's pre-split SMOTE and the original Streamlit app's separate feature encoding would make evaluation and inference unreliable.
