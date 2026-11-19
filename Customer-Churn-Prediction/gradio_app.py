"""Browser form for the locally trained customer churn model."""
from __future__ import annotations

import pandas as pd
import gradio as gr

from app import load_model
from train import DATA, NUMERIC, prepare_frame


data = prepare_frame(pd.read_csv(DATA))
features = [column for column in data.columns if column not in ("customerID", "Churn")]
sample = data.iloc[0]


def component(column: str):
    if column in NUMERIC:
        return gr.Number(
            label=column,
            value=float(sample[column]),
            precision=0 if column == "tenure" else 2,
        )
    choices = sorted(data[column].dropna().unique().tolist(), key=str)
    chosen = sample[column]
    if hasattr(chosen, "item"):
        chosen = chosen.item()
    return gr.Dropdown(label=column, choices=choices, value=chosen)


def predict(*values):
    try:
        artifact = load_model()
    except FileNotFoundError:
        raise gr.Error("Model missing. Run python train.py in this project folder first.")
    row = pd.DataFrame([dict(zip(features, values))])[artifact["features"]]
    if row["tenure"].iloc[0] is None or not 0 <= row["tenure"].iloc[0] <= 72:
        raise gr.Error("Tenure must be between 0 and 72 months.")
    for column in ("MonthlyCharges", "TotalCharges"):
        value = row[column].iloc[0]
        if value is not None and value < 0:
            raise gr.Error(f"{column} cannot be negative.")
    probability = float(artifact["pipeline"].predict_proba(row)[0, 1])
    outcome = "Likely to churn" if probability >= 0.5 else "Likely to stay"
    return (
        f"### {outcome}\nEstimated churn probability: **{probability:.1%}** "
        "(classification threshold: 50%).",
        {"Churn": probability, "Stay": 1 - probability},
    )


with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown("# Customer Churn Prediction")
    gr.Markdown(
        "Explore a sample telecom churn model. Edit the customer profile and click Predict. "
        "This is an educational estimate, not a retention decision."
    )
    controls = {}
    groups = {
        "Customer and billing": ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"],
        "Phone and internet services": ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"],
    }
    for title, columns in groups.items():
        with gr.Accordion(title, open=title == "Customer and billing"):
            for start in range(0, len(columns), 2):
                with gr.Row():
                    for column in columns[start:start + 2]:
                        controls[column] = component(column)
    button = gr.Button("Predict churn", variant="primary")
    summary = gr.Markdown()
    confidence = gr.Label(label="Prediction probabilities", num_top_classes=2)
    button.click(fn=predict, inputs=[controls[column] for column in features], outputs=[summary, confidence])


if __name__ == "__main__":
    demo.launch(inbrowser=True)
