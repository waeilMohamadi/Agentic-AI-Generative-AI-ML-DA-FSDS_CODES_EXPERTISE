# BTCVision Predictor
This project is a Streamlit web application that predicts the closing price of Bitcoin (BTC) based on market data from Tether (USDT) and Binance Coin (BNB).

## Overview

The app uses a pre-trained **Random Forest Regressor** model to make predictions. Users can input the closing price and volume for both USDT and BNB, and the model will estimate the BTC closing price.

## Features

-   **Interactive Sidebar**: Easy-to-use input fields for market data.
-   **Real-time Prediction**: Instant prediction updates upon clicking the "Predict" button.
-   **Multi-Currency Converter**: Convert between USDT, BNB, ETH, XRP, SOL, USDC, TRX, DOGE, and ADA using real-time user inputs.
-   **Model Transparency**: Detailed breakdown of how the AI model processes inputs (Scaling & Logic explanation).
-   **Robust Preprocessing**: Includes manual feature scaling based on the original training dataset to ensure accurate model inputs.
-   **High-Tech Crypto UI**: Modern dark theme with gradient typography, glassmorphism effects, and a **Sticky/Fixed Header** for easy navigation.

## Installation

1.  **Clone or Download** this repository.
2.  **Install Dependencies**:
    Ensure you have Python installed. Install the required libraries using pip:

    ```bash
    pip install streamlit pandas scikit-learn numpy
    ```

## Deployment
The easiest way to deploy this app is with **Streamlit Community Cloud**.

1.  **GitHub**:
    -   Create a new repository on GitHub.
    -   Upload all files (including `requirements.txt` and `random_forest_model.pkl`) to the repository.
2.  **Streamlit Cloud**:
    -   Go to [share.streamlit.io](https://share.streamlit.io/).
    -   Log in and click "New app".
    -   Select your GitHub repository.
    -   Set **Main file path** to `app.py`.
    -   Click **Deploy**!

## Usage

1.  Navigate to the project directory in your terminal:
    ```bash
    cd path/to/project
    ```

2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

3.  The application will open in your default web browser (usually at `http://localhost:8501`).

## Model Details

-   **Model**: Random Forest Regressor (`random_forest_model.pkl`)
-   **Input Features**:
    1.  `Close (USDT)`
    2.  `Volume (USDT)`
    3.  `Close (BNB)`
    4.  `Volume (BNB)`
-   **Scaling**: The app applies Min-Max scaling to inputs using statistics derived from historical crypto data (Nov 2017 - Jan 2020).

## Files

-   `app.py`: The main application script.
-   `random_forest_model.pkl`: The saved machine learning model.
-   `Bitcoinpriceprediction.ipynb`: (Optional) The Jupyter notebook used for data analysis and model training.
