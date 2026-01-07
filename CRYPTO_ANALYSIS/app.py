import streamlit as st
import pandas as pd
import pickle
import os

# Set page config for a wider layout and custom title
st.set_page_config(page_title="BTCVision Predictor", page_icon="📈", layout="wide")

# Load the trained model
# Use absolute path relative to this script to ensure it works on Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'random_forest_model.pkl')

with open(model_path, 'rb') as file:
    model_rf = pickle.load(file)

# Feature statistics from training data (Min, Max)
FEATURE_STATS = {
    'Close (USDT)': {'min': 0.974248, 'max': 1.053585},
    'Volume (USDT)': {'min': 9.989859e+09, 'max': 2.790675e+11},
    'Close (BNB)': {'min': 9.386050, 'max': 710.464050},
    'Volume (BNB)': {'min': 1.061036e+08, 'max': 1.798295e+10}
}

def scale_value(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def predict_btc_price(usdt_close, usdt_volume, bnb_close, bnb_volume):
    scaled_data = {
        'Close (USDT)': scale_value(usdt_close, FEATURE_STATS['Close (USDT)']['min'], FEATURE_STATS['Close (USDT)']['max']),
        'Volume (USDT)': scale_value(usdt_volume, FEATURE_STATS['Volume (USDT)']['min'], FEATURE_STATS['Volume (USDT)']['max']),
        'Close (BNB)': scale_value(bnb_close, FEATURE_STATS['Close (BNB)']['min'], FEATURE_STATS['Close (BNB)']['max']),
        'Volume (BNB)': scale_value(bnb_volume, FEATURE_STATS['Volume (BNB)']['min'], FEATURE_STATS['Volume (BNB)']['max'])
    }
    input_df = pd.DataFrame([scaled_data], columns=['Close (USDT)', 'Volume (USDT)', 'Close (BNB)', 'Volume (BNB)'])
    prediction = model_rf.predict(input_df)
    return prediction[0]

def local_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Dark Theme Backgrounds */
        .stApp {
            background-color: #0E1117;
            background-image: radial-gradient(circle at 50% 0%, #1c2331 0%, #0E1117 70%);
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #161B22;
            border-right: 1px solid #2d333b;
        }
        
        /* Gradient Title */
        .title-text {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, #00F2EA 0%, #FF0050 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .subtitle-text {
            font-size: 1.2rem;
            color: #8b949e;
            margin-bottom: 2rem;
        }

        /* Fixed Header */
        .sticky-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 999999;
            background-color: #0E1117;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center;
        }
        
        /* Spacer to prevent content from hiding behind fixed header */
        .header-spacer {
            height: 100px;
            width: 100%;
        }

        /* Result Card Glassmorphism */
        .result-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            margin-top: 2rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        }
        
        .result-label {
            color: #8b949e;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
        }
        
        .result-value {
            font-size: 3.5rem;
            font-weight: 700;
            color: #ffffff;
            text-shadow: 0 0 20px rgba(0, 242, 234, 0.3);
        }

        /* Button Styling */
        .stButton > button {
            width: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, #00F2EA 0%, #0078FF 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .stButton > button:hover {
            opacity: 0.9;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 242, 234, 0.4);
        }
        
        /* Input Field Styling */
        .stNumberInput label {
            color: #c9d1d9 !important;
            font-weight: 600;
        }
        
    </style>
    """, unsafe_allow_html=True)

def main():
    local_css()

    # Inject Fixed Header (Outside Columns for full width)
    st.markdown("""
    <div class="sticky-header">
        <div class="title-text">BTCVision Predictor</div>
        <div class="subtitle-text">Advanced AI-powered Bitcoin price forecasting based on market correlation analytics.</div>
    </div>
    <div class="header-spacer"></div>
    """, unsafe_allow_html=True)

    
    # Sidebar Inputs
    with st.sidebar:
        st.image("https://cryptologos.cc/logos/bitcoin-btc-logo.png", width=50)
        st.markdown("### Market Data Input")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### USDT")
            usdt_close = st.number_input('Close Price ($)', min_value=0.0, value=1.0000, format="%.4f", key="usdt_c")
            usdt_volume = st.number_input('Volume', min_value=0.0, value=50000000000.0, step=100000000.0, key="usdt_v")
        
        st.markdown("---")
        
        with col2:
            st.markdown("#### BNB")
            bnb_close = st.number_input('Close Price ($)', min_value=0.0, value=300.00, format="%.2f", key="bnb_c")
            bnb_volume = st.number_input('Volume', min_value=0.0, value=1000000000.0, step=1000000.0, key="bnb_v")
            
        st.markdown("---")
        
        # Additional Currencies for Converter
        with st.expander("📉 Other Crypto Rates", expanded=False):
            eth_price = st.number_input('ETH Price ($)', value=2250.00, format="%.2f")
            xrp_price = st.number_input('XRP Price ($)', value=0.60, format="%.4f")
            sol_price = st.number_input('SOL Price ($)', value=140.00, format="%.2f")
            usdc_price = st.number_input('USDC Price ($)', value=1.00, format="%.4f")
            trx_price = st.number_input('TRX Price ($)', value=0.12, format="%.4f")
            doge_price = st.number_input('DOGE Price ($)', value=0.15, format="%.4f")
            ada_price = st.number_input('ADA Price ($)', value=0.45, format="%.4f")

        # Dictionary of all available prices
        prices = {
            "USDT": usdt_close,
            "BNB": bnb_close,
            "ETH": eth_price,
            "XRP": xrp_price,
            "SOL": sol_price,
            "USDC": usdc_price,
            "TRX": trx_price,
            "DOGE": doge_price,
            "ADA": ada_price
        }

        st.markdown("---")
        
        st.markdown("### Currency Converter")
        
        # Two columns for From/To selection
        c1, c2 = st.columns(2)
        with c1:
            from_curr = st.selectbox("From", list(prices.keys()), index=0, key="from_curr")
        with c2:
            to_curr = st.selectbox("To", list(prices.keys()), index=1, key="to_curr")
            
        conv_amount = st.number_input("Amount", min_value=0.0, value=100.0, key="conv_amt")
        
        if params_ok := (prices[from_curr] > 0 and prices[to_curr] > 0):
            # Conversion logic: 
            # Value in USD = Amount * Price_From
            # Value in Target = Value in USD / Price_To
            # Result = Amount * (Price_From / Price_To)
            
            rate = prices[from_curr] / prices[to_curr]
            result = conv_amount * rate
            
            st.info(f"""
            **{conv_amount:,.2f} {from_curr}**
            ≈
            **{result:,.4f} {to_curr}**
            
            Rate: 1 {from_curr} = {rate:,.4f} {to_curr}
            """)
        else:
            st.warning("Prices must be > 0")

        st.markdown("---")
        predict_btn = st.button('🚀 Predict Price')

    # Main Content Area
    col_spacer, col_main, col_spacer2 = st.columns([1, 2, 1])
    
    with col_main:
        # Sticky Header Wrapper
        st.markdown("""
        <div class="sticky-header">
            <div class="title-text">BTCVision Predictor</div>
            <div class="subtitle-text">Advanced AI-powered Bitcoin price forecasting based on market correlation analytics.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if predict_btn:
            with st.spinner('Analyzing market patterns...'):
                try:
                    predicted_price = predict_btc_price(usdt_close, usdt_volume, bnb_close, bnb_volume)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-label">Predicted Bitcoin Close Price</div>
                        <div class="result-value">${predicted_price:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Explain the calculation (Show Scaled Values)
                    with st.expander("ℹ️ How was this calculated? (Model Inputs)"):
                        st.write("The AI model doesn't use the raw values (like billions for volume). It scales everything to a 0-1 range first.")
                        
                        # Re-calculate scaled values for display
                        s_usdt_c = scale_value(usdt_close, FEATURE_STATS['Close (USDT)']['min'], FEATURE_STATS['Close (USDT)']['max'])
                        s_usdt_v = scale_value(usdt_volume, FEATURE_STATS['Volume (USDT)']['min'], FEATURE_STATS['Volume (USDT)']['max'])
                        s_bnb_c = scale_value(bnb_close, FEATURE_STATS['Close (BNB)']['min'], FEATURE_STATS['Close (BNB)']['max'])
                        s_bnb_v = scale_value(bnb_volume, FEATURE_STATS['Volume (BNB)']['min'], FEATURE_STATS['Volume (BNB)']['max'])
                        
                        st.code(f"""
# 1. Scaling Process (Formula: (Input - Min) / (Max - Min))

USDT Close:  ({usdt_close:.4f} - {FEATURE_STATS['Close (USDT)']['min']:.4f}) / ... = {s_usdt_c:.4f}
USDT Volume: ({usdt_volume:,.0f} - {FEATURE_STATS['Volume (USDT)']['min']:,.0f}) / ... = {s_usdt_v:.4f}
BNB Close:   ({bnb_close:.2f} - {FEATURE_STATS['Close (BNB)']['min']:.2f}) / ... = {s_bnb_c:.4f}
BNB Volume:  ({bnb_volume:,.0f} - {FEATURE_STATS['Volume (BNB)']['min']:,.0f}) / ... = {s_bnb_v:.4f}

# 2. Prediction
The model takes these 4 scaled numbers: [{s_usdt_c:.4f}, {s_usdt_v:.4f}, {s_bnb_c:.4f}, {s_bnb_v:.4f}]
It passes them through hundreds of "Decision Trees" to find the pattern matching ${predicted_price:,.2f}.
This is NOT a simple average or sum of the inputs.
                        """)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")



if __name__ == '__main__':
    main()
