import streamlit as st
import pandas as pd
import numpy as np
from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_invoice_flag

# Page configuration
st.set_page_config(
    page_title="Vendor Invoice Intelligence Portal",
    page_icon="🛡️",
    layout="wide"
)

# Custom Styling (CSS Injection)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Apply font globally */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
}

/* Background gradient */
.stApp {
    background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 60%, #090d16 100%) !important;
    color: #f1f5f9 !important;
}

/* Sidebar styles */
section[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.9) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(12px);
}

section[data-testid="stSidebar"] .stMarkdown h1, 
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
    font-weight: 700;
}

/* Custom form wrapper styles */
div[data-testid="stForm"] {
    background: rgba(30, 41, 59, 0.35) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    padding: 2.5rem !important;
    box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.4) !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

div[data-testid="stForm"]:hover {
    border-color: rgba(99, 102, 241, 0.4) !important;
    box-shadow: 0 15px 45px 0 rgba(99, 102, 241, 0.12) !important;
}

/* Input Fields styling */
div[data-baseweb="input"], div[data-baseweb="select"] {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: all 0.2s ease;
}

div[data-baseweb="input"]:focus-within {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 2px rgba(129, 140, 248, 0.2) !important;
}

/* Styled Streamlit Button */
.stButton>button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px 0 rgba(99, 102, 241, 0.35) !important;
    width: 100% !important;
    margin-top: 10px !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 22px 0 rgba(99, 102, 241, 0.5) !important;
    background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%) !important;
}

.stButton>button:active {
    transform: translateY(0) !important;
}

/* Radio buttons container */
div[data-testid="stRadio"] div[role="radiogroup"] {
    background-color: rgba(15, 23, 42, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

div[data-testid="stRadio"] label {
    font-weight: 500 !important;
    color: #cbd5e1 !important;
}

/* Dividers */
hr {
    border-color: rgba(255, 255, 255, 0.08) !important;
    margin: 2rem 0 !important;
}

/* Subheadings styling */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif !important;
    color: #f8fafc !important;
}

/* Label styling override */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
    margin-bottom: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div style="padding: 2.5rem 0rem; text-align: center;">
    <span style="background: rgba(99, 102, 241, 0.15); color: #a5b4fc; padding: 0.4rem 1.2rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; border: 1px solid rgba(99, 102, 241, 0.3); text-transform: uppercase; letter-spacing: 1px;">
        🛡️ Vendor Audit & Compliance
    </span>
    <h1 style="font-size: 3.5rem; margin-top: 1rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #ffffff 40%, #c7d2fe 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; letter-spacing: -1px;">
        Invoice Intelligence Portal
    </h1>
    <p style="font-size: 1.15rem; color: #94a3b8; max-width: 750px; margin: 0 auto 2rem auto; line-height: 1.6;">
        Leveraging advanced machine learning models to detect financial risk, audit vendor pricing anomalies, and forecast logistics costs.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0;">
    <h3 style="margin-bottom: 5px;">🤖 AI Engine Control</h3>
    <p style="color: #64748b; font-size: 0.85rem; margin-bottom: 20px;">Select the cognitive module to evaluate.</p>
</div>
""", unsafe_allow_html=True)

selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [
        "Freight Cost Prediction",
        "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
<div style="margin-top: 30px; padding: 1.2rem; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px;">
    <h4 style="font-size: 0.95rem; margin-bottom: 8px; color: #e2e8f0;">⚡ Business Impact</h4>
    <ul style="color: #94a3b8; font-size: 0.85rem; padding-left: 15px; margin: 0; line-height: 1.6;">
        <li>Optimized Freight spend prediction</li>
        <li>Automated risk assessment</li>
        <li>Sub-second invoice screening latency</li>
    </ul>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Freight Cost Prediction
# ---------------------------------------------------------
if selected_model == "Freight Cost Prediction":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 5px;">🚛 Freight Cost Forecasting</h2>
        <p style="color: #94a3b8; font-size: 0.95rem;">Forecast the freight charges of vendor shipping based on the Invoice Value.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("freight_form"):
        col1, col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "📦 Quantity (Units Ordered)",
                min_value=1,
                value=1200
            )
            
        with col2:
            dollars = st.number_input(
                "💰 Invoice Dollars ($)",
                min_value=1.0,
                value=18500.0
            )

        submit_freight = st.form_submit_button("🔮 Calculate Predicted Freight")

    if submit_freight:
        input_data = {
            "Dollars": [dollars]
        }

        try:
            prediction = predict_freight_cost(input_data)['Predicted_Freight']
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(56, 189, 248, 0.06) 100%); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 16px; padding: 2rem; text-align: center; margin-top: 1.5rem; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);">
                <div style="color: #a5b4fc; font-size: 0.95rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">📊 Estimated Freight Cost</div>
                <div style="font-size: 3.2rem; font-weight: 800; color: #38bdf8; background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">${prediction[0]:,.2f}</div>
                <p style="color: #64748b; font-size: 0.85rem; margin: 8px 0 0 0;">Evaluation executed using the trained Regression engine.</p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"⚠️ **Error running prediction:** {str(e)}")

# ---------------------------------------------------------
# Invoice Flag Prediction
# ---------------------------------------------------------
else:
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h2 style="font-size: 1.8rem; font-weight: 700; margin-bottom: 5px;">🚨 Invoice Manual Approval Prediction</h2>
        <p style="color: #94a3b8; font-size: 0.95rem;">Evaluate if a vendor invoice exhibits pricing or quantity anomalies requiring human auditor review.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("invoice_flag_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "📦 Invoice Quantity",
                min_value=1,
                value=6
            )
            freight = st.number_input(
                "🚚 Freight Cost ($)",
                min_value=0.0,
                value=15.0
            )
            
        with col2:
            invoice_dollars = st.number_input(
                "💵 Invoice Dollars ($)",
                min_value=1.0,
                value=352.95
            )
            total_item_quantity = st.number_input(
                "🔢 Total Item Quantity (PO)",
                min_value=1,
                value=162
            )

        with col3:
            total_item_dollars = st.number_input(
                "⚖️ Total Item Dollars (PO, $)",
                min_value=1.0,
                value=2476.0
            )

        submit_flag = st.form_submit_button("🧠 Analyze Audit Risk")

    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "Freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }

        flag_prediction = predict_invoice_flag(input_data)['Predicted_Flag']

        flagged = bool(flag_prediction[0])

        if flagged:
            st.markdown(f"""
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 16px; padding: 1.8rem; display: flex; align-items: center; gap: 1.2rem; margin-top: 1.5rem;">
                <span style="font-size: 2.2rem;">🚨</span>
                <div>
                    <h4 style="color: #fca5a5; margin: 0; font-size: 1.2rem; font-weight: 700;">Manual Approval Required</h4>
                    <p style="color: #f87171; margin: 4px 0 0 0; font-size: 0.95rem; line-height: 1.5;">This invoice exhibits significant pricing, quantity, or freight cost discrepancies and has been flagged for audit review.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.25); border-radius: 16px; padding: 1.8rem; display: flex; align-items: center; gap: 1.2rem; margin-top: 1.5rem;">
                <span style="font-size: 2.2rem;">✅</span>
                <div>
                    <h4 style="color: #86efac; margin: 0; font-size: 1.2rem; font-weight: 700;">Invoice Safe for Auto-Approval</h4>
                    <p style="color: #4ade80; margin: 4px 0 0 0; font-size: 0.95rem; line-height: 1.5;">The invoice patterns conform to typical purchase orders and are safe for automated clearing.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)