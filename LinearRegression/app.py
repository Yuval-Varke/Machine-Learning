import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(
    page_title="House Price Prediction & Analytics",
    page_icon="🏠",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    [data-testid="stMetricValue"] {
        color: black !important;
    }
    [data-testid="stMetricLabel"] {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and data
@st.cache_resource
def load_model():
    with open("linear_regression_price_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data
def load_data():
    return pd.read_csv("USA_Housing.csv")

model = load_model()
df = load_data()

# --- SIDEBAR: INPUTS ---
st.sidebar.title("Control Panel")
st.sidebar.markdown("Adjust properties to see dynamic prediction.")

def dual_input(label, key, min_val, max_val, default_val, step=0.1):
    """Creates a synced slider and number input using session state."""
    if key not in st.session_state:
        st.session_state[key] = float(default_val)
    
    st.sidebar.subheader(label)
    
    # Synchronization logic
    def update_slider():
        st.session_state[f"slide_{key}"] = st.session_state[f"num_{key}"]

    def update_num():
        st.session_state[f"num_{key}"] = st.session_state[f"slide_{key}"]

    val_slide = st.sidebar.slider(f"Slider for {label}", min_value=min_val, max_value=max_val, 
                                  step=step, key=f"slide_{key}", on_change=update_num, label_visibility="collapsed")
    
    val_num = st.sidebar.number_input(f"Edit Value", min_value=min_val, max_value=max_val, 
                                     step=0.01, format="%.2f", key=f"num_{key}", on_change=update_slider)
    
    return val_slide

income = dual_input("Avg. Area Income ($)", "income", 10000.0, 150000.0, 68000.0, 1.0)
age = dual_input("Avg. Area House Age", "age", 1.0, 15.0, 6.0, 0.01)
rooms = dual_input("Avg. Area Number of Rooms", "rooms", 1.0, 15.0, 7.0, 0.01)
bedrooms = dual_input("Avg. Area Number of Bedrooms", "bedrooms", 1.0, 10.0, 4.0, 0.01)
population = dual_input("Area Population", "pop", 100.0, 100000.0, 36000.0, 1.0)

# --- DASHBOARD ---
st.title("🏙️ House Price Prediction & Analytics")

# Prediction
input_features = np.array([[income, age, rooms, bedrooms, population]])
predicted_price = model.predict(input_features)[0]

# Metric row
col_metric, col_info = st.columns([1, 2])

with col_metric:
    st.metric(label="Estimated Market Value", value=f"${predicted_price:,.2f}")
    if st.button("Reset to Default", use_container_width=True):
        for key in ["income", "age", "rooms", "bedrooms", "pop"]:
            for suffix in ["", "num_", "slide_"]:
                full_key = f"{suffix}{key}"
                if full_key in st.session_state:
                    del st.session_state[full_key]
        st.rerun()


# Visualizations
tab1, tab2 = st.tabs(["Market Distribution", "Feature Correlations"])

with tab1:
    st.subheader("Property Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['Price'], kde=True, color="#2E86C1", ax=ax)
    plt.axvline(predicted_price, color='red', linestyle='--', label=f'Your Prediction: ${predicted_price/1e6:.2f}M')
    plt.title("USA Housing Market Price Density")
    plt.xlabel("Price ($)")
    plt.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Feature Correlation with Market Price")
    feature_to_plot = st.selectbox("Select feature to investigate:", 
                                   ["Avg. Area Income", "Avg. Area House Age", "Avg. Area Number of Rooms", "Area Population"])
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sample_df = df.sample(1000)
    sns.scatterplot(x=feature_to_plot, y='Price', data=sample_df, hue='Price', palette='viridis', alpha=0.6, ax=ax2, legend=False)
    sns.regplot(x=feature_to_plot, y='Price', data=sample_df, scatter=False, line_kws={'color':'#E74C3C', 'linewidth':2}, ax=ax2)
    plt.title(f"{feature_to_plot} vs Market Price (Market Intensity)")
    st.pyplot(fig2)

st.divider()
