import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
from pathlib import Path

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="F&B Anomaly Detection",
    page_icon="🍞",
    layout="wide"
)

# --- HELPER FUNCTION FOR DOWNLOADS ---
@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    """Loads all necessary artifacts for the dashboard."""
    model = joblib.load("artifacts/model_training/model.pkl")
    process_df = pd.read_csv("artifacts/data_generation/process_data.csv")
    quality_df = pd.read_csv("artifacts/data_generation/batch_quality.csv")
    model_ready_df = pd.read_csv("artifacts/feature_engineering/model_ready_data.csv")
    with open("artifacts/model_evaluation/metrics.json") as f:
        metrics = json.load(f)
    shap_plot_path = "artifacts/model_evaluation/shap_summary_plot.png"
    return model, process_df, quality_df, model_ready_df, metrics, shap_plot_path

model, process_df, quality_df, model_ready_df, metrics, shap_plot_path = load_artifacts()

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("F&B Process Monitor")
# ADDED NEW PAGE TO NAVIGATION
page = st.sidebar.radio("Navigate", ["📊 Overview", "💾 Dataset Deep Dive", "📈 Live Anomaly Prediction", "🔬 Model Performance"])

# --- PAGE 1: OVERVIEW ---
if page == "📊 Overview":
    st.title("Project Justification & Executive Summary 🏆")
    st.markdown("""
    Welcome to the F&B Process Anomaly Detection dashboard. This project delivers a complete, end-to-end machine learning solution for predicting quality anomalies in an industrial baking process. 
    This overview page directly addresses the **six key deliverables** of the hackathon, showcasing our methodology, unique solutions, and the final results.
    """)
    # ... (Rest of the Overview page code remains the same)
    st.divider()
    st.header("1. The Process Decoded: Industrial Bread Baking")
    st.graphviz_chart('''
    digraph { rankdir="LR"; node [shape=box, style=rounded, fontname="sans-serif"];
        ingredient [label="1. Ingredient Scaling\n(Flour, Water, Yeast)"];
        mixing [label="2. Mixing\n(Time, RPM, Dough Temp)"];
        proofing [label="3. Proofing\n(Time, Temp, Humidity)"];
        baking [label="4. Baking\n(Time, Temp Profile, Stability)"];
        cooling [label="5. Cooling & Packaging"];
        ingredient -> mixing -> proofing -> baking -> cooling;
    }
    ''')
    st.divider()
    st.header("2 & 3. The Data Strategy: High-Fidelity Synthetic Generation (Our X-Factor)")
    st.markdown("Below is a comparison of the 'Oven Temperature' data stream for a normal batch vs. a batch with a simulated oven failure.")
    col1, col2 = st.columns(2)
    with col1:
        good_batch_data = quality_df[quality_df['Is_Anomaly']==0].head(1)
        good_batch_id = good_batch_data['Batch_ID'].iloc[0]
        good_df = process_df[process_df['Batch_ID'] == good_batch_id]
        st.subheader("Normal Batch Signature")
        st.line_chart(good_df.set_index('Time')['Oven Temp (C)'])
    with col2:
        process_features = process_df.groupby('Batch_ID')['Oven Temp (C)'].std().reset_index()
        bad_batch_id = process_features.sort_values(by='Oven Temp (C)', ascending=False)['Batch_ID'].iloc[0]
        bad_df = process_df[process_df['Batch_ID'] == bad_batch_id]
        st.subheader("Anomalous Batch Signature")
        st.line_chart(bad_df.set_index('Time')['Oven Temp (C)'])
    st.divider()
    st.header("4. Quantifying Product Quality")
    quality_dist = quality_df['Is_Anomaly'].value_counts().reset_index()
    quality_dist.columns = ['Is_Anomaly', 'count']
    quality_dist['Is_Anomaly'] = quality_dist['Is_Anomaly'].map({0: 'Normal', 1: 'Anomaly'})
    fig_pie = px.pie(quality_dist, names='Is_Anomaly', values='count', title='Distribution of Batches in our Dataset', color='Is_Anomaly', color_discrete_map={'Normal':'#2ca02c', 'Anomaly':'#d62728'})
    st.plotly_chart(fig_pie, use_container_width=True)
    st.divider()
    st.header("5. The Intelligent Core: An Explainable AI Model")
    st.image(shap_plot_path, caption="SHAP Analysis: The features that have the biggest impact on our model's predictions.")
    st.divider()
    st.header("6. The Solution: This Interactive Dashboard")
    st.markdown("This Streamlit dashboard is the final deliverable, providing a user-friendly interface to interact with our solution.")


# --- NEW PAGE: DATASET DEEP DIVE ---
elif page == "💾 Dataset Deep Dive":
    st.title("Dataset Deep Dive")
    st.markdown("This section provides a detailed look into the high-fidelity synthetic dataset we engineered for this project.")

    # --- 1. Download Section ---
    st.header("Download the Datasets")
    st.markdown("Three key datasets were generated during our pipeline. You can download them here:")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            label="📥 Download Raw Process Data",
            data=convert_df_to_csv(process_df),
            file_name="process_data.csv",
            mime="text/csv",
        )
    with col2:
        st.download_button(
            label="📥 Download Batch Quality Data",
            data=convert_df_to_csv(quality_df),
            file_name="batch_quality.csv",
            mime="text/csv",
        )
    with col3:
        st.download_button(
            label="📥 Download Model-Ready Data",
            data=convert_df_to_csv(model_ready_df),
            file_name="model_ready_data.csv",
            mime="text/csv",
        )

    # --- 2. Dataset Preview ---
    st.header("Preview the Final Model-Ready Dataset")
    st.markdown("This is the final, feature-engineered dataset used to train our model. Each row represents one batch.")
    st.dataframe(model_ready_df, height=300)

    # --- 3. Feature Importance & Description ---
    st.header("Anatomy of a Prediction: Key Features Explained")
    st.markdown("Our model uses several powerful features to make its predictions. Here are the most important ones:")

    features_to_explain = {
        'Mixing Time (min)_mean': "The average time the dough was mixed. A critical process step; too little or too much time can ruin the gluten structure.",
        'hydration_deviation': "Our unique engineered feature. It measures the error between the actual water used and the ideal amount needed based on the flour's protein content.",
        'Oven Temp (C)_std': "The stability of the oven temperature. High standard deviation indicates a faulty heater, leading to unevenly baked products."
    }

    for feature, description in features_to_explain.items():
        with st.expander(f"🔬 Feature: {feature}"):
            st.write(description)
            
            # Create a histogram to show the distribution for normal vs. anomaly batches
            fig = px.histogram(model_ready_df, x=feature, color="Is_Anomaly", 
                               marginal="box", # Adds a box plot on top
                               barmode="overlay",
                               color_discrete_map={0:'#2ca02c', 1:'#d62728'},
                               labels={"Is_Anomaly": "Status (0=Normal, 1=Anomaly)"})
            st.plotly_chart(fig, use_container_width=True)
            st.info(f"Notice how the distributions for Normal (0) and Anomaly (1) batches differ for this feature. This separation is what the model learns from.")

    # --- 4. Anomaly Simulation Explained ---
    st.header("How We Created a Challenging Dataset")
    st.markdown("To train a robust model, we simulated three distinct, realistic industrial failure modes:")
    
    st.subheader("1. Equipment Failure: Unstable Oven")
    st.write("We simulated a faulty oven heater by introducing high variance into the `Oven Temp (C)` readings for certain batches. The model learns to detect this by looking at `Oven Temp (C)_std`.")

    st.subheader("2. Material Variance: Incorrect Hydration")
    st.write("We simulated batches where high-protein flour was used without adjusting the water content. Our `hydration_deviation` feature was specifically engineered to capture this subtle but critical error.")

    st.subheader("3. Operator Error: Short Mixing Time")
    st.write("We simulated batches where the mixing time was cut short. The model learns this simple but common mistake by analyzing the `Mixing Time (min)_mean` feature.")


# --- PAGE 3: LIVE ANOMALY PREDICTION ---
elif page == "📈 Live Anomaly Prediction":
    st.title("Live Anomaly Prediction")
    st.write("Adjust the sliders to simulate real-time process data and see the model's prediction.")

    st.sidebar.header("Simulate Batch Data")
    oven_temp_std = st.sidebar.slider("Oven Temp Stability (std)", 0.1, 6.0, 0.5, 0.1)
    hydration_dev = st.sidebar.slider("Hydration Deviation (%)", -10.0, 10.0, 0.0, 0.5)
    mixing_time = st.sidebar.slider("Mixing Time (min)", 20.0, 40.0, 35.0, 0.5)
    mixer_speed_std = st.sidebar.slider("Mixer Speed Stability (std)", 0.1, 5.0, 1.0, 0.1)
    mixer_speed_mean = st.sidebar.slider("Mixer Speed (RPM)", 130.0, 170.0, 150.0, 1.0)
    oven_temp_mean = st.sidebar.slider("Oven Temperature (°C)", 170.0, 190.0, 180.0, 0.5)

    input_data = pd.DataFrame({
        'Oven Temp (C)_mean': [oven_temp_mean], 'Oven Temp (C)_std': [oven_temp_std],
        'Mixer Speed (RPM)_mean': [mixer_speed_mean], 'Mixer Speed (RPM)_std': [mixer_speed_std],
        'Mixing Time (min)_mean': [mixing_time], 'hydration_deviation': [hydration_dev]
    })

    st.subheader("Current Input Parameters:")
    st.dataframe(input_data)

    if st.button("Predict Quality", type="primary"):
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        st.subheader("Prediction Result:")
        if prediction == 0:
            st.success(f"✅ STATUS: NORMAL (Anomaly Probability: {prediction_proba[1]:.2%})")
        else:
            st.error(f"🚨 STATUS: ANOMALY DETECTED (Anomaly Probability: {prediction_proba[1]:.2%})")


# --- PAGE 4: MODEL PERFORMANCE ---
elif page == "🔬 Model Performance":
    st.title("Model Performance & Explainability")
    st.write("Here we analyze the performance of our chosen XGBoost model.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Classification Report")
        st.json(metrics)
    with col2:
        st.subheader("Animated Feature Importance")
        shap_df = pd.DataFrame({
            'Feature': list(reversed([
                'Mixing Time (min)_mean', 'hydration_deviation', 'Oven Temp (C)_std',
                'Mixer Speed (RPM)_mean', 'Mixer Speed (RPM)_std', 'Oven Temp (C)_mean'
            ])),
            'Impact': list(reversed([2.3, 1.6, 0.8, 0.5, 0.5, 0.1]))
        })

        fig_bar = px.bar(shap_df, x='Impact', y='Feature', orientation='h', 
                         title="Feature Impact on Model Output")
        
        fig_bar.update_layout(transition_duration=800)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Model Explainability (SHAP Analysis)")
    st.image(shap_plot_path, caption="SHAP plot showing the impact of each feature on the model's predictions.")
    st.info("""
        **How to read this chart:** This plot shows the average impact of each feature on the model's output. The longer the bar, the more influence a feature has on the prediction. Our model intelligently uses multiple factors to determine the final quality.
    """)
