"""
Streamlit App for Diabetes Readmission Prediction
Fixed: Handles PyTorch import errors gracefully
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import pickle

# Try importing PyTorch (optional)
PYTORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    st.warning("⚠️ PyTorch not available. Deep learning models will be disabled.")
    st.info("To enable PyTorch models: pip install --upgrade typing_extensions torch")

# Page config
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-high {
        background: #ff6b6b;
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
    .risk-medium {
        background: #feca57;
        padding: 1rem;
        border-radius: 8px;
        color: #333;
    }
    .risk-low {
        background: #48dbfb;
        padding: 1rem;
        border-radius: 8px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏥 Diabetes Readmission Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")


# Sidebar - Model Selection
st.sidebar.header("🎯 Model Configuration")

# Available models based on PyTorch availability
if PYTORCH_AVAILABLE:
    available_models = [
        "MultiTask MLP",
        "Transformer"
    ]
else:
    available_models = []
    st.sidebar.info("💡 Install PyTorch for deep learning models")

model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    available_models,
    index=0
)


# MultiTask MLP Definition (only if PyTorch available)
if PYTORCH_AVAILABLE:
    class MultiTaskMLP(nn.Module):
        def __init__(self, d_in, hidden=[128, 64], dropout=0.3):
            super().__init__()
            layers = []
            d = d_in
            for h in hidden:
                layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
                d = h
            self.encoder = nn.Sequential(*layers)
            self.readmit_head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
            self.los_head = nn.Sequential(nn.Linear(d, 16), nn.ReLU(), nn.Linear(16, 1))
        
        def forward(self, x):
            z = self.encoder(x)
            return torch.sigmoid(self.readmit_head(z)).squeeze(-1), self.los_head(z).squeeze(-1)


    class TransformerMultiTask(nn.Module):
        def __init__(self, d_in, emb=64, heads=2, layers=1, dropout=0.2):
            super().__init__()
            self.proj = nn.Linear(d_in, emb)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=emb, nhead=heads, dim_feedforward=emb*2,
                dropout=dropout, activation="gelu", batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
            self.readmit_head = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, 1))
            self.los_head = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, 1))
        
        def forward(self, x):
            x = self.proj(x.unsqueeze(1))
            z = self.encoder(x).mean(dim=1)
            return torch.sigmoid(self.readmit_head(z)).squeeze(-1), self.los_head(z).squeeze(-1)


# Load Models Function
@st.cache_resource
def load_models():
    """Load trained models and preprocessor"""
    models = {}
    
    try:
        # Load preprocessor
        try:
            models['scaler'] = joblib.load(Path("models/scaler.pkl"))
            with open(Path("models/feature_names.pkl"), "rb") as f:
                models['feature_names'] = pickle.load(f)
        except FileNotFoundError:
            st.error("Preprocessor files (scaler.pkl, feature_names.pkl) not found!")
        
        # Load Gradient Boosting
        gb_path = Path("models/gradient_boosting_model.pkl")
        if gb_path.exists():
            models['Gradient Boosting'] = joblib.load(gb_path)
        
        # Load XGBoost
        xgb_path = Path("models/xgboost_model.pkl")
        if xgb_path.exists():
            models['XGBoost'] = joblib.load(xgb_path)
        
        # Load Logistic Regression
        lr_path = Path("models/logistic_regression_model.pkl")
        if lr_path.exists():
            models['Logistic Regression'] = joblib.load(lr_path)
        
        # Load PyTorch models (only if available)
        if PYTORCH_AVAILABLE:
            # Load MultiTask MLP
            mlp_path = Path("models/multitask_mlp_model.pt")
            mlp_config_path = Path("models/mlp_config.pkl")
            if mlp_path.exists() and mlp_config_path.exists():
                mlp_config = joblib.load(mlp_config_path)
                mlp = MultiTaskMLP(d_in=mlp_config['input_dim'])
                mlp.load_state_dict(torch.load(mlp_path, map_location='cpu'))
                mlp.eval()
                models['MultiTask MLP'] = mlp
            
            # Load Transformer
            trf_path = Path("models/transformer_model.pt")
            trf_config_path = Path("models/transformer_config.pkl")
            if trf_path.exists() and trf_config_path.exists():
                trf_config = joblib.load(trf_config_path)
                trf = TransformerMultiTask(d_in=trf_config['input_dim'])
                trf.load_state_dict(torch.load(trf_path, map_location='cpu'))
                trf.eval()
                models['Transformer'] = trf
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models


def get_risk_category(probability):
    """Categorize readmission risk"""
    if probability >= 0.4:
        return "High Risk", "risk-high"
    elif probability >= 0.2:
        return "Medium Risk", "risk-medium"
    else:
        return "Low Risk", "risk-low"


def get_recommendations(probability, los_pred=None):
    """Generate clinical recommendations"""
    recommendations = []
    
    if probability >= 0.4:
        recommendations.extend([
            "🚨 Schedule early follow-up (within 7 days)",
            "📋 Intensive discharge planning required",
            "🏠 Consider home health services",
            "💊 Medication reconciliation critical"
        ])
    elif probability >= 0.2:
        recommendations.extend([
            "📅 Standard follow-up (within 14 days)",
            "📄 Review discharge instructions",
            "💊 Ensure medication compliance"
        ])
    else:
        recommendations.append("✅ Standard care pathway")
    
    if los_pred is not None:
        if los_pred > 7:
            recommendations.append("⏰ Extended stay expected - plan resources")
        elif los_pred > 4:
            recommendations.append("⏱️ Moderate LOS - monitor progress")
    
    return recommendations


def make_prediction(model, X_processed, model_name):
    """Make prediction with selected model"""
    try:
        if model_name in ['Gradient Boosting', 'XGBoost', 'Logistic Regression']:
            # Scikit-learn models
            prob = model.predict_proba(X_processed)[0, 1]
            return prob, None
        
        elif PYTORCH_AVAILABLE and model_name in ['MultiTask MLP', 'Transformer']:
            # PyTorch models
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_processed)
                readmit_logits, los_pred = model(X_tensor)
                prob = torch.sigmoid(readmit_logits).item()
                los = los_pred.item()
                return prob, los
        else:
            st.error("Model not available")
            return None, None
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None


# Main App
def main():
    # Load models
    with st.spinner("🔄 Loading models..."):
        models = load_models()
    
    if not models:
        st.error("❌ No models found! Please train models first.")
        st.info("Run: python train_ensemble.py")
        return
    

    
    # Check if selected model is available
    if model_choice not in models:
        st.error(f"❌ {model_choice} model not found!")
        st.info(f"Available models: {list(models.keys())}")
        return
    
    st.success(f"✅ Loaded {len(models)-1} models")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Single Prediction", "📁 Batch Upload", "ℹ️ About"])
    
    # Tab 1: Single Prediction
    with tab1:
        st.header("Enter Patient Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                                              "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
        
        with col2:
            admission_type = st.number_input("Admission Type ID", min_value=1, max_value=8, value=1)
            discharge_disposition = st.number_input("Discharge Disposition ID", min_value=1, max_value=30, value=1)
            admission_source = st.number_input("Admission Source ID", min_value=1, max_value=25, value=7)
        
        with col3:
            num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0, max_value=150, value=50)
            num_procedures = st.number_input("Number of Procedures", min_value=0, max_value=10, value=0)
            num_medications = st.number_input("Number of Medications", min_value=1, max_value=100, value=15)
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=50, value=0)
            number_emergency = st.number_input("Emergency Visits", min_value=0, max_value=50, value=0)
            number_inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=50, value=0)
        
        with col5:
            number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=9)
            max_glu_serum = st.selectbox("Max Glucose Serum", ["None", ">200", ">300", "Norm"])
            A1Cresult = st.selectbox("A1C Result", ["None", ">7", ">8", "Norm"])
        
        with col6:
            change = st.selectbox("Change in Medications", ["Ch", "No"])
            diabetesMed = st.selectbox("Diabetes Medication", ["Yes", "No"])
            time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=3)
        
        if st.button("🔮 Predict Readmission Risk", type="primary"):
            # Create input dataframe
            patient_data = pd.DataFrame([{
                'race': race,
                'gender': gender,
                'age': age,
                'admission_type_id': admission_type,
                'discharge_disposition_id': discharge_disposition,
                'admission_source_id': admission_source,
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_procedures': num_procedures,
                'num_medications': num_medications,
                'number_outpatient': number_outpatient,
                'number_emergency': number_emergency,
                'number_inpatient': number_inpatient,
                'number_diagnoses': number_diagnoses,
                'max_glu_serum': max_glu_serum,
                'A1Cresult': A1Cresult,
                'change': change,
                'diabetesMed': diabetesMed
            }])
            
            # Preprocess
            df_processed = patient_data.copy()
            df_encoded = pd.get_dummies(df_processed, drop_first=True)
            # Align features
            if 'feature_names' in models:
                for col in models['feature_names']:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_encoded = df_encoded[models['feature_names']]
            # Scale
            if 'scaler' in models:
                X_scaled = models['scaler'].transform(df_encoded)
            else:
                X_scaled = df_encoded.values

            X_processed = X_scaled
            
            # Predict
            prob, los_pred = make_prediction(models[model_choice], X_processed, model_choice)
            
            if prob is not None:
                risk_category, risk_class = get_risk_category(prob)
                recommendations = get_recommendations(prob, los_pred)
                
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                # Display results
                col_res1, col_res2, col_res3 = st.columns(3)
                
                with col_res1:
                    st.markdown(f'<div class="metric-card"><h2>{prob*100:.1f}%</h2><p>Readmission Probability</p></div>', unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f'<div class="{risk_class}"><h3>{risk_category}</h3></div>', unsafe_allow_html=True)
                
                with col_res3:
                    if los_pred is not None:
                        st.markdown(f'<div class="metric-card"><h2>{los_pred:.1f}</h2><p>Predicted LOS (days)</p></div>', unsafe_allow_html=True)
                    else:
                        st.info("LOS prediction not available for this model")
                
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Readmission Risk"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "#48dbfb"},
                            {'range': [20, 40], 'color': "#feca57"},
                            {'range': [40, 100], 'color': "#ff6b6b"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 40
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("💡 Clinical Recommendations")
                for rec in recommendations:
                    st.markdown(f"- {rec}")
    
    # Tab 2: Batch Upload
    with tab2:
        st.header("📁 Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader("Upload CSV file with patient data", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(df)} patients")
            st.dataframe(df.head())
            
            if st.button("Run Batch Predictions"):
                with st.spinner("Processing..."):
                    df_processed = df.copy()
                    df_encoded = pd.get_dummies(df_processed, drop_first=True)
                    # Align features
                    if 'feature_names' in models:
                        for col in models['feature_names']:
                            if col not in df_encoded.columns:
                                df_encoded[col] = 0
                        df_encoded = df_encoded[models['feature_names']]
                    # Scale
                    if 'scaler' in models:
                        X_scaled = models['scaler'].transform(df_encoded)
                    else:
                        X_scaled = df_encoded.values
                    
                    X_processed = X_scaled
                    
                    results = []
                    for i in range(len(df)):
                        prob, los_pred = make_prediction(models[model_choice], X_processed[i:i+1], model_choice)
                        if prob is not None:
                            risk_category, _ = get_risk_category(prob)
                            results.append({
                                'Patient_ID': i+1,
                                'Readmission_Probability': f"{prob*100:.1f}%",
                                'Risk_Category': risk_category,
                                'Predicted_LOS': f"{los_pred:.1f}" if los_pred else "N/A"
                            })
                    
                    results_df = pd.DataFrame(results)
                    st.success(f"✅ Processed {len(results)} patients")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
    
    # Tab 3: About
    with tab3:
        st.header("ℹ️ About This Application")
        st.markdown("""
        ### Diabetes Readmission Prediction System
        
        **Purpose:** Predict 30-day hospital readmission risk for diabetes patients
        
        **Available Models:**
        - **Gradient Boosting** (AUC: 0.638) - Best overall performance
        - **XGBoost** (AUC: 0.632) - Fast and accurate
        - **Logistic Regression** (AUC: 0.611) - Interpretable baseline
        """)
        
        if PYTORCH_AVAILABLE:
            st.markdown("""
        - **MultiTask MLP** (AUC: 0.630, R²: 0.768) - Deep learning, predicts LOS too
        - **Transformer** (AUC: 0.630, R²: 0.757) - Attention-based model
            """)
        
        st.markdown("""
        **Risk Categories:**
        - 🔴 **High Risk** (≥40%): Intensive intervention needed
        - 🟡 **Medium Risk** (20-40%): Standard monitoring
        - 🟢 **Low Risk** (<20%): Routine care
        
        **Note:** For research purposes only. Not for clinical use without validation.
        """)
        
        if not PYTORCH_AVAILABLE:
            st.info("""
            **To enable deep learning models:**
            ```bash
            pip install --upgrade typing_extensions
            pip install torch
            ```
            Then restart the app.
            """)


if __name__ == "__main__":
    main()