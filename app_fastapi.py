"""
Diabetes Readmission Prediction API
FastAPI Backend - Production Ready

Usage:
    uvicorn app_fastapi:app --reload
    
API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from io import StringIO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Diabetes Readmission Prediction API",
    description="Predict 30-day readmission risk and length of stay for diabetes patients",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== MODEL DEFINITIONS ====================

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


# ==================== PYDANTIC MODELS ====================

class PatientInput(BaseModel):
    """Single patient input schema"""
    race: str = Field(..., example="Caucasian")
    gender: str = Field(..., example="Male")
    age: str = Field(..., example="[50-60)")
    admission_type_id: int = Field(..., ge=1, le=8, example=1)
    time_in_hospital: int = Field(..., ge=1, le=14, example=3)
    num_lab_procedures: int = Field(..., ge=0, example=40)
    num_procedures: int = Field(..., ge=0, le=6, example=1)
    num_medications: int = Field(..., ge=1, le=50, example=15)
    number_outpatient: int = Field(..., ge=0, example=0)
    number_emergency: int = Field(..., ge=0, example=0)
    number_inpatient: int = Field(..., ge=0, example=0)
    diag_1: str = Field(..., example="250.00")
    diag_2: Optional[str] = Field(None, example="")
    diag_3: Optional[str] = Field(None, example="")
    # Add other required fields as needed


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    readmission_probability: float = Field(..., ge=0, le=1)
    readmission_risk_category: str
    predicted_los_days: float = Field(..., gt=0)
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    top_risk_factors: Optional[List[Dict[str, float]]] = None
    recommendations: Optional[List[str]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse]
    summary: Dict[str, int]


class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


# ==================== FEATURE ENGINEERING ====================

def charlson(row):
    score = 0
    for c in ["diag_1", "diag_2", "diag_3"]:
        v = str(row.get(c, "")) if pd.notna(row.get(c)) else ""
        if v.startswith("250"): score += 1
        if v.startswith(("410","411","412","413","414")): score += 1
        if v.startswith("428"): score += 1
        if v.startswith(("582","583","585","586")): score += 2
        if v.startswith(("490","491","492","493","494","495","496")): score += 1
    return min(score, 10)

def lace(row):
    los = row.get("time_in_hospital", 0.0)
    L = 7 if los >= 14 else min(int(los), 7)
    A = 3 if row.get("admission_type_id", 3) in [1, 2] else 0
    C = (5 if row["Charlson_Index"] >= 4 else 3 if row["Charlson_Index"] == 3 else 
         2 if row["Charlson_Index"] == 2 else 1 if row["Charlson_Index"] == 1 else 0)
    E = 4 if row.get("number_emergency", 0) >= 4 else int(row.get("number_emergency", 0))
    return L + A + C + E

def hospital_score(row):
    s = 0
    if row.get("num_procedures", 0) > 0: s += 1
    if row.get("admission_type_id", 3) in [1, 2]: s += 1
    n_inp = row.get("number_inpatient", 0)
    if n_inp >= 5: s += 5
    elif n_inp >= 2: s += 2
    if row.get("time_in_hospital", 0) >= 5: s += 2
    return s

def engineer_features(df):
    df = df.copy()
    df["Charlson_Index"] = df.apply(charlson, axis=1)
    df["LACE_Index"] = df.apply(lace, axis=1)
    df["HOSPITAL_Score"] = df.apply(hospital_score, axis=1)
    df["Days_Since_Last_Discharge"] = df["number_inpatient"].apply(
        lambda n: 365 if n == 0 else int(365 / (n + 1))
    )
    df["Polypharmacy_Count"] = df["num_medications"]
    df["Recent_Hosp_Count"] = df["number_inpatient"]
    return df


# ==================== MODEL LOADING ====================

class ModelManager:
    """Singleton model manager"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self):
        """Load model and preprocessing objects"""
        if self.initialized:
            return
        
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load model
            self.model = MultiTaskMLP(d_in=2335, hidden=[128, 64], dropout=0.3)
            
            try:
                state_dict = torch.load("best_multitask_model.pth", map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("✅ Model loaded successfully")
            except FileNotFoundError:
                logger.warning("⚠️ Model file not found. Using untrained model.")
            
            self.model.eval()
            self.model.to(self.device)
            
            # Load preprocessing
            try:
                with open("scaler.pkl", "rb") as f:
                    self.scaler = pickle.load(f)
                with open("feature_names.pkl", "rb") as f:
                    self.feature_names = pickle.load(f)
                logger.info("✅ Preprocessing objects loaded")
            except FileNotFoundError:
                logger.warning("⚠️ Preprocessing files not found")
                from sklearn.preprocessing import StandardScaler
                self.scaler = StandardScaler()
                self.feature_names = None
            
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def predict(self, df):
        """Make predictions"""
        try:
            # Feature engineering
            df_processed = engineer_features(df)
            df_processed = df_processed.drop(columns=["readmitted", "time_in_hospital"], errors='ignore')
            
            # Encode
            df_encoded = pd.get_dummies(df_processed, drop_first=True)
            
            # Align features
            if self.feature_names is not None:
                for col in self.feature_names:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                df_encoded = df_encoded[self.feature_names]
            
            # Scale
            X_scaled = self.scaler.transform(df_encoded)
            
            # Predict
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                readmit_prob, los_pred = self.model(X_tensor)
                readmit_prob = readmit_prob.cpu().numpy()
                los_pred = los_pred.cpu().numpy()
            
            return readmit_prob, los_pred
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# Initialize model manager
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting up...")
    model_manager.initialize()


# ==================== API ENDPOINTS ====================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Diabetes Readmission Prediction API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthCheck, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        model_loaded=model_manager.initialized,
        device=str(model_manager.device)
    )


def get_risk_category(prob: float) -> str:
    """Determine risk category"""
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def get_recommendations(prob: float, los: float) -> List[str]:
    """Generate clinical recommendations"""
    recommendations = []
    
    if prob >= 0.7:
        recommendations.extend([
            "Schedule follow-up within 7 days",
            "Implement intensive discharge planning",
            "Consider care coordination services",
            "Review medication adherence carefully"
        ])
    elif prob >= 0.4:
        recommendations.extend([
            "Schedule follow-up within 14 days",
            "Provide discharge education materials",
            "Ensure medication reconciliation"
        ])
    else:
        recommendations.extend([
            "Standard discharge process",
            "Routine follow-up as needed"
        ])
    
    if los > 7:
        recommendations.append("Extended LOS - Review discharge barriers")
    
    return recommendations


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(patient: PatientInput):
    """
    Predict readmission risk for a single patient
    
    Returns:
    - Readmission probability (0-1)
    - Risk category (LOW/MEDIUM/HIGH)
    - Predicted length of stay
    - Clinical recommendations
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([patient.dict()])
        
        # Predict
        readmit_prob, los_pred = model_manager.predict(df)
        
        prob = float(readmit_prob[0])
        los = float(los_pred[0])
        risk_cat = get_risk_category(prob)
        recs = get_recommendations(prob, los)
        
        return PredictionResponse(
            readmission_probability=prob,
            readmission_risk_category=risk_cat,
            predicted_los_days=los,
            recommendations=recs
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Predict readmission risk for multiple patients from CSV
    
    Upload a CSV file with patient data
    """
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))
        
        logger.info(f"Processing {len(df)} patients")
        
        # Predict
        readmit_probs, los_preds = model_manager.predict(df)
        
        # Build responses
        predictions = []
        summary = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        
        for i in range(len(df)):
            prob = float(readmit_probs[i])
            los = float(los_preds[i])
            risk_cat = get_risk_category(prob)
            recs = get_recommendations(prob, los)
            
            summary[risk_cat] += 1
            
            predictions.append(
                PredictionResponse(
                    readmission_probability=prob,
                    readmission_risk_category=risk_cat,
                    predicted_los_days=los,
                    recommendations=recs
                )
            )
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get model information"""
    return {
        "model_type": "Multi-Task MLP",
        "architecture": "Deep Neural Network",
        "input_features": len(model_manager.feature_names) if model_manager.feature_names else "Unknown",
        "outputs": ["readmission_probability", "length_of_stay"],
        "training_data": "UCI Diabetes 130-US Hospitals",
        "performance": {
            "auc": 0.68,
            "mae_los": 2.1
        }
    }


# ==================== ERROR HANDLERS ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
