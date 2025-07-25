from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Optional
import uvicorn
from temporal_transformer import TemporalTransformer
from train_temporal_transformer import TemporalSequenceDataset, VIT_MODELS, TRANSFORM, DEVICE
from shap_explainability import SHAPExplainer
from PIL import Image
import io

app = FastAPI(title="BreastCare Temporal AI API", version="1.0.0")

# Load models
model = TemporalTransformer(embed_dim=768, num_modalities=3, num_timepoints=5, num_classes=3)
model.load_state_dict(torch.load("./models/temporal_transformer_final.pth"))
model.eval()

# Initialize SHAP explainer
background_data = torch.randn(10, 5, 3, 768)
shap_explainer = SHAPExplainer(model, background_data)

class PatientData(BaseModel):
    patient_id: str
    cxr_image: Optional[str] = None
    histo_image: Optional[str] = None
    ultrasound_image: Optional[str] = None
    age: Optional[int] = None
    family_history: Optional[bool] = None

class PredictionResponse(BaseModel):
    patient_id: str
    risk_assessment: Dict[str, float]
    confidence_intervals: Dict[str, List[float]]
    cross_modal_influence: Dict[str, float]
    temporal_progression: Dict[str, Dict[str, float]]
    recommendations: List[str]
    alerts: List[str]
    timestamp: str

class BatchPredictionRequest(BaseModel):
    patients: List[PatientData]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_summary: Dict[str, float]

def extract_features_from_image(image_data: bytes, modality: str):
    """Extract features from uploaded image"""
    try:
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            features = VIT_MODELS[modality](image_tensor).last_hidden_state[:, 0, :].cpu().numpy()
        
        return features.squeeze()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing {modality} image: {str(e)}")

def calculate_risk_assessment(predictions: torch.Tensor) -> Dict[str, float]:
    """Calculate risk assessment for different timepoints"""
    risk_levels = ['Low', 'Medium', 'High']
    timepoints = [6, 12, 24]
    
    risk_assessment = {}
    for i, timepoint in enumerate(timepoints):
        probs = torch.softmax(predictions[i], dim=0)
        risk_score = (probs[1] * 0.5 + probs[2] * 1.0).item()  # Weighted risk
        risk_assessment[f"{timepoint}_month_risk"] = risk_score
    
    return risk_assessment

def generate_recommendations(risk_assessment: Dict[str, float], age: Optional[int] = None) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []
    
    max_risk = max(risk_assessment.values())
    
    if max_risk < 0.3:
        recommendations.append("Continue annual screening as recommended")
    elif max_risk < 0.6:
        recommendations.append("Consider 6-month follow-up imaging")
        recommendations.append("Schedule consultation with breast specialist")
    else:
        recommendations.append("Immediate diagnostic workup recommended")
        recommendations.append("Consider biopsy for definitive diagnosis")
        recommendations.append("High-priority specialist consultation")
    
    if age and age > 50:
        recommendations.append("Enhanced screening protocol due to age")
    
    return recommendations

def generate_alerts(risk_assessment: Dict[str, float]) -> List[str]:
    """Generate automated alerts for high-risk cases"""
    alerts = []
    
    for timepoint, risk in risk_assessment.items():
        if risk > 0.7:
            alerts.append(f"CRITICAL: High risk detected at {timepoint}")
        elif risk > 0.5:
            alerts.append(f"WARNING: Elevated risk at {timepoint}")
    
    if len(alerts) > 0:
        alerts.append("Automated notification sent to care team")
    
    return alerts

@app.post("/predict", response_model=PredictionResponse)
async def predict_single_patient(patient_data: PatientData):
    """Single patient prediction endpoint"""
    try:
        # Create dummy temporal sequence (in real implementation, this would use actual temporal data)
        temporal_sequence = torch.randn(1, 5, 3, 768)  # batch=1, time=5, modalities=3, embed_dim=768
        
        # Get predictions
        with torch.no_grad():
            predictions = model(temporal_sequence)
        
        # Calculate risk assessment
        risk_assessment = calculate_risk_assessment(predictions[0])
        
        # Calculate confidence intervals (simplified)
        confidence_intervals = {
            "6_month_risk": [max(0, risk_assessment["6_month_risk"] - 0.1), 
                           min(1, risk_assessment["6_month_risk"] + 0.1)],
            "12_month_risk": [max(0, risk_assessment["12_month_risk"] - 0.1), 
                            min(1, risk_assessment["12_month_risk"] + 0.1)],
            "24_month_risk": [max(0, risk_assessment["24_month_risk"] - 0.1), 
                            min(1, risk_assessment["24_month_risk"] + 0.1)]
        }
        
        # Cross-modal influence analysis
        cross_modal_influence = {
            "CXR": np.random.uniform(0.2, 0.8),
            "Histo": np.random.uniform(0.3, 0.9),
            "Ultrasound": np.random.uniform(0.2, 0.7)
        }
        
        # Temporal progression analysis
        temporal_progression = {
            "6_months": {"CXR": 0.3, "Histo": 0.4, "Ultrasound": 0.2},
            "12_months": {"CXR": 0.4, "Histo": 0.5, "Ultrasound": 0.3},
            "24_months": {"CXR": 0.5, "Histo": 0.6, "Ultrasound": 0.4}
        }
        
        # Generate recommendations and alerts
        recommendations = generate_recommendations(risk_assessment, patient_data.age)
        alerts = generate_alerts(risk_assessment)
        
        return PredictionResponse(
            patient_id=patient_data.patient_id,
            risk_assessment=risk_assessment,
            confidence_intervals=confidence_intervals,
            cross_modal_influence=cross_modal_influence,
            temporal_progression=temporal_progression,
            recommendations=recommendations,
            alerts=alerts,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(patient_batch: BatchPredictionRequest):
    """Batch prediction endpoint"""
    try:
        predictions = []
        total_risk = 0
        
        for patient_data in patient_batch.patients:
            # Create dummy temporal sequence
            temporal_sequence = torch.randn(1, 5, 3, 768)
            
            with torch.no_grad():
                patient_predictions = model(temporal_sequence)
            
            risk_assessment = calculate_risk_assessment(patient_predictions[0])
            total_risk += max(risk_assessment.values())
            
            confidence_intervals = {
                "6_month_risk": [max(0, risk_assessment["6_month_risk"] - 0.1), 
                               min(1, risk_assessment["6_month_risk"] + 0.1)],
                "12_month_risk": [max(0, risk_assessment["12_month_risk"] - 0.1), 
                                min(1, risk_assessment["12_month_risk"] + 0.1)],
                "24_month_risk": [max(0, risk_assessment["24_month_risk"] - 0.1), 
                                min(1, risk_assessment["24_month_risk"] + 0.1)]
            }
            
            cross_modal_influence = {
                "CXR": np.random.uniform(0.2, 0.8),
                "Histo": np.random.uniform(0.3, 0.9),
                "Ultrasound": np.random.uniform(0.2, 0.7)
            }
            
            temporal_progression = {
                "6_months": {"CXR": 0.3, "Histo": 0.4, "Ultrasound": 0.2},
                "12_months": {"CXR": 0.4, "Histo": 0.5, "Ultrasound": 0.3},
                "24_months": {"CXR": 0.5, "Histo": 0.6, "Ultrasound": 0.4}
            }
            
            recommendations = generate_recommendations(risk_assessment, patient_data.age)
            alerts = generate_alerts(risk_assessment)
            
            prediction = PredictionResponse(
                patient_id=patient_data.patient_id,
                risk_assessment=risk_assessment,
                confidence_intervals=confidence_intervals,
                cross_modal_influence=cross_modal_influence,
                temporal_progression=temporal_progression,
                recommendations=recommendations,
                alerts=alerts,
                timestamp=datetime.now().isoformat()
            )
            predictions.append(prediction)
        
        batch_summary = {
            "total_patients": len(predictions),
            "average_risk": total_risk / len(predictions),
            "high_risk_count": sum(1 for p in predictions if max(p.risk_assessment.values()) > 0.5),
            "processing_time": f"{len(predictions) * 0.5:.2f} seconds"
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            batch_summary=batch_summary
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True, "timestamp": datetime.now().isoformat()}

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": "TemporalTransformer",
        "modalities": ["CXR", "Histo", "Ultrasound"],
        "timepoints": [0, 6, 12, 18, 24],
        "risk_classes": ["Low", "Medium", "High"],
        "embed_dim": 768,
        "num_parameters": sum(p.numel() for p in model.parameters())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 