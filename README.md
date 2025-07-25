# BreastGuard: Multi-Modal Temporal AI System 🏥

**Cross-Modal Temporal Progression Prediction with Federated Explainable AI**

Advanced AI system for breast cancer screening using multi-modal temporal transformers, federated learning, and comprehensive explainability.

## 🎯 Core Features

- **Multi-Modal Analysis**: CXR, Histopathology, Ultrasound
- **Temporal Prediction**: 6, 12, 24-month cancer risk forecasting
- **Cross-Modal Attention**: Findings influence predictions across modalities
- **Federated Learning**: Privacy-preserving training across hospitals
- **SHAP Explainability**: Interpretable predictions with visual explanations
- **Real-time API**: FastAPI endpoint for clinical deployment

## 🚀 Quick Start

### 1. Download Dataset
```bash
# Download from Kaggle
kaggle datasets download -d vishwamalani/breast-cancer-dataset
unzip breast-cancer-dataset.zip
```

### 2. Install & Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete system
python src/main_pipeline.py
```

### 3. Access API
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 System Components

- **Data Prep**: EDA, temporal sequence generation
- **Model Training**: ViT+LoRA for each modality + temporal transformer
- **Federated Learning**: Multi-hospital simulation with privacy
- **Explainability**: SHAP analysis and visualizations
- **Clinical API**: Real-time inference and reporting

## 🔧 API Usage

```python
import requests

# Single prediction
response = requests.post("http://localhost:8000/predict", json={
    "patient_id": "P001",
    "age": 45,
    "family_history": True
})

# Batch prediction
response = requests.post("http://localhost:8000/predict/batch", json={
    "patients": [{"patient_id": "P001", "age": 45}]
})
```

## 🏥 Clinical Features

- Personalized screening intervals
- High-risk patient identification
- Treatment response prediction
- Automated alerts and referrals

## 🔒 Privacy & Security

- Federated learning (no data sharing)
- Differential privacy guarantees
- HIPAA-compliant processing

## 📁 Project Structure

```
src/
├── data_loader.py              # Multi-modal dataset loading
├── eda.py                      # Exploratory data analysis
├── temporal_sequence_prep.py   # Synthetic temporal sequences
├── train_vit_lora.py          # CXR ViT+LoRA training
├── train_histo_lora.py        # Histopathological training
├── train_ultrasound_lora.py   # Ultrasound training
├── temporal_transformer.py    # Cross-modal temporal transformer
├── train_temporal_transformer.py # Temporal model training
├── validation_testing.py      # Model validation
├── federated_learning.py      # Federated learning simulation
├── shap_explainability.py     # SHAP analysis
├── clinical_integration.py    # Clinical features
├── fastapi_endpoint.py        # Production API
└── main_pipeline.py           # Complete system orchestration
```

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: Vision Transformers (ViT)
- **PEFT**: LoRA fine-tuning
- **FastAPI**: Production API
- **SHAP**: Explainability
- **Opacus**: Differential privacy
- **Flower**: Federated learning

## 📈 Performance

- **Inference Time**: < 30 seconds
- **Modalities**: CXR + Histo + Ultrasound
- **Temporal Prediction**: 6, 12, 24 months
- **Privacy**: HIPAA-compliant federated learning

---

**BreastGuard** - Revolutionizing breast cancer screening through advanced AI. 🏥✨ 