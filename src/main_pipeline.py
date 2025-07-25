#!/usr/bin/env python3
"""
BreastCare Temporal AI - Complete Pipeline
Orchestrates the entire system from data preparation to clinical deployment
"""

import os
import sys
import subprocess
import time
from datetime import datetime
import torch
import uvicorn
from multiprocessing import Process

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-500:])  # Last 500 chars
        else:
            print(f"❌ {description} failed")
            print("Error:", result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        "./models",
        "./logs",
        "./reports",
        "./data/processed",
        "./data/temporal_sequences"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        "torch", "transformers", "peft", "fastapi", "uvicorn",
        "shap", "opacus", "flwr", "scikit-learn", "pandas", "numpy"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Please install missing packages: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed")
    return True

def run_data_preparation():
    """Run data preparation pipeline"""
    scripts = [
        ("src/eda.py", "Exploratory Data Analysis"),
        ("src/temporal_sequence_prep.py", "Temporal Sequence Preparation")
    ]
    
    for script, description in scripts:
        if not run_script(script, description):
            return False
    
    return True

def run_model_training():
    """Run model training pipeline"""
    scripts = [
        ("src/train_vit_lora.py", "CXR ViT+LoRA Training"),
        ("src/train_histo_lora.py", "Histopathological ViT+LoRA Training"),
        ("src/train_ultrasound_lora.py", "Ultrasound ViT+LoRA Training"),
        ("src/train_temporal_transformer.py", "Temporal Transformer Training")
    ]
    
    for script, description in scripts:
        if not run_script(script, description):
            return False
    
    return True

def run_validation_testing():
    """Run validation and testing"""
    return run_script("src/validation_testing.py", "Model Validation and Testing")

def run_federated_learning():
    """Run federated learning simulation"""
    return run_script("src/federated_learning.py", "Federated Learning Simulation")

def run_shap_analysis():
    """Run SHAP explainability analysis"""
    return run_script("src/shap_explainability.py", "SHAP Explainability Analysis")

def run_clinical_integration():
    """Run clinical integration system"""
    return run_script("src/clinical_integration.py", "Clinical Integration System")

def start_fastapi_server():
    """Start FastAPI server in a separate process"""
    def run_server():
        uvicorn.run("src.fastapi_endpoint:app", host="0.0.0.0", port=8000, reload=False)
    
    server_process = Process(target=run_server)
    server_process.start()
    return server_process

def generate_system_report():
    """Generate comprehensive system report"""
    report = {
        "system_name": "BreastCare Temporal AI",
        "version": "1.0.0",
        "deployment_timestamp": datetime.now().isoformat(),
        "components": {
            "data_preparation": "✅ Completed",
            "model_training": "✅ Completed", 
            "validation_testing": "✅ Completed",
            "federated_learning": "✅ Completed",
            "shap_explainability": "✅ Completed",
            "clinical_integration": "✅ Completed",
            "fastapi_endpoint": "✅ Running on http://localhost:8000"
        },
        "models_trained": [
            "CXR ViT+LoRA",
            "Histopathological ViT+LoRA", 
            "Ultrasound ViT+LoRA",
            "Temporal Transformer",
            "Federated Model"
        ],
        "api_endpoints": [
            "POST /predict - Single patient prediction",
            "POST /predict/batch - Batch prediction",
            "GET /health - Health check",
            "GET /model/info - Model information"
        ],
        "clinical_features": [
            "Personalized screening optimization",
            "High-risk patient identification",
            "Treatment response prediction",
            "Automated alerts and referrals"
        ]
    }
    
    # Save report
    import json
    with open("./reports/system_deployment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 BREASTCARE TEMPORAL AI SYSTEM DEPLOYED SUCCESSFULLY!")
    print("="*60)
    print(f"📊 System Report: ./reports/system_deployment_report.json")
    print(f"🌐 API Endpoint: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    print(f"🔍 Health Check: http://localhost:8000/health")
    print("="*60)
    
    return report

def main():
    """Main pipeline orchestration"""
    print("🚀 Starting BreastCare Temporal AI Pipeline")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Run pipeline stages
    stages = [
        ("Data Preparation", run_data_preparation),
        ("Model Training", run_model_training),
        ("Validation & Testing", run_validation_testing),
        ("Federated Learning", run_federated_learning),
        ("SHAP Analysis", run_shap_analysis),
        ("Clinical Integration", run_clinical_integration)
    ]
    
    for stage_name, stage_func in stages:
        print(f"\n🔄 Starting {stage_name}...")
        if not stage_func():
            print(f"❌ {stage_name} failed. Stopping pipeline.")
            sys.exit(1)
        print(f"✅ {stage_name} completed successfully")
    
    # Start FastAPI server
    print("\n🌐 Starting FastAPI server...")
    server_process = start_fastapi_server()
    time.sleep(5)  # Wait for server to start
    
    # Generate final report
    report = generate_system_report()
    
    print("\n🎯 SYSTEM READY FOR CLINICAL USE!")
    print("\n📋 Next Steps:")
    print("1. Access API documentation at http://localhost:8000/docs")
    print("2. Test endpoints with sample patient data")
    print("3. Integrate with hospital systems")
    print("4. Monitor system performance and alerts")
    
    try:
        # Keep the server running
        server_process.join()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down system...")
        server_process.terminate()
        server_process.join()
        print("✅ System shutdown complete")

if __name__ == "__main__":
    main() 