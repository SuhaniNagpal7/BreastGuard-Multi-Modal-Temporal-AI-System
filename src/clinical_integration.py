import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class PatientProfile:
    patient_id: str
    age: int
    family_history: bool
    previous_biopsies: int
    risk_factors: List[str]
    current_risk_score: float
    last_screening_date: Optional[datetime] = None

@dataclass
class ScreeningRecommendation:
    patient_id: str
    recommended_interval: int  # months
    next_screening_date: datetime
    priority_level: str  # "Low", "Medium", "High", "Critical"
    reasoning: List[str]
    specialist_referral: bool

@dataclass
class TreatmentResponse:
    patient_id: str
    baseline_risk: float
    treatment_type: str
    response_score: float
    predicted_effectiveness: float
    follow_up_interval: int

class ClinicalIntegration:
    def __init__(self):
        self.patient_database = {}
        self.screening_protocols = {
            "low_risk": {"interval": 12, "modalities": ["CXR"]},
            "medium_risk": {"interval": 6, "modalities": ["CXR", "Ultrasound"]},
            "high_risk": {"interval": 3, "modalities": ["CXR", "Ultrasound", "Histo"]},
            "critical_risk": {"interval": 1, "modalities": ["CXR", "Ultrasound", "Histo"]}
        }
    
    def calculate_dynamic_screening_interval(self, patient: PatientProfile, 
                                           risk_assessment: Dict[str, float]) -> int:
        """Calculate personalized screening interval based on risk profile"""
        base_interval = 12  # months
        
        # Age-based adjustments
        if patient.age > 65:
            base_interval = max(6, base_interval - 2)
        elif patient.age < 40:
            base_interval = min(18, base_interval + 3)
        
        # Family history adjustment
        if patient.family_history:
            base_interval = max(3, base_interval - 4)
        
        # Previous biopsies adjustment
        if patient.previous_biopsies > 0:
            base_interval = max(3, base_interval - patient.previous_biopsies * 2)
        
        # Current risk score adjustment
        max_risk = max(risk_assessment.values())
        if max_risk > 0.7:
            base_interval = 1
        elif max_risk > 0.5:
            base_interval = 3
        elif max_risk > 0.3:
            base_interval = 6
        
        return base_interval
    
    def identify_high_risk_patients(self, patient_data: List[PatientProfile], 
                                  risk_threshold: float = 0.5) -> List[PatientProfile]:
        """Identify high-risk patients requiring immediate attention"""
        high_risk_patients = []
        
        for patient in patient_data:
            if patient.current_risk_score > risk_threshold:
                high_risk_patients.append(patient)
            elif patient.family_history and patient.current_risk_score > 0.3:
                high_risk_patients.append(patient)
            elif patient.previous_biopsies > 2:
                high_risk_patients.append(patient)
        
        # Sort by risk score (highest first)
        high_risk_patients.sort(key=lambda x: x.current_risk_score, reverse=True)
        
        return high_risk_patients
    
    def generate_screening_recommendations(self, patient: PatientProfile, 
                                         risk_assessment: Dict[str, float]) -> ScreeningRecommendation:
        """Generate personalized screening recommendations"""
        interval = self.calculate_dynamic_screening_interval(patient, risk_assessment)
        
        # Calculate next screening date
        if patient.last_screening_date:
            next_date = patient.last_screening_date + timedelta(days=interval*30)
        else:
            next_date = datetime.now() + timedelta(days=interval*30)
        
        # Determine priority level
        max_risk = max(risk_assessment.values())
        if max_risk > 0.7:
            priority = "Critical"
            specialist_referral = True
        elif max_risk > 0.5:
            priority = "High"
            specialist_referral = True
        elif max_risk > 0.3:
            priority = "Medium"
            specialist_referral = False
        else:
            priority = "Low"
            specialist_referral = False
        
        # Generate reasoning
        reasoning = []
        if patient.family_history:
            reasoning.append("Family history of breast cancer")
        if patient.previous_biopsies > 0:
            reasoning.append(f"Previous biopsies: {patient.previous_biopsies}")
        if max_risk > 0.5:
            reasoning.append(f"Elevated risk score: {max_risk:.2f}")
        if patient.age > 65:
            reasoning.append("Age-related risk factors")
        
        return ScreeningRecommendation(
            patient_id=patient.patient_id,
            recommended_interval=interval,
            next_screening_date=next_date,
            priority_level=priority,
            reasoning=reasoning,
            specialist_referral=specialist_referral
        )
    
    def predict_treatment_response(self, patient: PatientProfile, 
                                 baseline_imaging: Dict[str, float],
                                 treatment_type: str) -> TreatmentResponse:
        """Predict treatment response based on baseline imaging"""
        # Simulate treatment response prediction
        baseline_risk = patient.current_risk_score
        
        # Treatment effectiveness varies by type
        effectiveness_map = {
            "surgery": 0.8,
            "chemotherapy": 0.6,
            "radiation": 0.7,
            "hormone_therapy": 0.5,
            "targeted_therapy": 0.75
        }
        
        base_effectiveness = effectiveness_map.get(treatment_type, 0.5)
        
        # Adjust based on patient factors
        if patient.age < 50:
            base_effectiveness *= 1.1  # Better response in younger patients
        elif patient.age > 70:
            base_effectiveness *= 0.9  # Reduced response in elderly
        
        if patient.family_history:
            base_effectiveness *= 0.95  # Slightly reduced response with family history
        
        # Calculate response score
        response_score = baseline_risk * (1 - base_effectiveness)
        
        # Determine follow-up interval
        if response_score < 0.2:
            follow_up = 6  # months
        elif response_score < 0.4:
            follow_up = 3  # months
        else:
            follow_up = 1  # month
        
        return TreatmentResponse(
            patient_id=patient.patient_id,
            baseline_risk=baseline_risk,
            treatment_type=treatment_type,
            response_score=response_score,
            predicted_effectiveness=base_effectiveness,
            follow_up_interval=follow_up
        )
    
    def generate_automated_alerts(self, patients: List[PatientProfile], 
                                risk_assessments: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Generate automated alerts for critical cases"""
        alerts = []
        
        for patient in patients:
            if patient.patient_id in risk_assessments:
                risk_assessment = risk_assessments[patient.patient_id]
                max_risk = max(risk_assessment.values())
                
                if max_risk > 0.7:
                    alerts.append({
                        "patient_id": patient.patient_id,
                        "alert_type": "CRITICAL",
                        "message": f"Critical risk detected: {max_risk:.2f}",
                        "action_required": "Immediate specialist consultation",
                        "timestamp": datetime.now().isoformat()
                    })
                elif max_risk > 0.5:
                    alerts.append({
                        "patient_id": patient.patient_id,
                        "alert_type": "HIGH_RISK",
                        "message": f"High risk detected: {max_risk:.2f}",
                        "action_required": "Accelerated screening protocol",
                        "timestamp": datetime.now().isoformat()
                    })
                
                # Check for rapid progression
                if len(risk_assessment) >= 2:
                    risk_values = list(risk_assessment.values())
                    progression_rate = (risk_values[-1] - risk_values[0]) / len(risk_values)
                    if progression_rate > 0.1:
                        alerts.append({
                            "patient_id": patient.patient_id,
                            "alert_type": "RAPID_PROGRESSION",
                            "message": f"Rapid progression detected: {progression_rate:.2f}",
                            "action_required": "Immediate diagnostic workup",
                            "timestamp": datetime.now().isoformat()
                        })
        
        return alerts
    
    def create_patient_cohort_report(self, patients: List[PatientProfile], 
                                   recommendations: List[ScreeningRecommendation]) -> Dict:
        """Create comprehensive cohort report"""
        total_patients = len(patients)
        high_risk_count = len([p for p in patients if p.current_risk_score > 0.5])
        avg_risk = np.mean([p.current_risk_score for p in patients])
        
        interval_distribution = {}
        for rec in recommendations:
            interval = rec.recommended_interval
            interval_distribution[interval] = interval_distribution.get(interval, 0) + 1
        
        return {
            "cohort_summary": {
                "total_patients": total_patients,
                "high_risk_patients": high_risk_count,
                "average_risk_score": avg_risk,
                "high_risk_percentage": (high_risk_count / total_patients) * 100
            },
            "screening_distribution": interval_distribution,
            "priority_distribution": {
                "Critical": len([r for r in recommendations if r.priority_level == "Critical"]),
                "High": len([r for r in recommendations if r.priority_level == "High"]),
                "Medium": len([r for r in recommendations if r.priority_level == "Medium"]),
                "Low": len([r for r in recommendations if r.priority_level == "Low"])
            },
            "specialist_referrals": len([r for r in recommendations if r.specialist_referral]),
            "generated_at": datetime.now().isoformat()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize clinical integration system
    clinical_system = ClinicalIntegration()
    
    # Create sample patient data
    patients = [
        PatientProfile("P001", 45, True, 1, ["family_history"], 0.6),
        PatientProfile("P002", 62, False, 0, ["age"], 0.3),
        PatientProfile("P003", 38, True, 2, ["family_history", "previous_biopsies"], 0.8)
    ]
    
    # Sample risk assessments
    risk_assessments = {
        "P001": {"6_month_risk": 0.6, "12_month_risk": 0.7, "24_month_risk": 0.8},
        "P002": {"6_month_risk": 0.3, "12_month_risk": 0.4, "24_month_risk": 0.5},
        "P003": {"6_month_risk": 0.8, "12_month_risk": 0.9, "24_month_risk": 0.95}
    }
    
    # Generate recommendations
    recommendations = []
    for patient in patients:
        if patient.patient_id in risk_assessments:
            rec = clinical_system.generate_screening_recommendations(patient, risk_assessments[patient.patient_id])
            recommendations.append(rec)
    
    # Identify high-risk patients
    high_risk = clinical_system.identify_high_risk_patients(patients)
    
    # Generate alerts
    alerts = clinical_system.generate_automated_alerts(patients, risk_assessments)
    
    # Create cohort report
    cohort_report = clinical_system.create_patient_cohort_report(patients, recommendations)
    
    print("=== Clinical Integration System Test Results ===")
    print(f"High-risk patients identified: {len(high_risk)}")
    print(f"Alerts generated: {len(alerts)}")
    print(f"Cohort report created for {cohort_report['cohort_summary']['total_patients']} patients")
    
    # Save results
    with open("clinical_integration_results.json", "w") as f:
        json.dump({
            "recommendations": [vars(r) for r in recommendations],
            "alerts": alerts,
            "cohort_report": cohort_report
        }, f, indent=2, default=str)
    
    print("Results saved to clinical_integration_results.json") 