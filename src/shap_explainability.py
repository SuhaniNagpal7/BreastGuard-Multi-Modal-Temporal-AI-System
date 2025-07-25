import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from temporal_transformer import TemporalTransformer
from train_temporal_transformer import TemporalSequenceDataset, VIT_MODELS, TRANSFORM, DEVICE
from PIL import Image
import pandas as pd

class SHAPExplainer:
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data
        self.explainer = shap.DeepExplainer(model, background_data)
    
    def explain_prediction(self, input_data):
        """Generate SHAP values for a single prediction"""
        shap_values = self.explainer.shap_values(input_data)
        return shap_values
    
    def explain_cross_modal_influence(self, input_data):
        """Analyze how findings in one modality influence others"""
        shap_values = self.explainer.shap_values(input_data)
        
        # Analyze cross-modal attention weights
        cross_modal_influence = {}
        modalities = ['CXR', 'Histo', 'Ultrasound']
        
        for i, modality in enumerate(modalities):
            influence_scores = np.mean(np.abs(shap_values[0][:, i, :]), axis=1)
            cross_modal_influence[modality] = influence_scores
        
        return cross_modal_influence
    
    def visualize_cross_modal_influence(self, cross_modal_influence):
        """Visualize cross-modal influence matrix"""
        modalities = list(cross_modal_influence.keys())
        influence_matrix = np.array([cross_modal_influence[mod] for mod in modalities])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(influence_matrix, annot=True, cmap='Reds', 
                   xticklabels=modalities, yticklabels=modalities)
        plt.title('Cross-Modal Influence Matrix')
        plt.xlabel('Target Modality')
        plt.ylabel('Source Modality')
        plt.tight_layout()
        plt.savefig('./cross_modal_influence.png')
        plt.show()
    
    def generate_temporal_explanations(self, input_data, timepoints=[0, 6, 12, 18, 24]):
        """Generate temporal explanation sequences"""
        shap_values = self.explainer.shap_values(input_data)
        
        temporal_explanations = {}
        modalities = ['CXR', 'Histo', 'Ultrasound']
        
        for t, timepoint in enumerate(timepoints):
            time_explanations = {}
            for m, modality in enumerate(modalities):
                # Get SHAP values for this timepoint and modality
                modality_importance = np.mean(np.abs(shap_values[0][t, m, :]))
                time_explanations[modality] = modality_importance
            temporal_explanations[timepoint] = time_explanations
        
        return temporal_explanations
    
    def visualize_temporal_sequence(self, temporal_explanations):
        """Visualize temporal explanation sequence"""
        timepoints = list(temporal_explanations.keys())
        modalities = list(temporal_explanations[timepoints[0]].keys())
        
        fig, axes = plt.subplots(1, len(modalities), figsize=(15, 5))
        
        for i, modality in enumerate(modalities):
            values = [temporal_explanations[t][modality] for t in timepoints]
            axes[i].plot(timepoints, values, marker='o', linewidth=2, markersize=8)
            axes[i].set_title(f'{modality} Temporal Progression')
            axes[i].set_xlabel('Time (months)')
            axes[i].set_ylabel('SHAP Importance')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./temporal_explanations.png')
        plt.show()
    
    def create_3d_anatomical_mapping(self, input_data):
        """Create 3D anatomical mapping of findings"""
        shap_values = self.explainer.shap_values(input_data)
        
        # Simulate 3D anatomical coordinates
        anatomical_coords = {
            'CXR': {'x': 0, 'y': 0, 'z': 0},
            'Histo': {'x': 1, 'y': 0, 'z': 0},
            'Ultrasound': {'x': 0.5, 'y': 1, 'z': 0}
        }
        
        # Create 3D visualization
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        modalities = ['CXR', 'Histo', 'Ultrasound']
        colors = ['red', 'blue', 'green']
        
        for i, modality in enumerate(modalities):
            coords = anatomical_coords[modality]
            importance = np.mean(np.abs(shap_values[0][:, i, :]))
            
            ax.scatter(coords['x'], coords['y'], coords['z'], 
                      c=colors[i], s=importance*1000, alpha=0.7, label=modality)
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.set_title('3D Anatomical Mapping of Cross-Modal Findings')
        ax.legend()
        
        plt.savefig('./3d_anatomical_mapping.png')
        plt.show()

def generate_comprehensive_report(explainer, input_data, patient_id):
    """Generate comprehensive SHAP analysis report"""
    print(f"=== SHAP Analysis Report for Patient {patient_id} ===\n")
    
    # Cross-modal influence analysis
    cross_modal_influence = explainer.explain_cross_modal_influence(input_data)
    print("Cross-Modal Influence Analysis:")
    for modality, influence in cross_modal_influence.items():
        print(f"  {modality}: {np.mean(influence):.4f}")
    
    # Temporal progression analysis
    temporal_explanations = explainer.generate_temporal_explanations(input_data)
    print("\nTemporal Progression Analysis:")
    for timepoint, explanations in temporal_explanations.items():
        print(f"  {timepoint} months:")
        for modality, importance in explanations.items():
            print(f"    {modality}: {importance:.4f}")
    
    # Generate visualizations
    explainer.visualize_cross_modal_influence(cross_modal_influence)
    explainer.visualize_temporal_sequence(temporal_explanations)
    explainer.create_3d_anatomical_mapping(input_data)
    
    print("\nVisualizations saved as:")
    print("  - cross_modal_influence.png")
    print("  - temporal_explanations.png")
    print("  - 3d_anatomical_mapping.png")

# Main SHAP analysis pipeline
if __name__ == "__main__":
    # Load trained model
    model = TemporalTransformer(embed_dim=768, num_modalities=3, 
                               num_timepoints=5, num_classes=3)
    model.load_state_dict(torch.load("./models/temporal_transformer_final.pth"))
    model.eval()
    
    # Create background data for SHAP
    background_data = torch.randn(10, 5, 3, 768)  # 10 background samples
    
    # Initialize SHAP explainer
    explainer = SHAPExplainer(model, background_data)
    
    # Load test data
    dataset = TemporalSequenceDataset("./synthetic_temporal_sequences.csv")
    test_sample, _ = dataset[0]  # Get first patient
    
    # Generate comprehensive report
    generate_comprehensive_report(explainer, test_sample.unsqueeze(0), "P001") 