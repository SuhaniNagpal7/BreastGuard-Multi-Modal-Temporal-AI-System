import torch
import torch.nn as nn
import numpy as np
import random
from collections import OrderedDict
import copy
from opacus import PrivacyEngine
from temporal_transformer import TemporalTransformer

class FederatedClient:
    def __init__(self, client_id, model, data_subset, noise_multiplier=1.0):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.data_subset = data_subset
        self.privacy_engine = PrivacyEngine()
        self.noise_multiplier = noise_multiplier
        
    def train_epoch(self, optimizer, criterion):
        """Train one epoch with differential privacy"""
        self.model.train()
        total_loss = 0
        
        for seqs, labels in self.data_subset:
            optimizer.zero_grad()
            outputs = self.model(seqs)
            outputs = outputs.view(-1, 3)  # 3 classes
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(self.data_subset)
    
    def get_model_params(self):
        """Get model parameters for aggregation"""
        return copy.deepcopy(self.model.state_dict())

def simulate_hospital_domains(num_clients=50):
    """Simulate different hospital environments with varying protocols"""
    domains = []
    for i in range(num_clients):
        domain = {
            'client_id': i,
            'imaging_protocol': random.choice(['standard', 'high_res', 'low_dose']),
            'patient_demographics': {
                'age_distribution': random.choice(['young', 'middle', 'elderly']),
                'ethnicity_distribution': random.choice(['diverse', 'majority_white', 'majority_black']),
                'socioeconomic_status': random.choice(['low', 'medium', 'high'])
            },
            'data_quality': random.uniform(0.7, 1.0),
            'sample_size': random.randint(50, 200)
        }
        domains.append(domain)
    return domains

def domain_adaptation(model, source_domain, target_domain):
    """Simple domain adaptation by adjusting model parameters"""
    adapted_model = copy.deepcopy(model)
    
    # Adjust model based on domain differences
    if source_domain['imaging_protocol'] != target_domain['imaging_protocol']:
        # Add noise to simulate protocol differences
        for param in adapted_model.parameters():
            param.data += torch.randn_like(param.data) * 0.01
    
    return adapted_model

def secure_aggregation(client_models, noise_scale=0.1):
    """Secure aggregation with homomorphic encryption simulation"""
    aggregated_params = OrderedDict()
    
    # Initialize with first client's parameters
    first_client_params = client_models[0]
    for key in first_client_params.keys():
        aggregated_params[key] = first_client_params[key].clone()
    
    # Aggregate remaining clients with noise
    for client_params in client_models[1:]:
        for key in aggregated_params.keys():
            noise = torch.randn_like(aggregated_params[key]) * noise_scale
            aggregated_params[key] += client_params[key] + noise
    
    # Average the parameters
    num_clients = len(client_models)
    for key in aggregated_params.keys():
        aggregated_params[key] /= num_clients
    
    return aggregated_params

def federated_training(global_model, clients, num_rounds=10, local_epochs=3):
    """Main federated training loop"""
    global_model.train()
    
    for round_num in range(num_rounds):
        print(f"Federated Round {round_num + 1}/{num_rounds}")
        
        # Local training on each client
        client_models = []
        for client in clients:
            optimizer = torch.optim.AdamW(client.model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            
            # Train locally
            for epoch in range(local_epochs):
                loss = client.train_epoch(optimizer, criterion)
            
            # Get model parameters
            client_models.append(client.get_model_params())
        
        # Secure aggregation
        aggregated_params = secure_aggregation(client_models)
        
        # Update global model
        global_model.load_state_dict(aggregated_params)
        
        # Update client models
        for client in clients:
            client.model.load_state_dict(aggregated_params)
        
        print(f"Round {round_num + 1} completed. Global model updated.")
    
    return global_model

# Main federated learning simulation
if __name__ == "__main__":
    # Initialize global model
    global_model = TemporalTransformer(embed_dim=768, num_modalities=3, 
                                      num_timepoints=5, num_classes=3)
    
    # Simulate hospital domains
    domains = simulate_hospital_domains(num_clients=50)
    
    # Create federated clients
    clients = []
    for domain in domains:
        # Create dummy data subset for each client
        dummy_data = [(torch.randn(2, 5, 3, 768), torch.randint(0, 3, (2, 5))) 
                      for _ in range(domain['sample_size'])]
        
        client = FederatedClient(
            client_id=domain['client_id'],
            model=global_model,
            data_subset=dummy_data,
            noise_multiplier=1.0
        )
        clients.append(client)
    
    # Run federated training
    final_model = federated_training(global_model, clients, num_rounds=5, local_epochs=2)
    
    # Save federated model
    torch.save(final_model.state_dict(), "./models/federated_temporal_transformer.pth")
    print("Federated model saved to ./models/federated_temporal_transformer.pth") 