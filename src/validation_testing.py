import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import numpy as np
import os
from temporal_transformer import TemporalTransformer
from train_temporal_transformer import TemporalSequenceDataset, VIT_MODELS, PROCESSOR, TRANSFORM, DEVICE, EMBED_DIM, NUM_MODALITIES, NUM_TIMEPOINTS, NUM_CLASSES
from torch.utils.data import DataLoader, Subset

def calculate_metrics(y_true, y_pred):
    """Calculate accuracy, precision, recall, F1 for each class"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def validate_model(model, dataloader, criterion, device):
    """Validate model and return metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seqs, labels in dataloader:
            seqs, labels = seqs.to(device), labels.to(device)
            outputs = model(seqs)
            outputs = outputs.view(-1, NUM_CLASSES)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels_np)
    
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader)
    return metrics

def train_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs, device, patience=5):
    """Train model with validation and early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            outputs = outputs.view(-1, NUM_CLASSES)
            labels = labels.view(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        val_metrics = validate_model(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model

# Main validation/testing pipeline
if __name__ == "__main__":
    # Load dataset
    dataset = TemporalSequenceDataset("./synthetic_temporal_sequences.csv")
    
    # Split into train/val/test
    train_idx, temp_idx = train_test_split(range(len(dataset)), test_size=0.3, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Initialize model
    model = TemporalTransformer(embed_dim=EMBED_DIM, num_modalities=NUM_MODALITIES, 
                               num_timepoints=NUM_TIMEPOINTS, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Train with validation
    model = train_with_validation(model, train_loader, val_loader, optimizer, criterion, epochs=5, device=DEVICE)
    
    # Test final model
    test_metrics = validate_model(model, test_loader, criterion, DEVICE)
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "./models/temporal_transformer_final.pth")
    print("Model saved to ./models/temporal_transformer_final.pth") 