import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from config import DATA_PATH, BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO

class PINNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders():
    df = pd.read_csv(DATA_PATH)
    
    # Handle missing values
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    # Sort to ensure sequential order for shifting
    df.sort_values(by=['algorithm', 'episode', 'intersection_id', 'step_number'], inplace=True)
    
    # Features we use for input:
    feature_cols = ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption', 'current_phase', 'phase_duration']
    target_cols = ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption']
    
    # Shift targets to get t+1
    for col in target_cols:
        df[f'next_{col}'] = df.groupby(['algorithm', 'episode', 'intersection_id'])[col].shift(-1)
        
    # Drop rows with NaN (the last step of each episode)
    df.dropna(inplace=True)
    
    X_raw = df[feature_cols].values
    y_raw = df[[f'next_{c}' for c in target_cols]].values
    
    # Normalization
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_scaled = x_scaler.fit_transform(X_raw)
    y_scaled = y_scaler.fit_transform(y_raw)
    
    # Train / Val / Test Split
    total_size = len(X_scaled)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    
    X_train = X_scaled[:train_size]
    y_train = y_scaled[:train_size]
    
    X_val = X_scaled[train_size:train_size+val_size]
    y_val = y_scaled[train_size:train_size+val_size]
    
    X_test = X_scaled[train_size+val_size:]
    y_test = y_scaled[train_size+val_size:]
    
    train_dataset = PINNDataset(X_train, y_train)
    val_dataset = PINNDataset(X_val, y_val)
    test_dataset = PINNDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, x_scaler, y_scaler, df
