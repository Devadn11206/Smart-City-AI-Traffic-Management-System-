import pandas as pd
import numpy as np
import torch
import os
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

def get_dataloaders(action_aware=True):
    workspace_dir = "c:\\Users\\ddnan\\intenrship_IIIT"
    if action_aware:
        data_path = os.path.join(workspace_dir, 'benchmark_research', 'historical_data', 'traffic_history_action.csv')
        feature_cols = ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption', 'current_phase', 'phase_duration', 'selected_action', 'action_queue_length', 'action_vehicle_count']
    else:
        data_path = os.path.join(workspace_dir, 'benchmark_research', 'historical_data', 'traffic_history_6x6.csv')
        feature_cols = ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption', 'current_phase', 'phase_duration']
        
    df = pd.read_csv(data_path)
    
    # Handle missing values
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    # Apply variance-stabilizing transform to throughput (square root)
    if action_aware:
        df['throughput'] = np.sqrt(df['throughput'].clip(lower=0))
    
    # Sort to ensure sequential order for shifting
    df.sort_values(by=['algorithm', 'episode', 'intersection_id', 'step_number'], inplace=True)
    
    target_cols = ['traffic_density', 'queue_length', 'throughput', 'avg_speed', 'fuel_consumption']
    
    # Shift targets to get t+1
    for col in target_cols:
        df[f'next_{col}'] = df.groupby(['algorithm', 'episode', 'intersection_id'])[col].shift(-1)
        
    # Drop rows with NaN (the last step of each episode)
    df.dropna(inplace=True)
    
    # Print and report the raw action distribution
    print("\n--- Action Distribution in Raw Dataset ---")
    action_counts = df['selected_action'].value_counts().sort_index()
    for act, count in action_counts.items():
        print(f"Action {act}: {count} samples ({count/len(df)*100:.2f}%)")
    print("-------------------------------------------\n")
    
    X_raw = df[feature_cols].values
    y_raw = df[[f'next_{c}' for c in target_cols]].values
    
    # Train / Val / Test Split
    total_size = len(X_raw)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    
    X_train_raw = X_raw[:train_size]
    y_train_raw = y_raw[:train_size]
    X_val_raw = X_raw[train_size:train_size+val_size]
    y_val_raw = y_raw[train_size:train_size+val_size]
    X_test_raw = X_raw[train_size+val_size:]
    y_test_raw = y_raw[train_size+val_size:]
    
    if action_aware:
        # Balance training set by oversampling minority classes (using raw integers)
        actions = X_train_raw[:, 7].astype(int)
        unique_actions, counts = np.unique(actions, return_counts=True)
        max_count = np.max(counts)
        
        print("--- Balancing Training Set (Oversampling) ---")
        print(f"Original Train counts: {dict(zip(unique_actions, counts))}")
        
        balanced_X = []
        balanced_y = []
        for action in unique_actions:
            idx = np.where(actions == action)[0]
            if len(idx) > 0:
                resampled_idx = np.random.choice(idx, size=max_count, replace=True)
                balanced_X.append(X_train_raw[resampled_idx])
                balanced_y.append(y_train_raw[resampled_idx])
                
        X_train_raw = np.vstack(balanced_X)
        y_train_raw = np.vstack(balanced_y)
        
        # Shuffle training set
        shuffle_idx = np.random.permutation(len(X_train_raw))
        X_train_raw = X_train_raw[shuffle_idx]
        y_train_raw = y_train_raw[shuffle_idx]
        
        balanced_actions = X_train_raw[:, 7].astype(int)
        _, balanced_counts = np.unique(balanced_actions, return_counts=True)
        print(f"Balanced Train counts: {dict(zip(unique_actions, balanced_counts))}")
        print("----------------------------------------------\n")
        
    # Scale variables
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X_train = x_scaler.fit_transform(X_train_raw)
    y_train = y_scaler.fit_transform(y_train_raw)
    
    X_val = x_scaler.transform(X_val_raw)
    y_val = y_scaler.transform(y_val_raw)
    
    X_test = x_scaler.transform(X_test_raw)
    y_test = y_scaler.transform(y_test_raw)
    
    train_dataset = PINNDataset(X_train, y_train)
    val_dataset = PINNDataset(X_val, y_val)
    test_dataset = PINNDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, x_scaler, y_scaler, df
