"""
Hyperparameter Tuning Script for NBA Playoff Predictor
Tests different combinations of dropout, learning rate, batch size, and weight decay
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau

# Add model path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Model', 'Future Prediction'))
from hybrid_model_future_pred import (
    load_player_stats, load_playoff_stats, preprocess_data,
    HybridNBAModel, NBADataset
)

def train_and_evaluate_config(
    train_dataset, val_dataset, test_dataset, meta_test,
    dropout_rate, learning_rate, batch_size, weight_decay,
    n_players, n_team_features, max_epochs=200, patience=10
):
    """Train and evaluate a single hyperparameter configuration"""
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = HybridNBAModel(
        n_players=n_players,
        n_team_features=n_team_features,
        dropout_rate=dropout_rate,
        use_attention=True
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    model.train()
    for epoch in range(max_epochs):
        # Training
        epoch_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['player_stats'], batch['team_features'])
            loss = criterion(outputs, batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(batch['player_stats'], batch['team_features'])
                loss = criterion(outputs, batch['target'])
                epoch_val_loss += loss.item()
        model.train()
        
        train_loss_avg = epoch_train_loss / len(train_loader)
        val_loss_avg = epoch_val_loss / len(val_loader)
        scheduler.step(val_loss_avg)
        
        # Early stopping
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    model.eval()
    test_predictions = []
    test_targets = []
    test_teams = []
    test_seasons_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['player_stats'], batch['team_features'])
            test_predictions.extend(outputs.numpy())
            test_targets.extend(batch['target'].numpy())
            for idx in batch['original_idx'].numpy():
                test_teams.append(meta_test.iloc[idx]['Team'])
                test_seasons_list.append(meta_test.iloc[idx]['Season'])
    
    # Calculate metrics
    test_df = pd.DataFrame({
        'Season': test_seasons_list,
        'Team': test_teams,
        'Predicted': test_predictions,
        'Actual': test_targets
    })
    
    test_df['Predicted_Rank'] = test_df.groupby('Season')['Predicted'].rank(method='dense', ascending=True).astype(int)
    test_df['Actual_Rank'] = test_df.groupby('Season')['Actual'].rank(method='dense', ascending=True).astype(int)
    
    overall_actual = test_df['Actual_Rank'].values
    overall_predicted = test_df['Predicted_Rank'].values
    spearman = spearmanr(overall_actual, overall_predicted)[0]
    kendall = kendalltau(overall_actual, overall_predicted)[0]
    mae = np.mean(np.abs(overall_actual - overall_predicted))
    
    return {
        'dropout_rate': dropout_rate,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'weight_decay': weight_decay,
        'spearman': spearman,
        'kendall_tau': kendall,
        'mae': mae,
        'best_val_loss': best_val_loss,
        'epochs_trained': epoch + 1
    }

def main():
    base_path = "Preprocessing/Preprocessed Data"
    output_dir = "Results"
    
    print("=" * 70)
    print("Hyperparameter Tuning for NBA Playoff Predictor")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    player_df = load_player_stats(base_path)
    playoff_df_historical = load_playoff_stats(base_path)
    
    historical_seasons = sorted([s for s in playoff_df_historical['Season'].unique() if s != '2024-25'])
    
    # 80/20 split
    np.random.seed(42)
    shuffled_seasons = historical_seasons.copy()
    np.random.shuffle(shuffled_seasons)
    n_train_seasons = int(0.8 * len(shuffled_seasons))
    train_seasons = sorted(shuffled_seasons[:n_train_seasons])
    test_seasons = sorted(shuffled_seasons[n_train_seasons:])
    
    playoff_df_train = playoff_df_historical[playoff_df_historical['Season'].isin(train_seasons)]
    playoff_df_test = playoff_df_historical[playoff_df_historical['Season'].isin(test_seasons)]
    
    # Preprocess
    print("2. Preprocessing data...")
    X_player_train, X_team_train, y_train, meta_train = preprocess_data(
        player_df, playoff_df_train, fit_scalers=False)
    X_player_test, X_team_test, y_test, meta_test = preprocess_data(
        player_df, playoff_df_test, fit_scalers=False)
    
    # Fit scalers
    player_scaler = StandardScaler()
    train_player_reshaped = X_player_train.reshape(-1, X_player_train.shape[-1])
    train_player_reshaped = np.nan_to_num(train_player_reshaped, nan=0)
    player_scaler.fit(train_player_reshaped)
    
    team_scaler = StandardScaler()
    train_team_clean = np.nan_to_num(X_team_train, nan=0)
    team_scaler.fit(train_team_clean)
    
    # Transform data
    X_player_train_scaled = player_scaler.transform(train_player_reshaped).reshape(X_player_train.shape)
    X_team_train_scaled = team_scaler.transform(train_team_clean)
    
    test_player_reshaped = X_player_test.reshape(-1, X_player_test.shape[-1])
    test_player_reshaped = np.nan_to_num(test_player_reshaped, nan=0)
    X_player_test_scaled = player_scaler.transform(test_player_reshaped).reshape(X_player_test.shape)
    
    test_team_clean = np.nan_to_num(X_team_test, nan=0)
    X_team_test_scaled = team_scaler.transform(test_team_clean)
    
    # Split train/val
    train_indices = np.arange(len(y_train))
    np.random.seed(42)
    np.random.shuffle(train_indices)
    n_train_samples = int(0.8 * len(train_indices))
    train_sample_indices = train_indices[:n_train_samples]
    val_sample_indices = train_indices[n_train_samples:]
    
    train_dataset = NBADataset(
        X_player_train_scaled[train_sample_indices],
        X_team_train_scaled[train_sample_indices],
        y_train[train_sample_indices],
        original_indices=train_sample_indices
    )
    val_dataset = NBADataset(
        X_player_train_scaled[val_sample_indices],
        X_team_train_scaled[val_sample_indices],
        y_train[val_sample_indices],
        original_indices=val_sample_indices
    )
    test_dataset = NBADataset(
        X_player_test_scaled, X_team_test_scaled, y_test,
        original_indices=np.arange(len(y_test))
    )
    
    n_team_features = X_team_train_scaled.shape[1]
    
    # Hyperparameter grid
    dropout_rates = [0.2, 0.3, 0.4]
    learning_rates = [0.0005, 0.001]
    batch_sizes = [8, 16]
    weight_decays = [1e-4, 5e-4]
    
    total_combinations = len(dropout_rates) * len(learning_rates) * len(batch_sizes) * len(weight_decays)
    
    print(f"\n3. Testing {total_combinations} hyperparameter combinations...")
    print(f"   Dropout rates: {dropout_rates}")
    print(f"   Learning rates: {learning_rates}")
    print(f"   Batch sizes: {batch_sizes}")
    print(f"   Weight decay: {weight_decays}\n")
    
    results = []
    best_spearman = -1
    best_config = None
    
    config_num = 0
    for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                for weight_decay in weight_decays:
                    config_num += 1
                    print(f"[{config_num}/{total_combinations}] Testing: dropout={dropout_rate}, lr={learning_rate}, batch={batch_size}, wd={weight_decay}")
                    
                    try:
                        result = train_and_evaluate_config(
                            train_dataset, val_dataset, test_dataset, meta_test,
                            dropout_rate, learning_rate, batch_size, weight_decay,
                            X_player_train.shape[1], n_team_features,
                            max_epochs=150, patience=8  # Reduced for faster tuning
                        )
                        results.append(result)
                        
                        if result['spearman'] > best_spearman:
                            best_spearman = result['spearman']
                            best_config = result
                        
                        print(f"   Result: Spearman={result['spearman']:.4f}, MAE={result['mae']:.2f}, Val Loss={result['best_val_loss']:.2f}\n")
                    except Exception as e:
                        print(f"   ERROR: {e}\n")
                        continue
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('spearman', ascending=False)
    
    output_file = os.path.join(output_dir, "hyperparameter_tuning_results.csv")
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING RESULTS")
    print("=" * 70)
    print("\nTop 5 Configurations:")
    print(results_df.head(5).to_string(index=False))
    
    print(f"\n\nBest Configuration:")
    print(f"  Dropout Rate: {best_config['dropout_rate']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Weight Decay: {best_config['weight_decay']}")
    print(f"  Spearman Correlation: {best_config['spearman']:.4f}")
    print(f"  Kendall Tau: {best_config['kendall_tau']:.4f}")
    print(f"  MAE: {best_config['mae']:.4f}")
    
    print(f"\nResults saved to: {output_file}")
    
    return results_df, best_config

if __name__ == "__main__":
    try:
        results_df, best_config = main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
