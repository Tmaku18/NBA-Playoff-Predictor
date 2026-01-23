"""
Feature Importance Analysis for Hybrid NBA Model
Computes permutation importance to identify which stats are most important
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau

# Add path to import model - adjust based on where script is run from
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'AI-Project-main', 'Model', 'Future Prediction')
if not os.path.exists(model_path):
    model_path = os.path.join(script_dir, 'Model', 'Future Prediction')
sys.path.insert(0, model_path)

from hybrid_model_future_pred import (
    load_player_stats, load_playoff_stats, preprocess_data,
    HybridNBAModel, NBADataset
)

def compute_permutation_importance(model, test_dataset, team_feature_names, n_repeats=3):
    """
    Calculate permutation importance for team features by measuring
    how much prediction error increases when each feature is randomly shuffled
    """
    model.eval()
    criterion = nn.MSELoss()
    
    # Calculate baseline loss
    baseline_losses = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            player_stats = sample['player_stats'].unsqueeze(0)
            team_features = sample['team_features'].unsqueeze(0)
            target = sample['target'].unsqueeze(0)
            
            output = model(player_stats, team_features)
            loss = criterion(output, target)
            baseline_losses.append(loss.item())
    baseline_loss = np.mean(baseline_losses)
    
    print(f"Baseline test loss: {baseline_loss:.4f}\n")
    print("Computing feature importance (this may take a few minutes)...")
    
    # Calculate importance for each feature
    importances = []
    
    for feature_idx, feature_name in enumerate(team_feature_names):
        feature_importance_scores = []
        
        for repeat in range(n_repeats):
            permuted_losses = []
            
            with torch.no_grad():
                for i in range(len(test_dataset)):
                    sample = test_dataset[i]
                    player_stats = sample['player_stats'].unsqueeze(0)
                    team_features = sample['team_features'].clone().unsqueeze(0)
                    target = sample['target'].unsqueeze(0)
                    
                    # Permute this feature by taking a random value from another sample
                    perm_idx = np.random.randint(0, len(test_dataset))
                    permuted_value = test_dataset[perm_idx]['team_features'][feature_idx]
                    team_features[0, feature_idx] = permuted_value
                    
                    output = model(player_stats, team_features)
                    loss = criterion(output, target)
                    permuted_losses.append(loss.item())
            
            # Importance = increase in loss when feature is permuted
            permuted_loss = np.mean(permuted_losses)
            importance = permuted_loss - baseline_loss
            feature_importance_scores.append(importance)
        
        importances.append({
            'Feature': feature_name,
            'Importance': np.mean(feature_importance_scores),
            'Std': np.std(feature_importance_scores),
            'Percent_Increase': (np.mean(feature_importance_scores) / baseline_loss) * 100
        })
        
        print(f"  {feature_name:15s}: {np.mean(feature_importance_scores):.4f} ± {np.std(feature_importance_scores):.4f}")
    
    # Sort by importance
    importance_df = pd.DataFrame(importances)
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def analyze_feature_importance():
    """Main function to train model and analyze feature importance"""
    
    # Adjust paths based on where script is run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Try different path combinations
    possible_base_paths = [
        os.path.join(script_dir, "AI-Project-main", "Preprocessing", "Preprocessed Data"),
        os.path.join(script_dir, "Preprocessing", "Preprocessed Data"),
        "AI-Project-main/Preprocessing/Preprocessed Data",
        "Preprocessing/Preprocessed Data"
    ]
    possible_output_dirs = [
        os.path.join(script_dir, "AI-Project-main", "Results"),
        os.path.join(script_dir, "Results"),
        "AI-Project-main/Results",
        "Results"
    ]
    
    base_path = None
    for path in possible_base_paths:
        if os.path.exists(path):
            base_path = path
            break
    if base_path is None:
        raise FileNotFoundError(f"Could not find data directory. Tried: {possible_base_paths}")
    
    output_dir = possible_output_dirs[0] if os.path.exists(os.path.dirname(possible_output_dirs[0])) else possible_output_dirs[2]
    
    print(f"Using base_path: {base_path}")
    print(f"Using output_dir: {output_dir}")
    
    # Team feature names (must match preprocessing order in hybrid_model_future_pred.py)
    # Updated: 20 features after removing noisy features and adding derived features
    team_feature_names = [
        'FT%_wt', 'TRB_sum', 'AST_sum', 'AST_top3', 'AST_top5',
        'STL_sum', 'PTS_sum', 'PTS_top3', 'PTS_top5', 'G_mean', 'G_std',
        'PTS_per_game', 'AST_TOV_ratio', 'TRB_per_min', 'MP_per_game',
        'Defensive_activity', 'Defensive_per_game', 'PTS_std', 'AST_std', 'Years_since_2003'
    ]
    
    print("=" * 70)
    print("Feature Importance Analysis for Hybrid NBA Model")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    player_df = load_player_stats(base_path)
    playoff_df_historical = load_playoff_stats(base_path)
    
    # Get historical seasons (exclude 2024-25)
    historical_seasons = sorted([s for s in playoff_df_historical['Season'].unique() if s != '2024-25'])
    
    # Use same 90/10 split as in the main model
    np.random.seed(42)
    shuffled_seasons = historical_seasons.copy()
    np.random.shuffle(shuffled_seasons)
    n_train_seasons = int(0.9 * len(shuffled_seasons))
    train_seasons = sorted(shuffled_seasons[:n_train_seasons])
    test_seasons = sorted(shuffled_seasons[n_train_seasons:])
    
    print(f"   Training seasons: {len(train_seasons)}")
    print(f"   Test seasons: {len(test_seasons)}")
    
    # Split data
    playoff_df_train = playoff_df_historical[playoff_df_historical['Season'].isin(train_seasons)]
    playoff_df_test = playoff_df_historical[playoff_df_historical['Season'].isin(test_seasons)]
    
    # Preprocess
    print("\n2. Preprocessing data...")
    X_player_train, X_team_train, y_train, meta_train = preprocess_data(
        player_df, playoff_df_train, fit_scalers=False)
    X_player_test, X_team_test, y_test, meta_test = preprocess_data(
        player_df, playoff_df_test, fit_scalers=False)
    
    # Fit scalers on training data only
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
    
    # Create datasets
    test_dataset = NBADataset(X_player_test_scaled, X_team_test_scaled, y_test)
    train_dataset = NBADataset(X_player_train_scaled, X_team_train_scaled, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Train model (quick training for importance analysis)
    print("\n3. Training model (75 epochs)...")
    n_team_features = X_team_train_scaled.shape[1]  # Should be 20 features
    model = HybridNBAModel(n_players=X_player_train.shape[1], n_team_features=n_team_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(75):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['player_stats'], batch['team_features'])
            loss = criterion(outputs, batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 25 == 0:
            print(f"   Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Compute feature importance
    print("\n4. Computing permutation importance...")
    importance_df = compute_permutation_importance(model, test_dataset, team_feature_names, n_repeats=3)
    
    # Display results
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE RANKINGS")
    print("=" * 70)
    print("\nHigher importance = feature has more impact on predictions")
    print("(Values show increase in loss when feature is randomly permuted)\n")
    print(importance_df.to_string(index=False))
    
    # Save results
    output_file = os.path.join(output_dir, "feature_importance_analysis.csv")
    importance_df.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TOP 5 MOST IMPORTANT FEATURES:")
    print("=" * 70)
    for i, row in importance_df.head(5).iterrows():
        print(f"{row['Feature']:20s}: {row['Importance']:.4f} ({row['Percent_Increase']:.1f}% increase in loss)")
    
    return importance_df

if __name__ == "__main__":
    try:
        importance_df = analyze_feature_importance()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
