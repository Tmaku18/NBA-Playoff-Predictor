import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm

# ============ CONFIGURATION ============
TEAM_NAME_MAP = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
    'BKN': 'Brooklyn Nets', 'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers', 'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers', 'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies', 'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves', 'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder', 'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns', 'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs', 'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
    # full name mappings
    'Detroit Pistons': 'Detroit Pistons',
    'Indiana Pacers': 'Indiana Pacers',
    'San Antonio Spurs': 'San Antonio Spurs',
    'New Jersey Nets': 'New Jersey Nets',
    'Dallas Mavericks': 'Dallas Mavericks'
}

# ============ DATA LOADING ============
def load_player_stats(base_path):
    player_files = glob.glob(os.path.join(base_path, "Player Stats Regular and Playoff", "*_filtered.xlsx"))
    dfs = []
    
    for file in player_files:
        if '~$' in file:
            continue
            
        season = os.path.basename(file).split('_')[0]
        # Read the "Regular" sheet (historical files have both Regular/Playoff, future files only have Regular)
        try:
            df = pd.read_excel(file, sheet_name="Regular")
        except ValueError:
            # If "Regular" sheet doesn't exist, read the first sheet
            df = pd.read_excel(file)
        df['Team'] = df['Team'].map(TEAM_NAME_MAP).fillna(df['Team'])
        df['Season'] = season
        
        # select and clean relevant columns
        player_stats = df[['Player', 'Team', 'Season', 'G', 'MP', 'FG%', '3P%', 'eFG%', 'FT%',
                          'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS']].copy()
        
        # handle missing values
        for col in ['FG%', '3P%', 'eFG%', 'FT%']:
            player_stats[col] = player_stats[col].fillna(0)  # Assume 0% if no attempts
            
        # fill other missing values with 0 
        player_stats = player_stats.fillna(0)
        
        dfs.append(player_stats)
    
    return pd.concat(dfs, ignore_index=True)

def load_playoff_stats(base_path):
    playoff_files = glob.glob(os.path.join(base_path, "Actual Playoff Team Stats", "*__playoff_actual_team_stats.xlsx"))
    dfs = []
    
    for file in playoff_files:
        season = os.path.basename(file).split('__')[0]
        df = pd.read_excel(file)
        clean_df = df.rename(columns={'Tm': 'Team', 'Rk': 'Playoff_Rank'})[['Team', 'Playoff_Rank']].copy()
        clean_df['Team'] = clean_df['Team'].map(TEAM_NAME_MAP).fillna(clean_df['Team'])
        clean_df['Season'] = season
        dfs.append(clean_df)
    
    playoff_df = pd.concat(dfs, ignore_index=True)
    
    # validate playoff rankings
    for season in playoff_df['Season'].unique():
        season_ranks = playoff_df[playoff_df['Season'] == season]['Playoff_Rank']
        assert season_ranks.min() >= 1, f"Invalid rank <1 in {season}"
        assert len(season_ranks.unique()) == len(season_ranks), f"Duplicate ranks in {season}"
    
    return playoff_df

# ============ PREPROCESSING ============
def preprocess_data(player_df, playoff_df, fit_scalers=False, player_scaler=None, team_scaler=None):
    # team aggregation with null handling
    # Fix: Use proper group data access instead of problematic lambda with loc
    def weighted_avg(group, pct_col):
        """Calculate weighted average for percentage stats using games played as weights"""
        values = group[pct_col]
        weights = group['G']  # games played as weights
        mask = weights > 0
        if mask.any():
            return np.average(values[mask], weights=weights[mask])
        return 0
    
    # Aggregate team stats with proper group handling
    team_agg_list = []
    for (team, season), group in player_df.groupby(['Team', 'Season']):
        # Base stats
        mp_sum = group['MP'].sum()
        pts_sum = group['PTS'].sum()
        ast_sum = group['AST'].sum()
        tov_sum = group['TOV'].sum()
        trb_sum = group['TRB'].sum()
        g_mean = group['G'].mean()
        
        agg_dict = {
            'Team': team,
            'Season': season,
            # Removed: MP_sum, FG%_wt, 3P%_wt, eFG%_wt, BLK_sum, PF_sum (negative importance)
            'FT%_wt': weighted_avg(group, 'FT%'),
            'TRB_sum': trb_sum,
            'AST_sum': ast_sum,
            'AST_top3': group['AST'].nlargest(3).mean() if len(group) >= 3 else 0,
            'STL_sum': group['STL'].sum(),
            'TOV_sum': tov_sum,
            'PTS_sum': pts_sum,
            'PTS_top3': group['PTS'].nlargest(3).mean() if len(group) >= 3 else 0,
            'G_mean': g_mean,
            'G_std': group['G'].std(),
            # Derived efficiency features
            'PTS_per_game': pts_sum / max(g_mean, 1),  # Points per game average
            'AST_TOV_ratio': ast_sum / max(tov_sum, 1),  # Assist-to-turnover ratio
            'TRB_per_min': trb_sum / max(mp_sum, 1),  # Rebounds per minute
            'MP_per_game': mp_sum / max(g_mean, 1)  # Minutes per game (normalized)
        }
        team_agg_list.append(agg_dict)
    
    team_agg = pd.DataFrame(team_agg_list)
    
    # Handle infinite values from division
    team_agg = team_agg.replace([np.inf, -np.inf], 0)
    
    # flatten multi-index columns (now 16 features instead of 18)
    team_agg.columns = [
        'Team', 'Season', 'FT%_wt', 'TRB_sum', 'AST_sum', 'AST_top3', 'STL_sum',
        'TOV_sum', 'PTS_sum', 'PTS_top3', 'G_mean', 'G_std',
        'PTS_per_game', 'AST_TOV_ratio', 'TRB_per_min', 'MP_per_game'
    ]
    
    # fill any remaining null values
    team_agg = team_agg.fillna(0)
    
    # merge with playoff data
    merged = pd.merge(team_agg, playoff_df, on=['Team', 'Season'], how='inner')
    
    # prepare player arrays
    player_cols = ['MP', 'FG%', '3P%', 'eFG%', 'FT%', 'TRB', 'AST', 
                   'STL', 'BLK', 'TOV', 'PF', 'PTS']
    player_arrays = []
    for _, row in merged.iterrows():
        team_players = player_df[(player_df['Team'] == row['Team']) & 
                               (player_df['Season'] == row['Season'])]
        arr = team_players[player_cols].values
        player_arrays.append(arr)
    
    # pad player arrays to uniform size
    max_players = max(arr.shape[0] for arr in player_arrays) if player_arrays else 0
    if max_players > 0:
        player_arrays = np.stack([
            np.pad(arr, ((0, max_players - arr.shape[0]), (0, 0)), 
            mode='constant', constant_values=0)
            for arr in player_arrays
        ])
    else:
        player_arrays = np.array([])
    
    # Handle null values before scaling
    player_arrays = np.nan_to_num(player_arrays, nan=0)
    
    # prepare team features (without scaling yet)
    team_features = merged.drop(['Team', 'Season', 'Playoff_Rank'], axis=1).values
    team_features = np.nan_to_num(team_features, nan=0)
    
    # Scale only if requested (to avoid data leakage)
    if fit_scalers:
        # scale player stats
        if player_scaler is None:
            player_scaler = StandardScaler()
            original_shape = player_arrays.shape
            player_arrays_reshaped = player_arrays.reshape(-1, original_shape[-1])
            player_arrays_reshaped = np.nan_to_num(player_arrays_reshaped, nan=0)
            player_arrays = player_scaler.fit_transform(player_arrays_reshaped).reshape(original_shape)
        else:
            # Transform only
            original_shape = player_arrays.shape
            player_arrays_reshaped = player_arrays.reshape(-1, original_shape[-1])
            player_arrays_reshaped = np.nan_to_num(player_arrays_reshaped, nan=0)
            player_arrays = player_scaler.transform(player_arrays_reshaped).reshape(original_shape)
        
        # scale team features
        if team_scaler is None:
            team_scaler = StandardScaler()
            team_features = team_scaler.fit_transform(team_features)
        else:
            team_features = team_scaler.transform(team_features)
    
    # final validation
    if len(player_arrays) > 0:
        assert not np.isnan(player_arrays).any(), "NaN values in player arrays"
    assert not np.isnan(team_features).any(), "NaN values in team features"
    if 'Playoff_Rank' in merged.columns:
        assert not np.isnan(merged['Playoff_Rank'].values).any(), "NaN values in targets"
    
    if fit_scalers:
        return player_arrays, team_features, merged.get('Playoff_Rank', pd.Series([0]*len(merged))).values, merged, player_scaler, team_scaler
    return player_arrays, team_features, merged.get('Playoff_Rank', pd.Series([0]*len(merged))).values, merged
# ============ NEURAL NETWORK ============
class NBADataset(Dataset):
    def __init__(self, player_arrays, team_features, targets, original_indices=None):
        self.player_data = torch.FloatTensor(player_arrays)
        self.team_data = torch.FloatTensor(team_features)
        self.targets = torch.FloatTensor(targets)
        self.original_indices = original_indices if original_indices is not None else np.arange(len(targets))
    
    def __len__(self): 
        return len(self.targets)
    
    def __getitem__(self, idx):
        return {
            'player_stats': self.player_data[idx],
            'team_features': self.team_data[idx],
            'target': self.targets[idx],
            'original_idx': self.original_indices[idx]
        }

class HybridNBAModel(nn.Module):
    def __init__(self, n_players, n_player_features=12, n_team_features=14):
        super().__init__()
        # player pathway (1D CNN)
        self.player_net = nn.Sequential(
            nn.Conv1d(n_player_features, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Reduces player dimension to 1
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.Dropout(0.3)
        )
        # team pathway (deeper: 2 layers instead of 1)
        self.team_net = nn.Sequential(
            nn.Linear(n_team_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # combined network
        self.combined = nn.Sequential(
            nn.Linear(256, 128),  # 128 from player + 128 from team
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Final prediction
        )
    
    def forward(self, player_stats, team_features):
        # player stats: [batch, players, features] -> [batch, features, players]
        player_stats = player_stats.permute(0, 2, 1)
        player_out = self.player_net(player_stats)
        team_out = self.team_net(team_features)
        return self.combined(torch.cat([player_out, team_out], dim=1)).squeeze()

# ============ TRAINING ============

# ============ CUSTOM SUBSET CLASS ============
class NBASubset(torch.utils.data.Subset):
    def __getitem__(self, idx):
        original_item = self.dataset[self.indices[idx]]
        return {
            'player_stats': original_item['player_stats'],
            'team_features': original_item['team_features'],
            'target': original_item['target'],
            'original_idx': self.indices[idx]  # Preserve original index
        }

# ============ EVALUATION FUNCTION ============
def train_and_evaluate_2025_only(base_path, output_dir="Results"):
    # playoff teams for 2024–25
    playoff_teams_2025 = [
        "Oklahoma City Thunder", "Houston Rockets", "Los Angeles Lakers", "Denver Nuggets",
        "Los Angeles Clippers", "Minnesota Timberwolves", "Golden State Warriors", "Memphis Grizzlies",
        "Sacramento Kings", "Dallas Mavericks",
        "Cleveland Cavaliers", "Boston Celtics", "New York Knicks", "Indiana Pacers",
        "Milwaukee Bucks", "Detroit Pistons", "Orlando Magic", "Atlanta Hawks",
        "Chicago Bulls", "Miami Heat"
    ]

    # Load all player data (including historical)
    player_df = load_player_stats(base_path)
    
    # Load historical playoff data for training
    playoff_df_historical = load_playoff_stats(base_path)
    
    # Filter to only historical seasons (exclude 2024-25)
    historical_seasons = sorted([s for s in playoff_df_historical['Season'].unique() if s != '2024-25'])
    
    if len(historical_seasons) == 0:
        raise ValueError("No historical playoff data found for training")
    
    # Split seasons into 80% train and 20% test (better evaluation)
    np.random.seed(42)  # For reproducibility
    shuffled_seasons = historical_seasons.copy()
    np.random.shuffle(shuffled_seasons)
    n_train_seasons = int(0.8 * len(shuffled_seasons))
    train_seasons = sorted(shuffled_seasons[:n_train_seasons])
    test_seasons = sorted(shuffled_seasons[n_train_seasons:])
    
    print(f"\n=== Data Split ===")
    print(f"Training seasons ({len(train_seasons)}): {train_seasons}")
    print(f"Testing seasons ({len(test_seasons)}): {test_seasons}")
    
    # Split historical data into train and test
    playoff_df_train = playoff_df_historical[playoff_df_historical['Season'].isin(train_seasons)]
    playoff_df_test = playoff_df_historical[playoff_df_historical['Season'].isin(test_seasons)]
    
    # Preprocess training data
    X_player_train, X_team_train, y_train, meta_train = preprocess_data(
        player_df, playoff_df_train, fit_scalers=False)
    
    # Preprocess test data (for evaluation)
    X_player_test, X_team_test, y_test, meta_test = preprocess_data(
        player_df, playoff_df_test, fit_scalers=False)
    
    # Fit scalers on TRAINING data only (to avoid data leakage)
    player_scaler = StandardScaler()
    train_player_reshaped = X_player_train.reshape(-1, X_player_train.shape[-1])
    train_player_reshaped = np.nan_to_num(train_player_reshaped, nan=0)
    player_scaler.fit(train_player_reshaped)
    
    team_scaler = StandardScaler()
    train_team_clean = np.nan_to_num(X_team_train, nan=0)
    team_scaler.fit(train_team_clean)
    
    # Transform training data
    X_player_train_scaled = player_scaler.transform(train_player_reshaped).reshape(X_player_train.shape)
    X_team_train_scaled = team_scaler.transform(train_team_clean)
    
    # Transform test data using scalers fitted on training data
    test_player_reshaped = X_player_test.reshape(-1, X_player_test.shape[-1])
    test_player_reshaped = np.nan_to_num(test_player_reshaped, nan=0)
    X_player_test_scaled = player_scaler.transform(test_player_reshaped).reshape(X_player_test.shape)
    
    test_team_clean = np.nan_to_num(X_team_test, nan=0)
    X_team_test_scaled = team_scaler.transform(test_team_clean)
    
    # Split training data into train (80%) and validation (20%) for early stopping
    train_indices = np.arange(len(y_train))
    np.random.seed(42)
    np.random.shuffle(train_indices)
    n_train_samples = int(0.8 * len(train_indices))
    train_sample_indices = train_indices[:n_train_samples]
    val_sample_indices = train_indices[n_train_samples:]
    
    # Create datasets
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
    test_dataset = NBADataset(X_player_test_scaled, X_team_test_scaled, y_test, original_indices=np.arange(len(y_test)))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize and train model with early stopping and LR scheduling
    # Team features: 14 features (removed 6 negative-importance features, added 4 derived features)
    n_team_features = X_team_train_scaled.shape[1]
    model = HybridNBAModel(n_players=X_player_train.shape[1], n_team_features=n_team_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    max_epochs = 200  # Increased max epochs since we have early stopping
    
    print(f"\n=== Training on {len(train_dataset)} team-seasons from {len(train_seasons)} seasons ===")
    print(f"Validation set: {len(val_dataset)} team-seasons")
    print(f"Test set: {len(test_dataset)} team-seasons from {len(test_seasons)} seasons\n")
    
    model.train()
    for epoch in range(max_epochs):
        # Training phase
        epoch_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(batch['player_stats'], batch['team_features'])
            loss = criterion(outputs, batch['target'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation phase
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
        
        # Learning rate scheduling
        scheduler.step(val_loss_avg)
        
        # Early stopping check
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f} (Best: {best_val_loss:.4f})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            print(f"Best validation loss: {best_val_loss:.4f} at epoch {epoch+1-patience}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with validation loss: {best_val_loss:.4f}")
    
    # Evaluate on test set
    print(f"\n=== Evaluating on test set ({len(test_dataset)} team-seasons from {len(test_seasons)} seasons) ===")
    model.eval()
    test_loss = 0.0
    test_predictions = []
    test_targets = []
    test_teams = []
    test_seasons_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['player_stats'], batch['team_features'])
            loss = criterion(outputs, batch['target'])
            test_loss += loss.item()
            test_predictions.extend(outputs.numpy())
            test_targets.extend(batch['target'].numpy())
            # Get team names and seasons from meta_test
            for idx in batch['original_idx'].numpy():
                test_teams.append(meta_test.iloc[idx]['Team'])
                test_seasons_list.append(meta_test.iloc[idx]['Season'])
    
    test_loss_avg = test_loss / len(test_loader)
    print(f"Test Loss (MSE): {test_loss_avg:.4f}")
    
    # Calculate per-season rankings and metrics
    test_df = pd.DataFrame({
        'Season': test_seasons_list,
        'Team': test_teams,
        'Predicted': test_predictions,
        'Actual': test_targets
    })
    
    # Calculate predicted ranks per season
    test_df['Predicted_Rank'] = test_df.groupby('Season')['Predicted'].rank(method='dense', ascending=True).astype(int)
    test_df['Actual_Rank'] = test_df.groupby('Season')['Actual'].rank(method='dense', ascending=True).astype(int)
    
    # Calculate metrics per season
    season_metrics = []
    for season in sorted(test_seasons):
        season_data = test_df[test_df['Season'] == season]
        actual_ranks = season_data['Actual_Rank'].values
        predicted_ranks = season_data['Predicted_Rank'].values
        
        spearman_corr, spearman_p = spearmanr(actual_ranks, predicted_ranks)
        kendall_corr, kendall_p = kendalltau(actual_ranks, predicted_ranks)
        mae = np.mean(np.abs(actual_ranks - predicted_ranks))
        perfect_matches = np.sum(actual_ranks == predicted_ranks)
        
        season_metrics.append({
            'Season': season,
            'Spearman_Correlation': spearman_corr,
            'Kendall_Tau': kendall_corr,
            'MAE': mae,
            'Perfect_Matches': perfect_matches,
            'Total_Teams': len(actual_ranks)
        })
    
    # Overall test metrics
    overall_actual = test_df['Actual_Rank'].values
    overall_predicted = test_df['Predicted_Rank'].values
    overall_spearman = spearmanr(overall_actual, overall_predicted)[0]
    overall_kendall = kendalltau(overall_actual, overall_predicted)[0]
    overall_mae = np.mean(np.abs(overall_actual - overall_predicted))
    overall_perfect = np.sum(overall_actual == overall_predicted)
    
    print("\n=== Test Set Performance (Per Season) ===")
    metrics_df = pd.DataFrame(season_metrics)
    print(metrics_df.to_string(index=False))
    
    print(f"\n=== Overall Test Set Performance ===")
    print(f"Spearman Correlation: {overall_spearman:.4f}")
    print(f"Kendall Tau: {overall_kendall:.4f}")
    print(f"Mean Absolute Error: {overall_mae:.4f}")
    print(f"Perfect Matches: {overall_perfect}/{len(overall_actual)} ({100*overall_perfect/len(overall_actual):.1f}%)")
    
    # Save test metrics
    metrics_df.to_csv(os.path.join(output_dir, "test_set_metrics.csv"), index=False)
    print(f"\nTest metrics saved to: {os.path.join(output_dir, 'test_set_metrics.csv')}")
    
    # Now prepare 2024-25 data for prediction
    print(f"\n=== Predicting on 2024-25 season ===")
    
    # manually create playoff_df for 2024–25 (without ranks, just for filtering)
    playoff_df_2025 = pd.DataFrame({
        'Team': playoff_teams_2025,
        'Season': ['2024-25'] * len(playoff_teams_2025),
        'Playoff_Rank': [0] * len(playoff_teams_2025)  # dummy value, not used
    })

    # Preprocess 2024-25 data (without scaling)
    X_player_2025, X_team_2025, y_dummy, meta_2025 = preprocess_data(
        player_df, playoff_df_2025, fit_scalers=False)

    season = "2024-25"
    season_mask = meta_2025['Season'] == season
    season_indices = np.where(season_mask)[0]

    if len(season_indices) < 4:
        raise ValueError(f"Too few teams to predict for {season}")

    # Transform 2024-25 data using scalers fitted on historical data
    season_player = X_player_2025[season_indices]
    season_team = X_team_2025[season_indices]
    
    season_player_reshaped = season_player.reshape(-1, season_player.shape[-1])
    season_player_reshaped = np.nan_to_num(season_player_reshaped, nan=0)
    season_player_scaled = player_scaler.transform(season_player_reshaped).reshape(season_player.shape)
    
    season_team_clean = np.nan_to_num(season_team, nan=0)
    season_team_scaled = team_scaler.transform(season_team_clean)
    
    # Create prediction dataset
    pred_dataset = NBADataset(
        season_player_scaled,
        season_team_scaled,
        y_dummy[season_indices],
        original_indices=season_indices
    )

    # Predict
    model.eval()
    full_player = torch.stack([pred_dataset[i]['player_stats'] for i in range(len(pred_dataset))])
    full_team = torch.stack([pred_dataset[i]['team_features'] for i in range(len(pred_dataset))])
    full_indices = [pred_dataset[i]['original_idx'] for i in range(len(pred_dataset))]

    with torch.no_grad():
        full_predictions = model(full_player, full_team).numpy()

    predicted_ranks = np.argsort(np.argsort(full_predictions)) + 1
    team_names = meta_2025['Team'].iloc[full_indices].tolist()

    results_2025 = pd.DataFrame({
        'Season': season,
        'Team': team_names,
        'Predicted_Rank': predicted_ranks
    }).sort_values('Predicted_Rank')

    print(f"\n=== Predicted 2024–25 Playoff Rankings ===")
    print(results_2025[['Predicted_Rank', 'Team']].rename(columns={'Predicted_Rank': 'Rank'}).to_string(index=False))

    # export
    os.makedirs(output_dir, exist_ok=True)
    pred_output_path = os.path.join(output_dir, "HYBRID_2024-25_predictions.csv")
    results_2025.to_csv(pred_output_path, index=False)
    print(f"\nPredictions for 2024–25 saved to: {pred_output_path}")

    return results_2025

    
if __name__ == "__main__":
    results = train_and_evaluate_2025_only("Preprocessing/Preprocessed Data")
    print(results.head())
