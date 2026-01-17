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
        stl_sum = group['STL'].sum()
        blk_sum = group['BLK'].sum()
        g_mean = group['G'].mean()
        n_players = len(group)
        
        # Top player features
        pts_top3 = group['PTS'].nlargest(3).mean() if n_players >= 3 else 0
        pts_top5 = group['PTS'].nlargest(5).mean() if n_players >= 5 else pts_top3
        ast_top3 = group['AST'].nlargest(3).mean() if n_players >= 3 else 0
        ast_top5 = group['AST'].nlargest(5).mean() if n_players >= 5 else ast_top3
        
        # Defensive efficiency features
        defensive_activity = stl_sum + blk_sum  # Combined steals and blocks
        defensive_per_game = defensive_activity / max(g_mean, 1)
        
        # Team chemistry/distribution metrics
        pts_std = group['PTS'].std() if n_players > 1 else 0
        ast_std = group['AST'].std() if n_players > 1 else 0
        
        # Temporal features: encode season as years since 2003
        year_start = int(season.split('-')[0])
        years_since_2003 = year_start - 2003
        
        agg_dict = {
            'Team': team,
            'Season': season,
            # Removed: MP_sum, FG%_wt, 3P%_wt, eFG%_wt, BLK_sum, PF_sum (negative importance)
            'FT%_wt': weighted_avg(group, 'FT%'),
            'TRB_sum': trb_sum,
            'AST_sum': ast_sum,
            'AST_top3': ast_top3,
            'AST_top5': ast_top5,  # New: top-5 assists
            'STL_sum': stl_sum,
            'TOV_sum': tov_sum,
            'PTS_sum': pts_sum,
            'PTS_top3': pts_top3,
            'PTS_top5': pts_top5,  # New: top-5 points
            'G_mean': g_mean,
            'G_std': group['G'].std(),
            # Derived efficiency features
            'PTS_per_game': pts_sum / max(g_mean, 1),  # Points per game average
            'AST_TOV_ratio': ast_sum / max(tov_sum, 1),  # Assist-to-turnover ratio
            'TRB_per_min': trb_sum / max(mp_sum, 1),  # Rebounds per minute
            'MP_per_game': mp_sum / max(g_mean, 1),  # Minutes per game (normalized)
            # New: Defensive efficiency features
            'Defensive_activity': defensive_activity,  # STL + BLK combined
            'Defensive_per_game': defensive_per_game,  # Defensive stats per game
            # New: Team chemistry/distribution metrics
            'PTS_std': pts_std,  # Scoring distribution
            'AST_std': ast_std,  # Assist distribution
            # New: Temporal features
            'Years_since_2003': years_since_2003  # Season encoding for temporal trends
        }
        team_agg_list.append(agg_dict)
    
    team_agg = pd.DataFrame(team_agg_list)
    
    # Handle infinite values from division
    team_agg = team_agg.replace([np.inf, -np.inf], 0)
    
    # flatten multi-index columns (now 21 features: 14 original + 6 new + 1 temporal)
    team_agg.columns = [
        'Team', 'Season', 'FT%_wt', 'TRB_sum', 'AST_sum', 'AST_top3', 'AST_top5',
        'STL_sum', 'TOV_sum', 'PTS_sum', 'PTS_top3', 'PTS_top5', 'G_mean', 'G_std',
        'PTS_per_game', 'AST_TOV_ratio', 'TRB_per_min', 'MP_per_game',
        'Defensive_activity', 'Defensive_per_game', 'PTS_std', 'AST_std', 'Years_since_2003'
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
    
    def get_season_groups(self, meta_df):
        """Get season group indices for rank-aware loss"""
        season_to_idx = {season: idx for idx, season in enumerate(sorted(meta_df['Season'].unique()))}
        return torch.LongTensor([season_to_idx[meta_df.iloc[self.original_indices[i]]['Season']] 
                                 for i in range(len(self))])

class RankAwareLoss(nn.Module):
    """Combined loss: MSE for regression + rank-aware component for ranking accuracy"""
    def __init__(self, mse_weight=0.7, rank_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions, targets, season_groups=None):
        """
        Args:
            predictions: tensor of predicted values
            targets: tensor of actual values
            season_groups: optional tensor indicating which season each sample belongs to
                          (for computing rank correlation within seasons)
        """
        # MSE component
        mse = self.mse_loss(predictions, targets)
        
        # Rank-aware component: Spearman correlation loss
        # We want to maximize correlation, so we minimize (1 - correlation)
        if season_groups is not None:
            # Compute rank correlation within each season
            rank_losses = []
            unique_seasons = torch.unique(season_groups)
            for season in unique_seasons:
                mask = (season_groups == season)
                if mask.sum() < 2:  # Need at least 2 samples for correlation
                    continue
                
                pred_season = predictions[mask]
                target_season = targets[mask]
                
                # Compute ranks
                pred_ranks = torch.argsort(torch.argsort(pred_season.squeeze(), descending=False))
                target_ranks = torch.argsort(torch.argsort(target_season.squeeze(), descending=False))
                
                # Normalize ranks to [0, 1]
                n = len(pred_ranks)
                pred_ranks_norm = pred_ranks.float() / (n - 1) if n > 1 else pred_ranks.float()
                target_ranks_norm = target_ranks.float() / (n - 1) if n > 1 else target_ranks.float()
                
                # Compute correlation (simplified: 1 - cosine similarity of ranks)
                pred_centered = pred_ranks_norm - pred_ranks_norm.mean()
                target_centered = target_ranks_norm - target_ranks_norm.mean()
                
                numerator = (pred_centered * target_centered).sum()
                pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
                target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
                
                correlation = numerator / (pred_std * target_std + 1e-8)
                rank_losses.append(1.0 - correlation)  # Minimize (1 - correlation)
            
            if len(rank_losses) > 0:
                rank_loss = torch.stack(rank_losses).mean()
            else:
                rank_loss = torch.tensor(0.0, device=predictions.device)
        else:
            # Global rank correlation
            pred_ranks = torch.argsort(torch.argsort(predictions.squeeze(), descending=False))
            target_ranks = torch.argsort(torch.argsort(targets.squeeze(), descending=False))
            
            n = len(pred_ranks)
            pred_ranks_norm = pred_ranks.float() / (n - 1) if n > 1 else pred_ranks.float()
            target_ranks_norm = target_ranks.float() / (n - 1) if n > 1 else target_ranks.float()
            
            pred_centered = pred_ranks_norm - pred_ranks_norm.mean()
            target_centered = target_ranks_norm - target_ranks_norm.mean()
            
            numerator = (pred_centered * target_centered).sum()
            pred_std = torch.sqrt((pred_centered ** 2).sum() + 1e-8)
            target_std = torch.sqrt((target_centered ** 2).sum() + 1e-8)
            
            correlation = numerator / (pred_std * target_std + 1e-8)
            rank_loss = (1.0 - correlation).to(predictions.device)
        
        total_loss = self.mse_weight * mse + self.rank_weight * rank_loss
        return total_loss, mse, rank_loss

class HybridNBAModel(nn.Module):
    def __init__(self, n_players, n_player_features=12, n_team_features=21, dropout_rate=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # Player pathway with attention mechanism
        self.player_conv1 = nn.Conv1d(n_player_features, 32, 3, padding=1)
        self.player_bn1 = nn.BatchNorm1d(32)
        self.player_conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.player_bn2 = nn.BatchNorm1d(64)
        
        if use_attention:
            # Attention mechanism to weight important players
            self.attention = nn.Sequential(
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )
        else:
            # Fallback to average pooling
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.player_fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.Dropout(dropout_rate)
        )
        # team pathway (deeper: 2 layers instead of 1)
        self.team_net = nn.Sequential(
            nn.Linear(n_team_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # combined network
        self.combined = nn.Sequential(
            nn.Linear(256, 128),  # 128 from player + 128 from team
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1)  # Final prediction
        )
    
    def forward(self, player_stats, team_features):
        # player stats: [batch, players, features] -> [batch, features, players]
        player_stats = player_stats.permute(0, 2, 1)
        
        # Convolutional layers
        x = self.player_conv1(player_stats)
        x = self.player_bn1(x)
        x = torch.relu(x)
        x = self.player_conv2(x)
        x = self.player_bn2(x)
        x = torch.relu(x)
        
        # Attention mechanism or average pooling
        if self.use_attention:
            # x shape: [batch, 64, players]
            # Transpose to [batch, players, 64] for attention
            x_transposed = x.permute(0, 2, 1)  # [batch, players, 64]
            # Compute attention weights
            attention_weights = self.attention(x_transposed)  # [batch, players, 1]
            attention_weights = torch.softmax(attention_weights, dim=1)  # Normalize across players
            # Apply attention: weighted sum
            player_out = torch.sum(attention_weights * x_transposed, dim=1)  # [batch, 64]
        else:
            # Average pooling fallback
            x = self.avg_pool(x)  # [batch, 64, 1]
            player_out = x.squeeze(-1)  # [batch, 64]
        
        # Final fully connected layer
        player_out = self.player_fc(player_out)  # [batch, 128]
        
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
    
    # Hyperparameters (optimized via hyperparameter tuning)
    # Best config: dropout=0.2, lr=0.0005, batch=8, wd=0.0001 -> Spearman=0.3620
    dropout_rate = 0.2
    learning_rate = 0.0005
    batch_size = 8
    weight_decay = 1e-4
    use_attention = True  # Enable attention mechanism
    use_rank_aware_loss = True  # Phase 3: Use rank-aware loss
    n_ensemble_models = 5  # Phase 3: Number of ensemble models
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize and train model with early stopping and LR scheduling
    # Team features: 21 features (removed 6 negative-importance features, added 4 derived + 6 new + 1 temporal)
    n_team_features = X_team_train_scaled.shape[1]
    
    # Phase 3: Train ensemble of models
    ensemble_models = []
    print(f"\n=== Training Ensemble of {n_ensemble_models} Models ===")
    
    for ensemble_idx in range(n_ensemble_models):
        print(f"\n--- Training Model {ensemble_idx + 1}/{n_ensemble_models} ---")
        
        # Initialize model with different random seed for each ensemble member
        torch.manual_seed(42 + ensemble_idx)
        np.random.seed(42 + ensemble_idx)
        
        model = HybridNBAModel(
            n_players=X_player_train.shape[1], 
            n_team_features=n_team_features, 
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Phase 3: Use rank-aware loss
        if use_rank_aware_loss:
            criterion = RankAwareLoss(mse_weight=0.7, rank_weight=0.3)
        else:
            criterion = nn.MSELoss()
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Early stopping parameters for this model
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        max_epochs = 200  # Reduced for ensemble training
        
        model.train()
        for epoch in range(max_epochs):
            # Training phase
            epoch_train_loss = 0.0
            epoch_train_mse = 0.0
            epoch_train_rank = 0.0
            
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['player_stats'], batch['team_features'])
                
                # Get season groups for rank-aware loss
                if use_rank_aware_loss:
                    # Get season indices for this batch
                    batch_indices = batch['original_idx'].numpy()
                    batch_seasons = [meta_train.iloc[idx]['Season'] for idx in batch_indices]
                    unique_seasons = sorted(set(batch_seasons))
                    season_to_idx = {s: i for i, s in enumerate(unique_seasons)}
                    season_groups = torch.LongTensor([season_to_idx[s] for s in batch_seasons])
                    
                    loss, mse_loss, rank_loss = criterion(outputs, batch['target'], season_groups)
                    epoch_train_mse += mse_loss.item()
                    epoch_train_rank += rank_loss.item()
                else:
                    loss = criterion(outputs, batch['target'])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_train_loss += loss.item() if isinstance(loss, torch.Tensor) else loss[0].item()
            
            # Validation phase
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch['player_stats'], batch['team_features'])
                    if use_rank_aware_loss:
                        batch_indices = batch['original_idx'].numpy()
                        batch_seasons = [meta_train.iloc[idx]['Season'] for idx in batch_indices]
                        unique_seasons = sorted(set(batch_seasons))
                        season_to_idx = {s: i for i, s in enumerate(unique_seasons)}
                        season_groups = torch.LongTensor([season_to_idx[s] for s in batch_seasons])
                        loss, _, _ = criterion(outputs, batch['target'], season_groups)
                    else:
                        loss = criterion(outputs, batch['target'])
                    epoch_val_loss += loss.item() if isinstance(loss, torch.Tensor) else loss[0].item()
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
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                if use_rank_aware_loss:
                    print(f"  Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss_avg:.4f} (MSE: {epoch_train_mse/len(train_loader):.4f}, Rank: {epoch_train_rank/len(train_loader):.4f}), Val Loss: {val_loss_avg:.4f}")
                else:
                    print(f"  Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        ensemble_models.append(model)
        print(f"  Model {ensemble_idx + 1} trained. Best validation loss: {best_val_loss:.4f}")
    
    print(f"\n=== Ensemble Training Complete ===\n")
    
    # Phase 3: Evaluate ensemble on test set
    print(f"\n=== Evaluating Ensemble on test set ({len(test_dataset)} team-seasons from {len(test_seasons)} seasons) ===")
    
    # Collect predictions from all ensemble models
    ensemble_predictions = []
    test_targets = []
    test_teams = []
    test_seasons_list = []
    
    for model in ensemble_models:
        model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            # Get predictions from all models
            batch_predictions = []
            for model in ensemble_models:
                outputs = model(batch['player_stats'], batch['team_features'])
                batch_predictions.append(outputs.numpy())
            
            # Average predictions across ensemble
            ensemble_batch_pred = np.mean(batch_predictions, axis=0)
            ensemble_predictions.extend(ensemble_batch_pred.flatten())
            
            # Collect targets and metadata (only once)
            test_targets.extend(batch['target'].numpy().flatten())
            for idx in batch['original_idx'].numpy():
                test_teams.append(meta_test.iloc[idx]['Team'])
                test_seasons_list.append(meta_test.iloc[idx]['Season'])
    
    # Calculate MSE loss
    test_predictions_array = np.array(ensemble_predictions)
    test_targets_array = np.array(test_targets)
    test_loss_avg = np.mean((test_predictions_array - test_targets_array) ** 2)
    print(f"Test Loss (MSE): {test_loss_avg:.4f}")
    
    # Calculate per-season rankings and metrics
    test_df = pd.DataFrame({
        'Season': test_seasons_list,
        'Team': test_teams,
        'Predicted': ensemble_predictions,
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

    # Phase 3: Predict using ensemble
    for model in ensemble_models:
        model.eval()
    
    full_player = torch.stack([pred_dataset[i]['player_stats'] for i in range(len(pred_dataset))])
    full_team = torch.stack([pred_dataset[i]['team_features'] for i in range(len(pred_dataset))])
    full_indices = [pred_dataset[i]['original_idx'] for i in range(len(pred_dataset))]

    with torch.no_grad():
        # Get predictions from all ensemble models
        ensemble_preds = []
        for model in ensemble_models:
            outputs = model(full_player, full_team)
            ensemble_preds.append(outputs.numpy())
        
        # Average predictions across ensemble
        full_predictions = np.mean(ensemble_preds, axis=0)

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

# ============ CROSS-VALIDATION FUNCTION ============
def cross_validate_model(base_path, n_folds=5, output_dir="Results"):
    """
    Phase 3: Cross-validation for robust evaluation
    Splits historical seasons into n_folds and evaluates on each fold
    """
    print("=" * 70)
    print("CROSS-VALIDATION EVALUATION")
    print("=" * 70)
    
    # Load data
    player_df = load_player_stats(base_path)
    playoff_df_historical = load_playoff_stats(base_path)
    
    historical_seasons = sorted([s for s in playoff_df_historical['Season'].unique() if s != '2024-25'])
    
    if len(historical_seasons) < n_folds:
        print(f"Warning: Only {len(historical_seasons)} seasons available, reducing folds to {len(historical_seasons)}")
        n_folds = len(historical_seasons)
    
    # Split seasons into folds
    np.random.seed(42)
    shuffled_seasons = historical_seasons.copy()
    np.random.shuffle(shuffled_seasons)
    
    fold_size = len(shuffled_seasons) // n_folds
    fold_results = []
    
    for fold in range(n_folds):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*70}")
        
        # Split into train and test for this fold
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else len(shuffled_seasons)
        test_seasons_fold = sorted(shuffled_seasons[test_start:test_end])
        train_seasons_fold = sorted([s for s in shuffled_seasons if s not in test_seasons_fold])
        
        print(f"Train seasons ({len(train_seasons_fold)}): {train_seasons_fold[:5]}...")
        print(f"Test seasons ({len(test_seasons_fold)}): {test_seasons_fold}")
        
        # Preprocess
        playoff_df_train = playoff_df_historical[playoff_df_historical['Season'].isin(train_seasons_fold)]
        playoff_df_test = playoff_df_historical[playoff_df_historical['Season'].isin(test_seasons_fold)]
        
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
        
        # Transform
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
        
        # Train ensemble (simplified: 3 models for CV)
        n_team_features = X_team_train_scaled.shape[1]
        ensemble_models = []
        dropout_rate = 0.2
        learning_rate = 0.0005
        batch_size = 8
        weight_decay = 1e-4
        
        for ensemble_idx in range(3):  # 3 models for faster CV
            torch.manual_seed(42 + ensemble_idx)
            np.random.seed(42 + ensemble_idx)
            
            model = HybridNBAModel(
                n_players=X_player_train.shape[1],
                n_team_features=n_team_features,
                dropout_rate=dropout_rate,
                use_attention=True
            )
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = RankAwareLoss(mse_weight=0.7, rank_weight=0.3)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            best_val_loss = float('inf')
            patience = 8
            patience_counter = 0
            best_model_state = None
            max_epochs = 100  # Reduced for CV
            
            model.train()
            for epoch in range(max_epochs):
                epoch_train_loss = 0.0
                for batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch['player_stats'], batch['team_features'])
                    
                    batch_indices = batch['original_idx'].numpy()
                    batch_seasons = [meta_train.iloc[idx]['Season'] for idx in batch_indices]
                    unique_seasons = sorted(set(batch_seasons))
                    season_to_idx = {s: i for i, s in enumerate(unique_seasons)}
                    season_groups = torch.LongTensor([season_to_idx[s] for s in batch_seasons])
                    
                    loss, _, _ = criterion(outputs, batch['target'], season_groups)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_train_loss += loss.item()
                
                model.eval()
                epoch_val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        outputs = model(batch['player_stats'], batch['team_features'])
                        batch_indices = batch['original_idx'].numpy()
                        batch_seasons = [meta_train.iloc[idx]['Season'] for idx in batch_indices]
                        unique_seasons = sorted(set(batch_seasons))
                        season_to_idx = {s: i for i, s in enumerate(unique_seasons)}
                        season_groups = torch.LongTensor([season_to_idx[s] for s in batch_seasons])
                        loss, _, _ = criterion(outputs, batch['target'], season_groups)
                        epoch_val_loss += loss.item()
                model.train()
                
                val_loss_avg = epoch_val_loss / len(val_loader)
                scheduler.step(val_loss_avg)
                
                if val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            ensemble_models.append(model)
        
        # Evaluate on test set
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        for model in ensemble_models:
            model.eval()
        
        ensemble_predictions = []
        test_targets = []
        test_teams = []
        test_seasons_list = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch_predictions = []
                for model in ensemble_models:
                    outputs = model(batch['player_stats'], batch['team_features'])
                    batch_predictions.append(outputs.numpy())
                
                ensemble_batch_pred = np.mean(batch_predictions, axis=0)
                ensemble_predictions.extend(ensemble_batch_pred.flatten())
                test_targets.extend(batch['target'].numpy().flatten())
                
                for idx in batch['original_idx'].numpy():
                    test_teams.append(meta_test.iloc[idx]['Team'])
                    test_seasons_list.append(meta_test.iloc[idx]['Season'])
        
        # Calculate metrics
        test_df = pd.DataFrame({
            'Season': test_seasons_list,
            'Team': test_teams,
            'Predicted': ensemble_predictions,
            'Actual': test_targets
        })
        
        test_df['Predicted_Rank'] = test_df.groupby('Season')['Predicted'].rank(method='dense', ascending=True).astype(int)
        test_df['Actual_Rank'] = test_df.groupby('Season')['Actual'].rank(method='dense', ascending=True).astype(int)
        
        overall_actual = test_df['Actual_Rank'].values
        overall_predicted = test_df['Predicted_Rank'].values
        spearman = spearmanr(overall_actual, overall_predicted)[0]
        kendall = kendalltau(overall_actual, overall_predicted)[0]
        mae = np.mean(np.abs(overall_actual - overall_predicted))
        perfect = np.sum(overall_actual == overall_predicted)
        
        fold_results.append({
            'Fold': fold + 1,
            'Test_Seasons': ', '.join(test_seasons_fold),
            'Spearman': spearman,
            'Kendall_Tau': kendall,
            'MAE': mae,
            'Perfect_Matches': perfect,
            'Total_Teams': len(overall_actual)
        })
        
        print(f"Fold {fold + 1} Results:")
        print(f"  Spearman: {spearman:.4f}, Kendall: {kendall:.4f}, MAE: {mae:.2f}, Perfect: {perfect}/{len(overall_actual)}")
    
    # Summary
    cv_results_df = pd.DataFrame(fold_results)
    mean_spearman = cv_results_df['Spearman'].mean()
    std_spearman = cv_results_df['Spearman'].std()
    mean_kendall = cv_results_df['Kendall_Tau'].mean()
    mean_mae = cv_results_df['MAE'].mean()
    
    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")
    print(cv_results_df.to_string(index=False))
    print(f"\nMean Spearman: {mean_spearman:.4f} ± {std_spearman:.4f}")
    print(f"Mean Kendall Tau: {mean_kendall:.4f}")
    print(f"Mean MAE: {mean_mae:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    cv_results_df.to_csv(os.path.join(output_dir, "cross_validation_results.csv"), index=False)
    print(f"\nCross-validation results saved to: {os.path.join(output_dir, 'cross_validation_results.csv')}")
    
    return cv_results_df

    
if __name__ == "__main__":
    # Run main training and prediction
    results = train_and_evaluate_2025_only("Preprocessing/Preprocessed Data")
    print(results.head())
    
    # Run cross-validation
    print("\n" + "="*70)
    print("Running Cross-Validation...")
    print("="*70)
    cv_results = cross_validate_model("Preprocessing/Preprocessed Data", n_folds=5)