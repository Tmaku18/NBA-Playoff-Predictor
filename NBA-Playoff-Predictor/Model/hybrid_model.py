import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
        agg_dict = {
            'Team': team,
            'Season': season,
            'MP_sum': group['MP'].sum(),
            'FG%_wt': weighted_avg(group, 'FG%'),
            '3P%_wt': weighted_avg(group, '3P%'),
            'eFG%_wt': weighted_avg(group, 'eFG%'),
            'FT%_wt': weighted_avg(group, 'FT%'),
            'TRB_sum': group['TRB'].sum(),
            'AST_sum': group['AST'].sum(),
            'AST_top3': group['AST'].nlargest(3).mean() if len(group) >= 3 else 0,
            'STL_sum': group['STL'].sum(),
            'BLK_sum': group['BLK'].sum(),
            'TOV_sum': group['TOV'].sum(),
            'PF_sum': group['PF'].sum(),
            'PTS_sum': group['PTS'].sum(),
            'PTS_top3': group['PTS'].nlargest(3).mean() if len(group) >= 3 else 0,
            'G_mean': group['G'].mean(),
            'G_std': group['G'].std()
        }
        team_agg_list.append(agg_dict)
    
    team_agg = pd.DataFrame(team_agg_list)
    
    # flatten multi-index columns
    team_agg.columns = [
        'Team', 'Season', 'MP_sum', 'FG%_wt', '3P%_wt', 'eFG%_wt', 'FT%_wt',
        'TRB_sum', 'AST_sum', 'AST_top3', 'STL_sum', 'BLK_sum', 'TOV_sum',
        'PF_sum', 'PTS_sum', 'PTS_top3', 'G_mean', 'G_std'
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
    assert not np.isnan(merged['Playoff_Rank'].values).any(), "NaN values in targets"
    
    if fit_scalers:
        return player_arrays, team_features, merged['Playoff_Rank'].values, merged, player_scaler, team_scaler
    return player_arrays, team_features, merged['Playoff_Rank'].values, merged
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
    def __init__(self, n_players, n_player_features=12, n_team_features=16):
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
        # team pathway (fully connected)
        self.team_net = nn.Sequential(
            nn.Linear(n_team_features, 128),
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
def train_and_evaluate(base_path, output_dir="Results"):
    # load data
    player_df = load_player_stats(base_path)
    playoff_df = load_playoff_stats(base_path)
    
    # preprocess without scaling (to avoid data leakage)
    X_player, X_team, y, meta = preprocess_data(player_df, playoff_df, fit_scalers=False)
    
    # group by season for proper evaluation
    seasons = sorted(meta['Season'].unique())
    all_results = []
    skipped_seasons = []
    
    print(f"\nTotal seasons found: {len(seasons)}")
    print(f"Seasons: {seasons}")
    
    for season in seasons:
        print(f"\n=== Processing season {season} ===")
        season_mask = meta['Season'] == season
        season_indices = np.where(season_mask)[0]
        
        # skip seasons with too few teams
        if len(season_indices) < 4:
            print(f"Skipping season {season} - only {len(season_indices)} teams available")
            skipped_seasons.append(season)
            continue

        # Get season data (unscaled)
        season_player = X_player[season_indices]
        season_team = X_team[season_indices]
        season_y = y[season_indices]
        season_meta = meta.iloc[season_indices].reset_index(drop=True)
        
        # Split indices for train/test
        n_train = int(0.8 * len(season_indices))
        indices = np.arange(len(season_indices))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        # Split data
        train_player = season_player[train_indices]
        train_team = season_team[train_indices]
        train_y = season_y[train_indices]
        
        test_player = season_player[test_indices]
        test_team = season_team[test_indices]
        test_y = season_y[test_indices]
        
        # Fit scalers ONLY on training data (fix data leakage)
        player_scaler = StandardScaler()
        train_player_reshaped = train_player.reshape(-1, train_player.shape[-1])
        train_player_reshaped = np.nan_to_num(train_player_reshaped, nan=0)
        player_scaler.fit(train_player_reshaped)
        
        team_scaler = StandardScaler()
        train_team_clean = np.nan_to_num(train_team, nan=0)
        team_scaler.fit(train_team_clean)
        
        # Transform both train and test with scalers fitted on train only
        train_player_scaled = player_scaler.transform(train_player_reshaped).reshape(train_player.shape)
        test_player_reshaped = test_player.reshape(-1, test_player.shape[-1])
        test_player_reshaped = np.nan_to_num(test_player_reshaped, nan=0)
        test_player_scaled = player_scaler.transform(test_player_reshaped).reshape(test_player.shape)
        
        train_team_scaled = team_scaler.transform(train_team_clean)
        test_team_clean = np.nan_to_num(test_team, nan=0)
        test_team_scaled = team_scaler.transform(test_team_clean)
        
        # Create datasets
        train_dataset = NBADataset(train_player_scaled, train_team_scaled, train_y, 
                                   original_indices=train_indices)
        test_dataset = NBADataset(test_player_scaled, test_team_scaled, test_y, 
                                  original_indices=test_indices)
        
        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # initialize model
        model = HybridNBAModel(n_players=X_player.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.MSELoss()
        
        # training with validation
        best_test_loss = float('inf')
        model.train()
        for epoch in range(50):
            # Training
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch['player_stats'], batch['team_features'])
                loss = criterion(outputs, batch['target'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation on test set
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    outputs = model(batch['player_stats'], batch['team_features'])
                    loss = criterion(outputs, batch['target'])
                    test_loss += loss.item()
            
            train_loss = epoch_loss / len(train_loader)
            test_loss = test_loss / len(test_loader)
            
            if test_loss < best_test_loss:
                best_test_loss = test_loss
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        
        # Evaluate on full season for rankings (but report test metrics separately)
        # Scale full season data using train scalers
        full_player_reshaped = season_player.reshape(-1, season_player.shape[-1])
        full_player_reshaped = np.nan_to_num(full_player_reshaped, nan=0)
        full_player_scaled = player_scaler.transform(full_player_reshaped).reshape(season_player.shape)
        
        full_team_clean = np.nan_to_num(season_team, nan=0)
        full_team_scaled = team_scaler.transform(full_team_clean)
        
        full_dataset = NBADataset(full_player_scaled, full_team_scaled, season_y, 
                                  original_indices=np.arange(len(season_indices)))
        
        model.eval()
        with torch.no_grad():
            full_player_tensor = torch.stack([full_dataset[i]['player_stats'] for i in range(len(full_dataset))])
            full_team_tensor = torch.stack([full_dataset[i]['team_features'] for i in range(len(full_dataset))])
            full_predictions = model(full_player_tensor, full_team_tensor).numpy()
        
        # create results dataframe
        team_names = season_meta['Team'].tolist()
        actual_ranks = np.argsort(np.argsort(season_y)) + 1
        predicted_ranks = np.argsort(np.argsort(full_predictions)) + 1
        
        season_results = pd.DataFrame({
            'Season': season,
            'Team': team_names,
            'Actual_Rank': actual_ranks,
            'Predicted_Rank': predicted_ranks
        }).sort_values('Predicted_Rank')
        
        print(f"\n=== Season {season} Rankings ===")
        print(season_results[['Predicted_Rank', 'Team', 'Actual_Rank']]
              .rename(columns={'Predicted_Rank': 'Rank'})
              .to_string(index=False))
        print(f"Best Test Loss: {best_test_loss:.4f}")
        
        all_results.append(season_results)
    
    if not all_results:
        error_msg = f"No valid seasons with sufficient data were processed.\n"
        error_msg += f"Total seasons found: {len(seasons)}\n"
        error_msg += f"Seasons skipped (too few teams): {skipped_seasons}\n"
        error_msg += f"Seasons processed: 0\n"
        error_msg += f"Please check your data - each season needs at least 4 teams."
        raise ValueError(error_msg)

    os.makedirs(output_dir, exist_ok=True)

    # Safely concatenate results (handle empty list)
    if len(all_results) == 0:
        raise ValueError("No seasons were successfully processed. Check your data and season filtering logic.")
    
    final_results = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(output_dir, "HYBRID_complete_rankings.csv")
    final_results.to_csv(output_path, 
                       columns=['Season', 'Predicted_Rank', 'Team', 'Actual_Rank'],
                       index=False)
    print(f"\nAll seasons rankings saved to: {output_path}")
    
    return final_results
    
if __name__ == "__main__":
    results = train_and_evaluate("Preprocessing/Preprocessed Data")
    print(results.head(20))