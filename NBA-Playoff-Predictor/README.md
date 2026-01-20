# NBA Playoff Predictor

A deep learning system that predicts NBA playoff team rankings using a hybrid neural network architecture. The model combines player statistics and team-level features to forecast playoff seeding for NBA teams.

## 🎯 Purpose

This project predicts the final playoff rankings of NBA teams by analyzing:
- **Player Statistics**: Individual player performance metrics from the regular season
- **Team-Level Features**: Aggregated team statistics and efficiency metrics
- **Historical Patterns**: Training data spans 21 seasons (2003-04 to 2023-24)

The model uses an ensemble of hybrid neural networks trained on historical data to predict playoff rankings for the current season (2024-25).

## 🏗️ Project Structure

```
NBA-Playoff-Predictor/
├── Model/
│   ├── Future Prediction/
│   │   └── hybrid_model_future_pred.py    # Main prediction model
│   └── hybrid_model.py                     # Historical evaluation model
├── Preprocessing/
│   ├── Future Prediction/
│   │   └── preprocessing_player_stats_R_prediction.py  # Preprocess player stats for predictions
│   ├── Preprocessed Data/
│   │   ├── Actual Playoff Team Stats/      # Historical playoff rankings (2003-24)
│   │   └── Player Stats Regular and Playoff/  # Preprocessed player data (2003-25)
│   ├── Raw Data/
│   │   ├── Actual Playoff Stats Raw Data/  # Raw playoff ranking files
│   │   └── Player Stats Regular and Playoff Raw Data/  # Raw player statistics
│   ├── preprocessing_adv_team_P_stats.py   # Preprocess advanced team playoff stats
│   └── preprocessing_player_stats_R_and_P.py  # Preprocess player regular/playoff stats
├── Results/
│   ├── HYBRID_2024-25_predictions.csv      # Current season predictions
│   ├── Complete_Rankings.csv               # Historical predictions
│   ├── test_set_metrics.csv                # Test set performance metrics
│   ├── cross_validation_results.csv        # Cross-validation results
│   ├── hyperparameter_tuning_results.csv   # Hyperparameter search results
│   ├── feature_importance_analysis.csv     # Feature importance rankings
│   ├── overall_accuracy_metrics.csv        # Overall performance metrics
│   ├── seasonal_accuracy_metrics.csv       # Per-season performance metrics
│   ├── metrics.py                          # Metrics calculation utilities
│   └── visualize.py                        # Visualization utilities
├── analyze_feature_importance.py           # Feature importance analysis script
├── hyperparameter_tuning.py                # Hyperparameter optimization script
├── main.py                                 # Entry point (placeholder)
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## 🧠 Model Architecture

### Hybrid Neural Network

The model uses a **dual-pathway architecture** that processes two types of inputs:

1. **Player Pathway** (1D CNN with Attention):
   - Processes individual player statistics for each team
   - Uses 1D convolutional layers to extract features from player arrays
   - Implements an attention mechanism to weight important players
   - Outputs a 128-dimensional representation

2. **Team Pathway** (Fully Connected Network):
   - Processes aggregated team-level features (20 features)
   - Includes efficiency metrics, derived features, and temporal information
   - Two-layer deep network with batch normalization and dropout
   - Outputs a 128-dimensional representation

3. **Combined Network**:
   - Concatenates player and team representations (256 dimensions)
   - Fully connected layers with dropout regularization
   - Outputs a single prediction value (playoff rank)

### Key Features

- **Ensemble Learning**: Trains 20 models and averages predictions for stability
- **Rank-Aware Loss**: Custom loss function combining MSE and Spearman rank correlation
- **Attention Mechanism**: Dynamically weights the importance of players within a team
- **Regularization**: Dropout, batch normalization, and weight decay to prevent overfitting
- **Early Stopping**: Prevents overfitting by monitoring validation performance
- **Learning Rate Scheduling**: Adapts learning rate during training

## 📊 Features

### Team-Level Features (20 total):
1. **Efficiency Metrics**: `eFG%_wt`, `FT%_wt`, `TS%_wt`
2. **Derived Features**: `PTS_per_game`, `AST_TOV_ratio`, `TRB_per_min`, `MP_per_game`
3. **Top Player Stats**: `PTS_top5`, `AST_top5`
4. **Defensive Metrics**: `Defensive_activity`, `Defensive_per_game`
5. **Team Chemistry**: `PTS_std`, `AST_std`
6. **Aggregated Stats**: `PTS_sum`, `AST_sum`, `TRB_sum`, `STL_sum`, `FT%_wt`
7. **Temporal Feature**: `Years_since_2003`

### Player Features (12 per player):
- Games played, Minutes, Field Goal %, 3-Point %, Effective FG %, Free Throw %
- Total Rebounds, Assists, Steals, Blocks, Turnovers, Personal Fouls, Points

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Tmaku18/NBA-Playoff-Predictor.git
cd NBA-Playoff-Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Make Predictions for 2024-25 Season

Run the main prediction script:
```bash
python "Model/Future Prediction/hybrid_model_future_pred.py"
```

This will:
- Load historical player and playoff data
- Train an ensemble of 20 models using 80/20 train/test split
- Evaluate performance on held-out test set
- Generate predictions for 2024-25 playoff rankings
- Save results to `Results/HYBRID_2024-25_predictions.csv`

#### Hyperparameter Tuning

Optimize model hyperparameters:
```bash
python hyperparameter_tuning.py
```

This tests different combinations of:
- Dropout rates: [0.1, 0.15, 0.2]
- Learning rates: [0.0001, 0.0003, 0.0005]
- Batch sizes: [4, 8]
- Weight decay: [0.0001, 0.0005, 0.001]

Results are saved to `Results/hyperparameter_tuning_results.csv`.

#### Feature Importance Analysis

Analyze which features contribute most to predictions:
```bash
python analyze_feature_importance.py
```

Results are saved to `Results/feature_importance_analysis.csv`.

#### Preprocess Data

If you need to regenerate preprocessed data:
```bash
# For player statistics
python "Preprocessing/Future Prediction/preprocessing_player_stats_R_prediction.py"

# For historical player stats
python "Preprocessing/preprocessing_player_stats_R_and_P.py"

# For advanced team playoff stats
python "Preprocessing/preprocessing_adv_team_P_stats.py"
```

## 📈 Performance Metrics

The model is evaluated using multiple metrics:

- **Spearman Rank Correlation**: Measures how well predicted rankings match actual rankings
- **Kendall's Tau**: Alternative rank correlation metric
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and actual ranks
- **Perfect Matches**: Percentage of teams with exactly correct rank predictions

### Current Performance

- **Cross-Validation**: Mean Spearman 0.3147 ± 0.0316 (5-fold CV)
- **Test Set**: Spearman 0.2889, Perfect Matches 12.0%
- **Best Hyperparameters**: Dropout=0.1, Learning Rate=0.0005, Batch Size=8, Weight Decay=0.0005

## 🔧 Model Configuration

### Training Parameters

- **Ensemble Size**: 20 models
- **Epochs**: 75 (with early stopping)
- **Batch Size**: 8
- **Learning Rate**: 0.0005 (with ReduceLROnPlateau scheduler)
- **Dropout Rate**: 0.1
- **Weight Decay**: 0.0005
- **Loss Function**: RankAwareLoss (70% MSE, 30% rank correlation)

### Data Split

- **Train/Test Split**: 80/20 by seasons (not random samples)
- **Train/Validation Split**: 80/20 within training data
- **Cross-Validation**: 5-fold cross-validation across historical seasons

## 📝 Data Sources

- **Player Statistics**: Regular season player data from 2003-04 to 2024-25
- **Playoff Rankings**: Actual playoff seeding from 2003-04 to 2023-24
- Data files are stored in Excel format (`.xlsx`) in the `Preprocessing/` directories

## 🛠️ Development

### Key Scripts

- `hybrid_model_future_pred.py`: Main model implementation with training, evaluation, and prediction
- `hyperparameter_tuning.py`: Grid search for optimal hyperparameters
- `analyze_feature_importance.py`: Permutation importance analysis
- `preprocessing_*.py`: Data preprocessing and feature engineering scripts

### Model Improvements

Recent improvements include:
- Removed noisy features with negative importance (TOV_sum, MP_sum, etc.)
- Implemented rank-aware loss function for better ranking accuracy
- Added attention mechanism for player weighting
- Expanded hyperparameter search space
- Increased ensemble size for better stability
- Feature engineering (efficiency metrics, derived features, temporal features)

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

**Tmaku18**
- GitHub: [@Tmaku18](https://github.com/Tmaku18)

## 🙏 Acknowledgments

- NBA data sources for historical statistics
- PyTorch team for the deep learning framework
- Scikit-learn for preprocessing utilities

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: Predictions are based on historical patterns and should be interpreted with appropriate context. Actual playoff outcomes depend on many factors beyond statistical analysis.
