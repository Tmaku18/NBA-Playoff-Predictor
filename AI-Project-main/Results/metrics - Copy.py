import pandas as pd
from scipy.stats import spearmanr, kendalltau
import numpy as np
import os

def calculate_ranking_accuracy(csv_path):
    # load the results
    df = pd.read_csv(csv_path)
    
    # initialize results storage
    season_metrics = []
    overall_actual = []
    overall_predicted = []
    
    # calculate metrics per season
    for season in df['Season'].unique():
        season_df = df[df['Season'] == season]
        
        # get rankings
        actual = season_df['Actual_Rank'].values
        predicted = season_df['Predicted_Rank'].values
        
        # store for overall calculation
        overall_actual.extend(actual)
        overall_predicted.extend(predicted)
        
        # calculate metrics
        spearman_corr, spearman_p = spearmanr(actual, predicted)
        kendall_corr, kendall_p = kendalltau(actual, predicted)
        mae = np.mean(np.abs(actual - predicted))
        perfect_matches = np.sum(actual == predicted)
        
        season_metrics.append({
            'Season': season,
            'Spearman_Correlation': spearman_corr,
            'Kendall_Tau': kendall_corr,
            'MAE': mae,
            'Perfect_Matches': perfect_matches,
            'Total_Teams': len(actual)
        })
    
    # calculate overall metrics
    overall_spearman = spearmanr(overall_actual, overall_predicted)[0]
    overall_kendall = kendalltau(overall_actual, overall_predicted)[0]
    overall_mae = np.mean(np.abs(np.array(overall_actual) - np.array(overall_predicted)))
    overall_perfect = np.sum(np.array(overall_actual) == np.array(overall_predicted))
    
    #cConvert to dfs
    season_results = pd.DataFrame(season_metrics)
    overall_results = pd.DataFrame({
        'Metric': ['Spearman_Correlation', 'Kendall_Tau', 'MAE', 'Perfect_Matches', 'Total_Teams'],
        'Value': [overall_spearman, overall_kendall, overall_mae, overall_perfect, len(overall_actual)]
    })
    
    return season_results, overall_results

def main():
    # path to results CSV (matches output from hybrid_model.py)
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Complete_Rankings.csv")
    
    # calculate accuracy metrics
    season_results, overall_results = calculate_ranking_accuracy(csv_path)
    
    # results
    print("=== Seasonal Accuracy Metrics ===")
    print(season_results.to_string(index=False))
    
    print("\n=== Overall Accuracy Metrics ===")
    print(overall_results.to_string(index=False))
    
    # save results to new CSV
    season_results.to_csv(os.path.join(script_dir, "seasonal_accuracy_metrics.csv"), index=False)
    overall_results.to_csv(os.path.join(script_dir, "overall_accuracy_metrics.csv"), index=False)
    print("\nMetrics saved to CSV files")

if __name__ == "__main__":
    main()