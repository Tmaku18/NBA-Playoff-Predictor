import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your model output CSV
df = pd.read_csv("Results/HYBRID_complete_rankings.csv")
df['Season'] = df['Season'].astype(str)
df['Actual_Rank'] = pd.to_numeric(df['Actual_Rank'])
df['Predicted_Rank'] = pd.to_numeric(df['Predicted_Rank'])

# Best and worst seasons by MAE from external results
seasons = ['2023-24']

seasons = seasons

# Plot for each season
for season in seasons:
    season_df = df[df['Season'].str.contains(season)]

    if season_df.empty:
        print(f"No data found for season: {season}")
        continue

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=season_df, x="Actual_Rank", y="Predicted_Rank", hue="Team", s=200)
    plt.plot([1, season_df['Actual_Rank'].max()], [1, season_df['Actual_Rank'].max()], '--', color='gray', label='Perfect Prediction')
    plt.xlabel("Actual Rank", fontsize=20)
    plt.ylabel("Predicted Rank", fontsize=20)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
