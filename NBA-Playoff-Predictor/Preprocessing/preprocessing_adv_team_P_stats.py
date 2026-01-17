import pandas as pd
import os

# create new directory to store data
output_dir = "Preprocessing\\Preprocessed Data\\Actual Playoff Team Stats"

for year in range(2003, 2024): # modify this to change what files to filter
    season = f"{year}-{str(year+1)[-2:]}"  # formats as "2003-04", "2004-05", etc.
    input_file = f"Preprocessing/Raw Data/Actual Playoff Stats Raw Data/{season}.xlsx"  # construct file path
    output_file = f"{output_dir}/{season}__playoff_actual_team_stats.xlsx"  # output file with "_playoff_actual_team_stats" suffix

    print(f"Processing {season}...")

    # read dataset to dataframe
    actual_team_stats_df = pd.read_excel(input_file)

    #drop league average rows
    actual_team_stats_df = actual_team_stats_df.iloc[:-1]
    


    print(f"Preprocessing successful for {season}\n")

    # save to new subdirectory "Actual Playoff Stats"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        actual_team_stats_df.to_excel(writer, index=False)  # save updated actual_team_stats sheet

    print(f"Saved preprocessed file: {output_file}\n")