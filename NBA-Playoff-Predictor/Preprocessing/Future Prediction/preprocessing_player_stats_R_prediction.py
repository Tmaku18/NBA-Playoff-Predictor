import pandas as pd
import os

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

# create new directory to store data
output_dir = "Preprocessing\\Preprocessed Data\\Player Stats Regular and Playoff"
os.makedirs(output_dir, exist_ok=True)

# manual list of 2024–25 playoff teams (must match TEAM_NAME_MAP mappings)
playoff_teams_2025 = [
    # Western Conference
    "Oklahoma City Thunder", "Houston Rockets", "Los Angeles Lakers", "Denver Nuggets",
    "Los Angeles Clippers", "Minnesota Timberwolves", "Golden State Warriors", "Memphis Grizzlies",
    "Sacramento Kings", "Dallas Mavericks",
    # Eastern Conference
    "Cleveland Cavaliers", "Boston Celtics", "New York Knicks", "Indiana Pacers",
    "Milwaukee Bucks", "Detroit Pistons", "Orlando Magic", "Atlanta Hawks",
    "Chicago Bulls", "Miami Heat"
]

# loop through only 2024–25
for year in range(2024, 2025):
    season = f"{year}-{str(year+1)[-2:]}"
    input_file = f"Preprocessing/Raw Data/Player Stats Regular and Playoff Raw Data/{season}.xlsx"
    output_file = f"{output_dir}/{season}_filtered.xlsx"

    print(f"Processing {season}...")

    # read regular sheet only
    regular_df = pd.read_excel(input_file, sheet_name="Regular")

    # drop league average row
    regular_df = regular_df.iloc[:-1]

    # map team names to standard
    regular_df['Team'] = regular_df['Team'].map(TEAM_NAME_MAP).fillna(regular_df['Team'])

    # filter by playoff teams
    if season == "2024-25":
        regular_df = regular_df[regular_df['Team'].isin(playoff_teams_2025)]

    # remove "awards" and "player additional" columns if they exist
    regular_df = regular_df.drop(columns=['Awards', 'Player Additional'], errors='ignore')

    # save to new subdirectory
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        regular_df.to_excel(writer, sheet_name="Regular", index=False)

    print(f"Saved preprocessed file: {output_file}\n")
