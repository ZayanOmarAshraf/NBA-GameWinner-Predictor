import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

# Load dataset
df = pd.read_csv(r'/Users/omar/Documents/pythontest/2023_nba_player_stats.csv', encoding='latin1')

# Aggregate Data by Team
team_totals = df.groupby('Team').sum()

# Compute Advanced Metrics
team_totals['PPG'] = team_totals['PTS'] / 82  # Points Per Game
team_totals['FG%'] = team_totals['FGM'] / team_totals['FGA']
team_totals['3P%'] = team_totals['3PM'] / team_totals['3PA']
team_totals['TOV/G'] = team_totals['TOV'] / 82  # Turnovers Per Game
team_totals['REB/G'] = (team_totals['OREB'] + team_totals['DREB']) / 82  # Rebounds Per Game
team_totals['OffRtg'] = team_totals['PTS'] / (team_totals['FGA'] + 0.44 * team_totals['FTA'])  # Offensive Rating
team_totals['DefRtg'] = team_totals['STL'] - team_totals['BLK']  # Defensive Impact Proxy
team_totals['NetRtg'] = team_totals['OffRtg'] - team_totals['DefRtg']  # Net Rating

# Normalize data using Min-Max Scaling
stats_to_normalize = ['PPG', 'FG%', '3P%', 'TOV/G', 'REB/G', 'OffRtg', 'DefRtg', 'NetRtg']
for col in stats_to_normalize:
    team_totals[col] = (team_totals[col] - team_totals[col].min()) / (team_totals[col].max() - team_totals[col].min())

# Get User Input
team1 = input("Enter first team (3-letter abbreviation): ").upper()
team2 = input("Enter second team (3-letter abbreviation): ").upper()

if team1 not in team_totals.index or team2 not in team_totals.index:
    print("One or both teams not found in dataset.")
else:
    # Extract feature vectors
    team1_stats = team_totals.loc[team1, stats_to_normalize].values
    team2_stats = team_totals.loc[team2, stats_to_normalize].values

    # Compute similarity using Euclidean distance
    distance = euclidean(team1_stats, team2_stats)

    # Apply Sigmoid Probability Scaling
    team1_win_prob = 1 / (1 + np.exp(distance - 1))  # Sigmoid for smooth probability
    team2_win_prob = 1 - team1_win_prob

    # Display Results
    print(f"\n🏀 **Predicted Win Probability:**")
    print(f"{team1}: {team1_win_prob:.2%}")
    print(f"{team2}: {team2_win_prob:.2%}")

    # Declare Winner
    winner = team1 if team1_win_prob > team2_win_prob else team2
    print(f"\n🎯 **Predicted Winner: {winner}** 🎯")
