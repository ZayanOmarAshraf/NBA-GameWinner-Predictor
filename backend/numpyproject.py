import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean

# Load dataset
df = pd.read_csv(r'/Users/omar/Documents/pythontest/2023_nba_player_stats.csv', encoding='latin1')

# Feature Engineering
team_totals = df.groupby('Team').sum()
team_totals = team_totals.drop(['PName', 'POS', 'GP', 'W', 'L'], axis=1)

# Compute Advanced Stats
team_totals['FG%'] = team_totals['FGM'] / team_totals['FGA']
team_totals['3P%'] = team_totals['3PM'] / team_totals['3PA']
team_totals['FT%'] = team_totals['FTM'] / team_totals['FTA']
team_totals['PPG'] = team_totals['PTS'] / 82
team_totals['TOV/G'] = team_totals['TOV'] / 82
team_totals['REB/G'] = (team_totals['OREB'] + team_totals['DREB']) / 82
team_totals['OffRtg'] = team_totals['PTS'] / (team_totals['FGA'] + 0.44 * team_totals['FTA'])
team_totals['DefRtg'] = team_totals['STL'] - team_totals['BLK']  # Proxy for defensive impact
team_totals['NetRtg'] = team_totals['OffRtg'] - team_totals['DefRtg']

# Normalize data
scaler = MinMaxScaler()
stats_to_normalize = ['PPG', 'FG%', '3P%', 'TOV/G', 'REB/G', 'OffRtg', 'DefRtg', 'NetRtg']
team_totals[stats_to_normalize] = scaler.fit_transform(team_totals[stats_to_normalize])

# Get team input
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

    # KNN-Like Score Calculation (Lower distance = Higher chance to win)
    team1_win_prob = 1 / (1 + np.exp(distance - 1))  # Sigmoid function for probability
    team2_win_prob = 1 - team1_win_prob

    print(f"\nPredicted Win Probability:")
    print(f"{team1}: {team1_win_prob:.2%}")
    print(f"{team2}: {team2_win_prob:.2%}")

    winner = team1 if team1_win_prob > team2_win_prob else team2
    print(f"🎯 \nPredicted Winner: {winner} 🎯")
