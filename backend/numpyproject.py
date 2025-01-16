import numpy as np

import pandas as pd

import csv

df = pd.read_csv(r'/Users/omar/Documents/pythontest/2023_nba_player_stats.csv', encoding = 'latin1')


team_totals = df.groupby('Team').sum()
team_totals = team_totals.drop(['PName','POS', 'GP', 'W', 'L', 'FG%', '3P%', 'FT%'], axis = 1)
team_totals['FG%'] = team_totals['FGM']/team_totals['FGA']
team_totals['3P%'] = team_totals['3PM']/team_totals['3PA']
team_totals['FT%'] = team_totals['FTM']/team_totals['FTA']
team_totals['PPG'] = team_totals['PTS']/82


# Ask the user for a team abbreviation
team1 = input("Please give me a team name (use 3-letter city abbreviation): ").upper()


if team1 in team_totals.index:
    team1_ppg = team_totals.loc[team1, 'PPG']
    team1_plus_minus = team_totals.loc[team1, '+/-']
    team1_FG_percentage = team_totals.loc[team1, 'FG%']
else:
    print(f"Team {team1} not found in the dataset.")

team2 = input("Please give me a team name (use 3-letter city abbreviation): ").upper()


if team2 in team_totals.index:
    team2_ppg = team_totals.loc[team2, 'PPG']
    team2_plus_minus = team_totals.loc[team2, '+/-']
    team2_FG_percentage = team_totals.loc[team2, 'FG%']
else:
    print(f"Team {team2} not found in the dataset.")

#Create Variable y which represents team1 chance of winning
x = 0

#Create Variable y which represents team1 chance of winning
y = 0

if team1_ppg>team2_ppg:
    x += 2
else:
    y +=2

if team1_plus_minus>team2_plus_minus:
    x += .5
else:
    y +=.5

if team1_FG_percentage>team2_FG_percentage:
    x += 1.75
else:
    y +=1.75

if x>y:
    print(f"I believe that {team1} will beat {team2}")
else:
    print(f"I believe that {team2} will beat {team1}")



