import pandas as pd

# Load the Excel file
df_teams = pd.read_csv(r"/Users/monish_006/Documents/RR/Datapack/teams.csv")

# Get england teams
england_teams = df_teams[df_teams["team_name"] == "England"]

# Get england team ID
team_id = england_teams.loc[0].team_id

# Get all the England players
df_playing_eleven = pd.read_csv(r"/Users/monish_006/Documents/RR/Datapack/playing_eleven.csv")
player_id_eng = df_playing_eleven[df_playing_eleven["team_id"] == team_id]

print(player_id_eng.player_id.unique())
print(len(player_id_eng.player_id.unique()))

df_comp_player = pd.read_csv(r"/Users/monish_006/Documents/RR/Datapack/competition_team_player_mapping.csv")
player_name_eng = df_comp_player[df_comp_player["player_id"].isin(player_id_eng["player_id"])]
print(player_name_eng.player_name.unique())

df_ball_by_ball = pd.read_csv(r"/Users/monish_006/Documents/RR/Datapack/ball_by_ball.csv.csv")
print(df_ball_by_ball.strik)

