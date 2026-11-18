import pandas as pd

data_dir = "/Users/monish_006/Documents/RR/Datapack"
df_teams = pd.read_csv(f"{data_dir}/teams.csv")
df_ball_by_ball = pd.read_csv(f"{data_dir}/ball_by_ball.csv")

# Sum each batter's runs for their team.
player_runs = (
    df_ball_by_ball.groupby(
        ["batting_team_id", "striker_id", "striker_name"], as_index=False
    )["runs"]
    .sum()
    .rename(
        columns={
            "batting_team_id": "team_id",
            "striker_name": "player_name",
            "runs": "total_runs",
        }
    )
)

# Attach team names, then keep every player tied for the team-high.
team_names = df_teams[["team_id", "team_name"]].drop_duplicates()
player_runs = player_runs.merge(
    team_names, on="team_id", how="left", validate="many_to_one"
)
team_high = player_runs.groupby("team_id")["total_runs"].transform("max")
top_scorers = player_runs.loc[
    player_runs["total_runs"].eq(team_high),
    ["team_name", "player_name", "total_runs"],
].sort_values(["team_name", "total_runs"], ascending=[True, False])

print(top_scorers.to_string(index=False))

# Count wickets credited to bowlers; run-outs and other uncredited dismissals
# are excluded.
credited_dismissals = ["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"]
wicket_deliveries = df_ball_by_ball[
    df_ball_by_ball["is_wicket"]
    & df_ball_by_ball["dismissal_type"].isin(credited_dismissals)
]
bowler_wickets = (
    wicket_deliveries.groupby(
        ["bowling_team_id", "bowler_id", "bowler_name"], as_index=False
    )
    .size()
    .rename(
        columns={
            "bowling_team_id": "team_id",
            "size": "total_wickets",
        }
    )
    .merge(team_names, on="team_id", how="left", validate="many_to_one")
)

team_wicket_high = bowler_wickets.groupby("team_id")["total_wickets"].transform(
    "max"
)
top_wicket_takers = bowler_wickets.loc[
    bowler_wickets["total_wickets"].eq(team_wicket_high),
    ["team_name", "bowler_name", "total_wickets"],
].sort_values(["team_name", "total_wickets"], ascending=[True, False])

print("\nHighest wicket-taker(s) for each team:")
print(top_wicket_takers.to_string(index=False))

