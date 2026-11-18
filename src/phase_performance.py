import pandas as pd

data_dir = "/Users/monish_006/Documents/RR/Datapack"
df_teams = pd.read_csv(f"{data_dir}/teams.csv")
df_ball_by_ball = pd.read_csv(
    f"{data_dir}/ball_by_ball.csv",
    usecols=[
        "overid",
        "batting_team_id",
        "bowling_team_id",
        "striker_id",
        "striker_name",
        "bowler_id",
        "bowler_name",
        "runs",
        "is_wicket",
        "dismissal_type",
    ],
)

df_ball_by_ball["phase"] = pd.cut(
    df_ball_by_ball["overid"],
    bins=[0, 6, 15, 20],
    labels=["Powerplay (1-6)", "Middle (7-15)", "Slog (16-20)"],
)
df_ball_by_ball = df_ball_by_ball[df_ball_by_ball["phase"].notna()]
team_names = df_teams[["team_id", "team_name"]].drop_duplicates()

player_runs = (
    df_ball_by_ball.groupby(
        ["phase", "batting_team_id", "striker_id", "striker_name"],
        as_index=False,
        observed=True,
    )["runs"]
    .sum()
    .rename(
        columns={
            "batting_team_id": "team_id",
            "striker_name": "player_name",
            "runs": "total_runs",
        }
    )
    .merge(team_names, on="team_id", how="left", validate="many_to_one")
)
phase_team_high_runs = player_runs.groupby(
    ["phase", "team_id"], observed=True
)["total_runs"].transform("max")
top_batsmen = player_runs.loc[
    player_runs["total_runs"].eq(phase_team_high_runs),
    ["phase", "team_name", "player_name", "total_runs"],
].sort_values(["phase", "team_name", "total_runs"], ascending=[True, True, False])

print("Highest run-scorer(s) for each team and phase:")
print(top_batsmen.to_string(index=False))

credited_dismissals = ["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"]
credited_wickets = df_ball_by_ball[
    df_ball_by_ball["is_wicket"]
    & df_ball_by_ball["dismissal_type"].isin(credited_dismissals)
]
bowler_keys = ["phase", "bowling_team_id", "bowler_id", "bowler_name"]

# Include bowlers who took no wickets in a phase, so a zero-wicket leader is
# still reported when nobody took a credited wicket.
bowlers_in_phase = (
    df_ball_by_ball.groupby(bowler_keys, as_index=False, observed=True)
    .size()
    .drop(columns="size")
)
wickets_by_bowler = (
    credited_wickets.groupby(bowler_keys, as_index=False, observed=True)
    .size()
    .rename(columns={"size": "total_wickets"})
)
bowler_wickets = (
    bowlers_in_phase.merge(
        wickets_by_bowler, on=bowler_keys, how="left", validate="one_to_one"
    )
    .fillna({"total_wickets": 0})
    .rename(columns={"bowling_team_id": "team_id"})
    .merge(team_names, on="team_id", how="left", validate="many_to_one")
)
bowler_wickets["total_wickets"] = bowler_wickets["total_wickets"].astype(int)
phase_team_high_wickets = bowler_wickets.groupby(
    ["phase", "team_id"], observed=True
)["total_wickets"].transform("max")
top_bowlers = bowler_wickets.loc[
    bowler_wickets["total_wickets"].eq(phase_team_high_wickets),
    ["phase", "team_name", "bowler_name", "total_wickets"],
].sort_values(
    ["phase", "team_name", "total_wickets"], ascending=[True, True, False]
)

print("\nHighest wicket-taker(s) for each team and phase:")
print(top_bowlers.to_string(index=False))
