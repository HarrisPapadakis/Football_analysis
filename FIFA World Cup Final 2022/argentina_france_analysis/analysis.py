
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path

project_folder = Path(__file__).resolve().parent


# 1. Φόρτωση του αρχείου JSON
file_path = Path("C:/Users/harri/Desktop/STATS_PROJECT/FIFA World Cup Final 2022/argentina_france_analysis/data/events.json")

with open(file_path, "r", encoding="utf-8") as file:
    events = json.load(file)

# 2. Μετατροπή των events σε DataFrame
df = pd.json_normalize(events)

# 3. Πρώτος έλεγχος δεδομένων
print("Συνολικά events:", len(df))
print("Διαστάσεις πίνακα:", df.shape)

print("\nΔιαθέσιμες στήλες:")
print(df.columns.tolist())

# 4. Προβολή βασικών πεδίων
columns = [
    "type.name",
    "team.name",
    "player.name",
    "minute",
    "second"
]

print("\nΠρώτα 10 events:")
print(df[columns].head(10).to_string(index=False))


# ==========================================
# PART B - BASIC MATCH DATA ANALYSIS
# Argentina vs France | World Cup Final 2022
# ==========================================

# Εξαιρούμε τη διαδικασία των πέναλτι (shootout).
# Κρατάμε κανονική διάρκεια και παράταση.
df_match = df[df["period"].isin([1, 2, 3, 4])].copy()

# 1. SHOTS - Σουτ
shots = df_match[df_match["type.name"] == "Shot"].copy()

# 2. GOALS - Γκολ από events σουτ
shots["is_goal"] = shots["shot.outcome.name"].eq("Goal")

# 3. PASSES - Πάσες
passes = df_match[df_match["type.name"] == "Pass"].copy()

# Στη StatsBomb, όταν δεν υπάρχει pass.outcome.name,
# η πάσα θεωρείται συνήθως επιτυχημένη.
passes["is_completed"] = passes["pass.outcome.name"].isna()

# 4. ΣΥΝΟΨΗ ΑΝΑ ΟΜΑΔΑ
teams = ["Argentina", "France"]

for team in teams:
    team_shots = shots[shots["team.name"] == team]
    team_passes = passes[passes["team.name"] == team]

    total_shots = len(team_shots)
    total_xg = team_shots["shot.statsbomb_xg"].sum()
    total_goals = int(team_shots["is_goal"].sum())

    completed_passes = int(team_passes["is_completed"].sum())
    attempted_passes = len(team_passes)

    print(f"\n{'=' * 35}")
    print(team)
    print(f"{'=' * 35}")
    print("Shots:", total_shots)
    print("xG:", round(total_xg, 2))
    print("Goals:", total_goals)
    print("Passes attempted:", attempted_passes)
    print("Passes completed:", completed_passes)

    if attempted_passes > 0:
        accuracy = completed_passes / attempted_passes * 100
        print("Pass accuracy:", round(accuracy, 1), "%")




# ==========================================
# PART B - POSSESSION & TERRITORIAL METRICS
# ==========================================

# 1. POSSESSION: ποσοστό των καταγεγραμμένων κατοχών
# Αυτό μετρά κατοχές, όχι ποσοστό χρόνου κατοχής.
possessions = ()
# ==========================================
# PROGRESSIVE PASSES & FINAL THIRD ENTRIES
# ==========================================

import numpy as np

# Βοηθητική συνάρτηση για τις συντεταγμένες
def get_x(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return value[0]
    return np.nan


# ==========================================
# ATTACKING DIRECTIONS BY PERIOD
# ==========================================

# +1: επίθεση προς x=120
# -1: επίθεση προς x=0
#
# Οι κατευθύνσεις της πρώτης περιόδου πρέπει να
# επιβεβαιωθούν από τις συντεταγμένες του αγώνα.
# Οι ομάδες αλλάζουν πλευρές στο ημίχρονο και
# στο ημίχρονο της παράτασης.

directions = {
    ("Argentina", 1):  1,
    ("France", 1):    -1,

    ("Argentina", 2): -1,
    ("France", 2):    1,

    ("Argentina", 3): -1,
    ("France", 3):    1,

    ("Argentina", 4):  1,
    ("France", 4):   -1,
}

print("\n--- ATTACKING DIRECTIONS ---")

for key, direction in directions.items():
    target = 120 if direction == 1 else 0
    print(f"{key}: attacking towards x={target}")


# 2. Progressive passes
# Μετράμε μόνο επιτυχημένες πάσες με έγκυρες συντεταγμένες.

passes = df_match[
    (df_match["type.name"] == "Pass")
    & (df_match["pass.outcome.name"].isna())
].copy()

passes["start_x"] = passes["location"].apply(get_x)
passes["end_x"] = passes["pass.end_location"].apply(get_x)

passes["direction"] = passes.apply(
    lambda row: directions.get(
        (row["team.name"], row["period"]), np.nan
    ),
    axis=1
)

# Αφαιρούμε εγγραφές χωρίς έγκυρα δεδομένα.
passes = passes.dropna(
    subset=["start_x", "end_x", "direction"]
).copy()

# Κοινός προσανατολισμός:
# 0 = δική μας εστία, 120 = αντίπαλη εστία.
passes["start_att"] = np.where(
    passes["direction"] == 1,
    passes["start_x"],
    120 - passes["start_x"]
)

passes["end_att"] = np.where(
    passes["direction"] == 1,
    passes["end_x"],
    120 - passes["end_x"]
)

passes["progress"] = (
    passes["end_att"] - passes["start_att"]
)

# Δικός μας λειτουργικός ορισμός progressive pass:
# 30m στο δικό μας μισό,
# 15m όταν διασχίζει τη μεσαία γραμμή,
# 10m στο επιθετικό μισό.

own_half = (
    (passes["start_att"] < 60)
    & (passes["end_att"] < 60)
    & (passes["progress"] >= 30)
)

crossing_halfway = (
    (passes["start_att"] < 60)
    & (passes["end_att"] >= 60)
    & (passes["progress"] >= 15)
)

attacking_half = (
    (passes["start_att"] >= 60)
    & (passes["progress"] >= 10)
)

passes["progressive"] = (
    (passes["progress"] > 0)
    & (
        own_half
        | crossing_halfway
        | attacking_half
    )
)

print("\n--- PROGRESSIVE PASSES ---")

for team in ["Argentina", "France"]:
    count = int(
        passes.loc[
            passes["team.name"] == team,
            "progressive"
        ].sum()
    )
    print(f"{team}: {count}")

# 3. Final third entries: επιτυχημένες πάσες και carries
# 3. Final third entries
# Μετράμε επιτυχημένες πάσες και carries
# που περνούν από έξω προς μέσα στην τελευταία τρίτη.

entries = df_match[
    (
        (df_match["type.name"] == "Pass")
        & (df_match["pass.outcome.name"].isna())
    )
    | (df_match["type.name"] == "Carry")
].copy()

entries["start_x"] = entries["location"].apply(get_x)

entries["end_x"] = entries.apply(
    lambda row: (
        get_x(row["pass.end_location"])
        if row["type.name"] == "Pass"
        else get_x(row["carry.end_location"])
    ),
    axis=1
)

entries["direction"] = entries.apply(
    lambda row: directions.get(
        (row["team.name"], row["period"]), np.nan
    ),
    axis=1
)

# Αφαιρούμε εγγραφές χωρίς έγκυρες συντεταγμένες
# ή γνωστή κατεύθυνση επίθεσης.
entries = entries.dropna(
    subset=["start_x", "end_x", "direction"]
).copy()

# Μετατροπή σε κοινό προσανατολισμό.
entries["start_att"] = np.where(
    entries["direction"] == 1,
    entries["start_x"],
    120 - entries["start_x"]
)

entries["end_att"] = np.where(
    entries["direction"] == 1,
    entries["end_x"],
    120 - entries["end_x"]
)

# Η τελευταία τρίτη αρχίζει στο x=80
# σε γήπεδο μήκους 120 μονάδων.
entries["final_third_entry"] = (
    (entries["start_att"] < 80)
    & (entries["end_att"] >= 80)
)

print("\n--- FINAL THIRD ENTRIES ---")

for team in ["Argentina", "France"]:
    count = int(
        entries.loc[
            entries["team.name"] == team,
            "final_third_entry"
        ].sum()
    )
    print(f"{team}: {count}")


# 4. Visualizations: Shots and Expected Goals (xG)

teams = ["Argentina", "France"]

# Υπολογισμός τελικών ανά ομάδα
shots_by_team = (
    df_match[df_match["type.name"] == "Shot"]
    .groupby("team.name")
    .size()
    .reindex(teams, fill_value=0)
)

# Υπολογισμός συνολικού xG ανά ομάδα
xg_by_team = (
    df_match[df_match["type.name"] == "Shot"]
    .groupby("team.name")["shot.statsbomb_xg"]
    .sum()
    .reindex(teams, fill_value=0)

)

# Γράφημα 1: Τελικές προσπάθειες
plt.figure(figsize=(8, 5))
shots_by_team.plot(kind="bar")
plt.title("Argentina vs France - Shots")
plt.xlabel("Team")
plt.ylabel("Number of Shots")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Γράφημα 2: Expected Goals
plt.figure(figsize=(8, 5))
xg_by_team.plot(kind="bar")
plt.title("Argentina vs France - Expected Goals (xG)")
plt.xlabel("Team")
plt.ylabel("Total xG")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# 5. Visualizations: Progressive Passes and Final Third Entries

teams = ["Argentina", "France"]

# Συγκεντρώνουμε τα αποτελέσματα που υπολογίσαμε προηγουμένως.
progressive_passes_by_team = (
    passes.groupby("team.name")["progressive"]
    .sum()
    .reindex(teams, fill_value=0)
)

final_third_entries_by_team = (
    entries.groupby("team.name")["final_third_entry"]
    .sum()
    .reindex(teams, fill_value=0)
)

comparison = pd.DataFrame({
    "Progressive Passes": progressive_passes_by_team,
    "Final Third Entries": final_third_entries_by_team
})

comparison.plot(kind="bar", figsize=(9, 5))

plt.title("Argentina vs France - Ball Progression")
plt.xlabel("Team")
plt.ylabel("Number of Actions")
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()



# 6. Player Analysis

# Κρατάμε μόνο τις τελικές προσπάθειες
shots = df_match[
    df_match["type.name"] == "Shot"
].copy()

# Υπολογισμός xG ανά παίκτη
player_xg = (
    shots.groupby(["team.name", "player.name"])["shot.statsbomb_xg"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\n--- TOP 10 PLAYERS BY xG ---")
print(player_xg.round(2))

# Γράφημα Top 10 παικτών σε xG
player_xg.sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("Top 10 Players by Expected Goals (xG)")
plt.xlabel("Total xG")
plt.ylabel("Player")
plt.tight_layout()
plt.show()


# Επιτυχημένες πάσες ανά παίκτη
successful_passes = df_match[
    (df_match["type.name"] == "Pass")
    & (df_match["pass.outcome.name"].isna())
    & (df_match["player.name"].notna())
].copy()

player_passes = (
    successful_passes.groupby(
        ["team.name", "player.name"]
    )
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\n--- TOP 10 PLAYERS BY COMPLETED PASSES ---")
print(player_passes)

# Γράφημα Top 10 παικτών σε επιτυχημένες πάσες
player_passes.sort_values().plot(
    kind="barh",
    figsize=(10, 6)
)

plt.title("Top 10 Players by Completed Passes")
plt.xlabel("Completed Passes")
plt.ylabel("Player")
plt.tight_layout()
plt.show()



# 7. Key Passes
# Πάσες που οδηγούν άμεσα σε τελική προσπάθεια.

key_passes = df_match[
    (df_match["type.name"] == "Pass")
    & (df_match["pass.shot_assist"] == True)
    & (df_match["player.name"].notna())
].copy()

key_passes_by_player = (
    key_passes.groupby(["team.name", "player.name"])
    .size()
    .sort_values(ascending=False)
)

print("\n--- KEY PASSES BY PLAYER ---")
print(key_passes_by_player.head(10))

if not key_passes_by_player.empty:
    key_passes_by_player.sort_values().plot(
        kind="barh",
        figsize=(10, 6)
    )

    plt.title("Top Players by Key Passes")
    plt.xlabel("Number of Key Passes")
    plt.ylabel("Player")
    plt.tight_layout()
    plt.show()
else:
    print("No key passes found in the selected data.")



# 8. Export results for portfolio

# Συνοπτικά στατιστικά ανά ομάδα
teams = ["Argentina", "France"]

team_summary = pd.DataFrame(index=teams)

team_summary["Shots"] = (
    df_match[df_match["type.name"] == "Shot"]
    .groupby("team.name")
    .size()
    .reindex(teams, fill_value=0)
)

team_summary["xG"] = (
    df_match[df_match["type.name"] == "Shot"]
    .groupby("team.name")["shot.statsbomb_xg"]
    .sum()
    .reindex(teams, fill_value=0)
)

team_summary["Passes Attempted"] = (
    df_match[df_match["type.name"] == "Pass"]
    .groupby("team.name")
    .size()
    .reindex(teams, fill_value=0)
)

team_summary["Passes Completed"] = (
    df_match[
        (df_match["type.name"] == "Pass")
        & (df_match["pass.outcome.name"].isna())
    ]
    .groupby("team.name")
    .size()
    .reindex(teams, fill_value=0)
)

team_summary["Pass Accuracy (%)"] = (
    team_summary["Passes Completed"]
    / team_summary["Passes Attempted"] * 100
).round(1)

team_summary["Progressive Passes"] = (
    passes.groupby("team.name")["progressive"]
    .sum()
    .reindex(teams, fill_value=0)
)

team_summary["Final Third Entries"] = (
    entries.groupby("team.name")["final_third_entry"]
    .sum()
    .reindex(teams, fill_value=0)
)

# Αποθήκευση βασικών στατιστικών ομάδων
team_summary.to_csv(
    project_folder / "team_summary.csv",
    encoding="utf-8-sig"
)

# Αποθήκευση xG ανά παίκτη
player_xg.to_csv(
    project_folder / "player_xg.csv",
    encoding="utf-8-sig"
)

# Αποθήκευση επιτυχημένων πασών ανά παίκτη
player_passes.to_csv(
    project_folder / "player_completed_passes.csv",
    encoding="utf-8-sig"
)

# Αποθήκευση key passes ανά παίκτη
key_passes_by_player.to_csv(
    project_folder / "player_key_passes.csv",
    encoding="utf-8-sig"
)

print("\n--- EXPORT COMPLETED ---")
print("Saved team_summary.csv")
print("Saved player_xg.csv")
print("Saved player_completed_passes.csv")
print("Saved player_key_passes.csv")
print("\n--- TEAM SUMMARY ---")
print(team_summary)

