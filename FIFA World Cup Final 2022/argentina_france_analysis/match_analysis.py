# ==========================================================
# Argentina vs France | FIFA World Cup Final 2022
# Ανάλυση StatsBomb events: Possession, xG, Shots,
# Progressive Passes, Final Third Entries, Pressing, Παίκτες
# ==========================================================
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- 1. ΦΟΡΤΩΣΗ ----------
DEFAULT_PATH = Path(
    "C:/Users/harri/Desktop/STATS_data/FIFA World Cup Final 2022/"
    "argentina_france_analysis/data/events.json"
)
# Μπορείς να δώσεις άλλο path ως όρισμα: python match_analysis.py "path\\to\\events.json"
file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH

with open(file_path, "r", encoding="utf-8") as f:
    events = json.load(f)

df = pd.json_normalize(events)
print("Συνολικά events:", len(df))
print("Διαστάσεις πίνακα:", df.shape)

TEAMS = ["Argentina", "France"]


def other(team):
    return TEAMS[1] if team == TEAMS[0] else TEAMS[0]


def get_x(v):
    return v[0] if isinstance(v, (list, tuple)) and len(v) >= 2 else np.nan


def col(frame, name):
    """Επιστρέφει στήλη ή κενή (NaN) αν λείπει από το αρχείο."""
    return frame[name] if name in frame.columns else pd.Series(np.nan, index=frame.index)


# Κανονική διάρκεια + παράταση (χωρίς shootout, period 5)
dm = df[df["period"].isin([1, 2, 3, 4])].copy()
dm["x"] = dm["location"].apply(get_x)

# Στη StatsBomb κάθε ομάδα επιτίθεται προς x=120 σε όλα τα events,
# άρα ΔΕΝ χρειάζεται εκτίμηση κατεύθυνσης.

# ---------- 2. POSSESSION ----------
print("\n=== POSSESSION ===")
# (α) ποσοστό κατοχών (sequences)
seq = dm.groupby("possession")["possession_team.name"].first().value_counts(normalize=True) * 100
# (β) ποσοστό χρόνου (άθροισμα duration events ανά ομάδα κατοχής)
dur = dm.groupby("possession_team.name")["duration"].sum()
dur = dur / dur.sum() * 100
# (γ) ποσοστό πασών
passes_all = dm[dm["type.name"] == "Pass"].copy()
pshare = passes_all["team.name"].value_counts(normalize=True) * 100
for t in TEAMS:
    print(f"{t}: κατοχές {seq[t]:.1f}% | χρόνος {dur[t]:.1f}% | πάσες {pshare[t]:.1f}%")

# ---------- 3. SHOTS & xG ----------
print("\n=== SHOTS / xG ===")
shots = dm[dm["type.name"] == "Shot"].copy()
shots["xg"] = shots["shot.statsbomb_xg"]
shots["goal"] = shots["shot.outcome.name"].eq("Goal")
shots["on_target"] = shots["shot.outcome.name"].isin(["Goal", "Saved", "Saved To Post"])
shots["is_pen"] = shots["shot.type.name"].eq("Penalty")

for t in TEAMS:
    s = shots[shots["team.name"] == t]
    op = s[~s["is_pen"]]
    print(
        f"{t}: shots {len(s)} | on target {int(s['on_target'].sum())} | goals {int(s['goal'].sum())} | "
        f"xG {s['xg'].sum():.2f} | npxG {op['xg'].sum():.2f} | xG/shot {s['xg'].mean():.3f} | "
        f"big chances (xG>=0.2) {int((s['xg'] >= 0.2).sum())}"
    )

print("\nxG ανά περίοδο (1,2 = κανονική, 3,4 = παράταση):")
print(shots.groupby(["period", "team.name"])["xg"].sum().round(2).unstack().to_string())

print("\nΓκολ:")
print(
    shots[shots["goal"]][["period", "minute", "team.name", "player.name", "shot.type.name", "xg"]]
    .round(2).to_string(index=False)
)

# ---------- 4. PROGRESSIVE PASSES ----------
# Ορισμός: επιτυχημένη πάσα που
#  - μένει στο δικό μας μισό (x<60) και προχωρά >= 30m, ή
#  - διασχίζει τη μεσαία γραμμή και προχωρά >= 15m, ή
#  - ξεκινά στο επιθετικό μισό (x>=60) και προχωρά >= 10m.
p = passes_all[passes_all["pass.outcome.name"].isna()].copy()
p["sx"] = p["location"].apply(get_x)
p["ex"] = p["pass.end_location"].apply(get_x)
p["progress"] = p["ex"] - p["sx"]

own_half = (p["sx"] < 60) & (p["ex"] < 60) & (p["progress"] >= 30)
crossing = (p["sx"] < 60) & (p["ex"] >= 60) & (p["progress"] >= 15)
att_half = (p["sx"] >= 60) & (p["progress"] >= 10)
p["progressive"] = (p["progress"] > 0) & (own_half | crossing | att_half)

print("\n=== PROGRESSIVE PASSES (επιτυχημένες) ===")
for t in TEAMS:
    pt = p[p["team.name"] == t]
    print(f"{t}: {int(pt['progressive'].sum())} από {len(pt)} επιτυχημένες πάσες")

# ---------- 5. FINAL THIRD ENTRIES ----------
# Επιτυχημένη πάσα ή carry που ξεκινά πριν το x=80 και τελειώνει μετά.
e = dm[dm["type.name"].isin(["Pass", "Carry"])].copy()
e["sx"] = e["x"]
e["ex"] = np.where(
    e["type.name"] == "Pass",
    col(e, "pass.end_location").apply(get_x),
    col(e, "carry.end_location").apply(get_x),
)
ok = ((e["type.name"] == "Pass") & e["pass.outcome.name"].isna()) | (e["type.name"] == "Carry")
e = e[ok].copy()
e["entry"] = (e["sx"] < 80) & (e["ex"] >= 80)

print("\n=== FINAL THIRD ENTRIES ===")
for t in TEAMS:
    x = e[(e["team.name"] == t) & e["entry"]]
    print(
        f"{t}: {len(x)} (πάσες {int((x['type.name'] == 'Pass').sum())}, "
        f"carries {int((x['type.name'] == 'Carry').sum())})"
    )

# ---------- 6. PRESSING ----------
# PPDA = πάσες αντιπάλου στο δικό του 60% του γηπέδου (x<=72)
#        / αμυντικές ενέργειες μας στο αντίστοιχο ζώνη (x>=48 στις δικές μας συντεταγμένες).
# Αμυντικές ενέργειες: tackles, interceptions, fouls committed.
# Χαμηλότερο PPDA = πιο έντονο pressing.
print("\n=== PRESSING ===")
pressures = dm[dm["type.name"] == "Pressure"]
recoveries = dm[dm["type.name"] == "Ball Recovery"]
for t in TEAMS:
    o = other(t)
    opp_passes = passes_all[(passes_all["team.name"] == o) & (passes_all["location"].apply(get_x) <= 72)]
    d = dm[(dm["team.name"] == t) & (dm["x"] >= 48)]
    acts = (
        d["type.name"].isin(["Interception", "Foul Committed"])
        | ((d["type.name"] == "Duel") & (col(d, "duel.type.name") == "Tackle"))
    ).sum()
    ppda = len(opp_passes) / max(acts, 1)

    pt = pressures[pressures["team.name"] == t]
    cp = int(col(pt, "counterpress").notna().sum())
    rt = recoveries[recoveries["team.name"] == t]
    print(
        f"{t}: PPDA {ppda:.2f} | pressures {len(pt)} (στο μισό αντιπάλου {int((pt['x'] >= 60).sum())}) | "
        f"counterpress {cp} | recoveries {len(rt)} (στο μισό αντιπάλου {int((rt['x'] >= 60).sum())}) | "
        f"μέσο ύψος recovery x={rt['x'].mean():.1f}"
    )

# ---------- 7. ΠΑΙΚΤΕΣ ----------
print("\n=== ΠΑΙΚΤΕΣ ===")
g = (
    shots.groupby(["team.name", "player.name"])
    .agg(shots=("xg", "size"), xg=("xg", "sum"), goals=("goal", "sum"))
    .reset_index()
    .sort_values("xg", ascending=False)
    .head(6)
    .round(2)
)
print("Κορυφαίοι σε xG:\n", g.to_string(index=False))

print("\nProgressive passes:\n",
      p[p["progressive"]].groupby(["team.name", "player.name"]).size()
      .sort_values(ascending=False).head(6).to_string())

kp = passes_all[col(passes_all, "pass.shot_assist").notna() | col(passes_all, "pass.goal_assist").notna()]
print("\nKey passes:\n",
      kp.groupby(["team.name", "player.name"]).size().sort_values(ascending=False).head(5).to_string())

print("\nPressures:\n",
      pressures.groupby(["team.name", "player.name"]).size().sort_values(ascending=False).head(5).to_string())

print("\nCarries στην τελική τρίτη:\n",
      e[(e["type.name"] == "Carry") & e["entry"]].groupby(["team.name", "player.name"]).size()
      .sort_values(ascending=False).head(5).to_string())

dr = dm[(dm["type.name"] == "Dribble") & (col(dm, "dribble.outcome.name") == "Complete")]
print("\nΝικηφόρα dribbles:\n",
      dr.groupby(["team.name", "player.name"]).size().sort_values(ascending=False).head(5).to_string())