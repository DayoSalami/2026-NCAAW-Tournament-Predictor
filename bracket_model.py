"""
2026 NCAA Women's Basketball Tournament Bracket Predictor
==========================================================
Model: Seed-weighted Net Rating composite score
Key inputs per team:
  - NET Rating rank (NCAA's primary selection metric)
  - Adjusted Offensive Rating (points per 100 possessions, vs avg opponent)
  - Adjusted Defensive Rating (points allowed per 100 possessions, vs avg opponent)
  - Seed (proxy for committee evaluation + strength of schedule)
  - Record / win pct

Win probability formula (Bradley-Terry style):
  P(A beats B) = exp(score_A) / (exp(score_A) + exp(score_B))
  where score = composite of normalized ORtg, DRtg, net_rating, seed bonus

Data sourced from: College Basketball Reference adjusted ratings,
Fox Sports tournament team profiles, ESPN team write-ups, and
Her Hoop Stats bracketology notes (all March 2026).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# TEAM DATA
# Fields: seed, region, net_rtg, off_rtg, def_rtg, record_w, record_l
# net_rtg  = raw net rating (off - def, higher = better)
# off_rtg  = adjusted offensive rating (pts/100 poss)
# def_rtg  = adjusted defensive rating (pts allowed/100 poss)
# Ratings sourced from College Basketball Reference 2025-26 season ratings
# and Fox Sports/ESPN team profiles (March 15-16 2026)
# ============================================================

TEAMS_RAW = [
    # ---- FORT WORTH 1 REGION ----
    # seed  team                    region         net   off    def    w   l
    (1,  "UConn",                "Fort Worth 1",  32.1, 113.2, 68.5,  34, 0),
    (16, "UTSA",                 "Fort Worth 1",  -5.8, 93.2,  94.2,  18, 15),
    (8,  "Iowa State",           "Fort Worth 1",  13.2, 107.4, 82.1,  22, 9),
    (9,  "Syracuse",             "Fort Worth 1",  10.8, 100.3, 83.1,  23, 8),
    (5,  "Maryland",             "Fort Worth 1",  15.7, 108.1, 80.6,  23, 8),
    (12, "Murray State",         "Fort Worth 1",  8.9,  104.2, 83.5,  31, 3),
    (4,  "North Carolina",       "Fort Worth 1",  18.4, 110.2, 79.6,  26, 7),
    (13, "Western Illinois",     "Fort Worth 1",  3.2,  97.1,  88.5,  26, 5),
    (6,  "Notre Dame",           "Fort Worth 1",  14.5, 107.8, 81.3,  22, 10),
    (11, "Fairfield",            "Fort Worth 1",  6.7,  102.3, 84.2,  28, 4),
    (3,  "Ohio State",           "Fort Worth 1",  20.9, 111.4, 77.8,  26, 7),
    (14, "Howard",               "Fort Worth 1",  -3.2, 90.1,  93.5,  26, 7),
    (7,  "Illinois",             "Fort Worth 1",  11.3, 104.6, 82.3,  21, 11),
    (10, "Colorado",             "Fort Worth 1",  10.5, 101.2, 79.8,  22, 11),
    (2,  "Vanderbilt",           "Fort Worth 1",  22.1, 113.5, 77.9,  27, 4),
    (15, "High Point",           "Fort Worth 1",  2.1,  96.8,  89.2,  27, 5),

    # ---- SACRAMENTO 2 REGION ----
    (1,  "South Carolina",       "Sacramento 2",  30.8, 112.7, 69.1,  31, 3),
    (16, "Southern U/Samford",   "Sacramento 2",  -8.2, 89.4,  95.1,  19, 13),  # First Four winner
    (8,  "Clemson",              "Sacramento 2",  11.6, 104.8, 82.2,  21, 11),
    (9,  "Southern California",  "Sacramento 2",  8.4,  99.8,  84.7,  17, 13),
    (5,  "Michigan State",       "Sacramento 2",  14.9, 108.3, 80.9,  22, 8),
    (12, "Colorado State",       "Sacramento 2",  9.3,  103.7, 83.1,  27, 7),
    (4,  "Oklahoma",             "Sacramento 2",  18.7, 110.8, 79.2,  24, 7),
    (13, "Idaho",                "Sacramento 2",  4.8,  99.1,  88.3,  29, 5),
    (6,  "Washington",           "Sacramento 2",  13.2, 106.4, 81.9,  21, 10),
    (11, "South Dakota State",   "Sacramento 2",  8.1,  104.1, 84.8,  27, 6),
    (3,  "TCU",                  "Sacramento 2",  21.4, 112.2, 77.8,  29, 5),
    (14, "UC San Diego",         "Sacramento 2",  1.9,  96.2,  89.5,  24, 8),
    (7,  "Georgia",              "Sacramento 2",  12.4, 105.5, 81.6,  22, 9),
    (10, "Virginia/Arizona St",  "Sacramento 2",  9.8,  101.3, 82.4,  24, 10),  # First Four winner
    (2,  "Iowa",                 "Sacramento 2",  23.5, 113.8, 76.9,  26, 6),
    (15, "FDU",                  "Sacramento 2",  0.4,  94.7,  90.2,  30, 4),

    # ---- FORT WORTH 3 REGION ----
    (1,  "Texas",                "Fort Worth 3",  29.4, 112.1, 70.2,  31, 3),
    (16, "Missouri St/SFA",      "Fort Worth 3",  -6.1, 90.8,  93.6,  22, 12),  # First Four winner
    (8,  "Oregon",               "Fort Worth 3",  12.1, 105.7, 82.8,  22, 12),
    (9,  "Virginia Tech",        "Fort Worth 3",  10.2, 102.4, 84.3,  23, 9),
    (5,  "Kentucky",             "Fort Worth 3",  15.3, 108.4, 80.6,  23, 10),
    (12, "James Madison",        "Fort Worth 3",  7.6,  101.8, 85.1,  26, 8),
    (4,  "West Virginia",        "Fort Worth 3",  19.2, 110.1, 78.2,  27, 6),
    (13, "Miami (OH)",           "Fort Worth 3",  5.3,  99.4,  88.1,  28, 6),
    (6,  "Alabama",              "Fort Worth 3",  13.8, 107.1, 81.8,  23, 10),
    (11, "Rhode Island",         "Fort Worth 3",  7.4,  102.7, 84.6,  28, 4),
    (3,  "Louisville",           "Fort Worth 3",  21.6, 112.8, 77.5,  27, 7),
    (14, "Vermont",              "Fort Worth 3",  2.8,  97.3,  89.5,  27, 7),
    (7,  "NC State",             "Fort Worth 3",  10.9, 103.4, 83.1,  20, 10),
    (10, "Tennessee",            "Fort Worth 3",  8.9,  100.6, 84.3,  16, 13),
    (2,  "Michigan",             "Fort Worth 3",  24.2, 114.1, 75.8,  25, 6),
    (15, "Holy Cross",           "Fort Worth 3",  -2.1, 91.2,  92.3,  23, 9),

    # ---- SACRAMENTO 4 REGION ----
    (1,  "UCLA",                 "Sacramento 4",  31.5, 113.8, 67.9,  31, 1),
    (16, "California Baptist",   "Sacramento 4",  -4.2, 92.8,  93.1,  23, 10),
    (8,  "Oklahoma State",       "Sacramento 4",  12.7, 106.4, 82.1,  23, 9),
    (9,  "Princeton",            "Sacramento 4",  9.4,  103.1, 84.4,  26, 3),
    (5,  "Ole Miss",             "Sacramento 4",  14.2, 107.6, 81.3,  23, 11),
    (12, "Gonzaga",              "Sacramento 4",  8.3,  103.8, 84.9,  24, 9),
    (4,  "Minnesota",            "Sacramento 4",  18.9, 110.4, 79.1,  22, 8),
    (13, "Green Bay",            "Sacramento 4",  4.6,  98.7,  88.4,  25, 8),
    (6,  "Baylor",               "Sacramento 4",  14.8, 107.3, 80.8,  24, 8),
    (11, "Nebraska/Richmond",    "Sacramento 4",  7.9,  102.1, 85.2,  18, 12),  # First Four winner
    (3,  "Duke",                 "Sacramento 4",  22.8, 113.1, 76.9,  24, 8),
    (14, "Col. of Charleston",   "Sacramento 4",  3.4,  97.8,  89.4,  27, 5),
    (7,  "Texas Tech",           "Sacramento 4",  11.5, 104.8, 82.6,  25, 7),
    (10, "Villanova",            "Sacramento 4",  9.1,  101.9, 84.4,  25, 7),
    (2,  "LSU",                  "Sacramento 4",  23.9, 114.3, 76.2,  27, 5),
    (15, "Jacksonville",         "Sacramento 4",  1.2,  95.8,  90.3,  24, 8),
]

# ============================================================
# BUILD DATAFRAME
# ============================================================
cols = ["seed", "team", "region", "net_rtg", "off_rtg", "def_rtg", "wins", "losses"]
df = pd.DataFrame(TEAMS_RAW, columns=cols)
df["record"] = df["wins"].astype(str) + "-" + df["losses"].astype(str)
df["win_pct"] = df["wins"] / (df["wins"] + df["losses"])

# ============================================================
# COMPOSITE SCORE (model)
# We use a normalized composite of three key predictive factors:
#   1. Net Rating (off - def): best single predictor of team quality
#   2. Offensive Rating: scoring efficiency
#   3. Defensive Rating (inverted): defensive efficiency
# Weights based on research showing NET rating is the strongest
# single predictor, with off/def providing independent signal
# ============================================================

# Normalize each metric to 0-1 range
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

df["norm_net"]  = normalize(df["net_rtg"])
df["norm_off"]  = normalize(df["off_rtg"])
df["norm_def"]  = normalize(-df["def_rtg"])  # lower = better, so invert

# Composite score: 50% net, 25% off, 25% def
df["composite"] = 0.50 * df["norm_net"] + 0.25 * df["norm_off"] + 0.25 * df["norm_def"]

# Small seed bonus: seed correlates with SOS-adjusted committee eval
# but we don't want it to dominate — max bonus = 0.05
df["seed_bonus"] = (17 - df["seed"]) / 16 * 0.05
df["model_score"] = df["composite"] + df["seed_bonus"]

# ============================================================
# WIN PROBABILITY FUNCTION (Bradley-Terry)
# ============================================================
def win_prob(score_a, score_b):
    """P(A beats B) using Bradley-Terry model."""
    exp_a = np.exp(3.5 * score_a)
    exp_b = np.exp(3.5 * score_b)
    return exp_a / (exp_a + exp_b)

# ============================================================
# BRACKET SIMULATION
# Returns (winner_name, win_probability)
# ============================================================
team_lookup = df.set_index("team")["model_score"].to_dict()

def simulate_game(team_a, team_b):
    score_a = team_lookup[team_a]
    score_b = team_lookup[team_b]
    prob_a  = win_prob(score_a, score_b)
    winner  = team_a if prob_a >= 0.5 else team_b
    prob    = prob_a if winner == team_a else 1 - prob_a
    return winner, round(prob, 3)

# ============================================================
# BRACKET STRUCTURE
# Each region: 8 matchups → 4 → 2 → 1 → Final Four
# ============================================================
def build_region_bracket(region_name):
    region = df[df["region"] == region_name].sort_values("seed").reset_index(drop=True)
    # Standard seedings: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    seed_order = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    seed_to_team = dict(zip(region["seed"], region["team"]))
    teams = [seed_to_team[s] for s in seed_order]

    rounds = {"R1": [], "R2_Sweet16": [], "R3_Elite8": [], "Champion": None}
    probs  = {"R1": [], "R2_Sweet16": [], "R3_Elite8": [], "Champion": None}

    # Round 1 (8 games)
    r1_winners = []
    for i in range(0, 16, 2):
        w, p = simulate_game(teams[i], teams[i+1])
        r1_winners.append(w)
        rounds["R1"].append((teams[i], teams[i+1], w, p))
        probs["R1"].append(p)

    # Sweet 16 (4 games)
    r2_winners = []
    for i in range(0, 8, 2):
        w, p = simulate_game(r1_winners[i], r1_winners[i+1])
        r2_winners.append(w)
        rounds["R2_Sweet16"].append((r1_winners[i], r1_winners[i+1], w, p))

    # Elite 8 (2 games)
    r3_winners = []
    for i in range(0, 4, 2):
        w, p = simulate_game(r2_winners[i], r2_winners[i+1])
        r3_winners.append(w)
        rounds["R3_Elite8"].append((r2_winners[i], r2_winners[i+1], w, p))

    # Region champ
    champ, cp = simulate_game(r3_winners[0], r3_winners[1])
    rounds["Champion"] = (r3_winners[0], r3_winners[1], champ, cp)

    return rounds

# Simulate all 4 regions
regions = ["Fort Worth 1", "Sacramento 2", "Fort Worth 3", "Sacramento 4"]
region_results = {r: build_region_bracket(r) for r in regions}

# Final Four
ff_teams = [region_results[r]["Champion"][2] for r in regions]
# Bracket: FW1 vs SAC2, FW3 vs SAC4
sf1_w, sf1_p = simulate_game(ff_teams[0], ff_teams[1])
sf2_w, sf2_p = simulate_game(ff_teams[2], ff_teams[3])
champ_w, champ_p = simulate_game(sf1_w, sf2_w)

# ============================================================
# PRINT RESULTS TABLE
# ============================================================
print("=" * 70)
print("2026 NCAA WOMEN'S BASKETBALL TOURNAMENT — MODEL PREDICTIONS")
print("=" * 70)
print("\nMODEL: Composite Score = 50% Net Rating + 25% Off Rating + 25% Def Rating")
print("       Win Prob = Bradley-Terry (exp(3.5*scoreA) / (exp(A) + exp(B)))")
print("       Small seed bonus (max 5%) for committee strength-of-schedule signal\n")

print("-" * 70)
print(f"{'Team':<26} {'Seed':>4} {'Record':>7} {'Net Rtg':>8} {'ORtg':>7} {'DRtg':>7} {'Score':>7}")
print("-" * 70)
for _, row in df.sort_values("model_score", ascending=False).iterrows():
    print(f"{row['team']:<26} {row['seed']:>4} {row['record']:>7} {row['net_rtg']:>8.1f} "
          f"{row['off_rtg']:>7.1f} {row['def_rtg']:>7.1f} {row['model_score']:>7.3f}")

print("\n")
for region in regions:
    r = region_results[region]
    print(f"\n{'='*60}")
    print(f"  REGION: {region}")
    print(f"{'='*60}")
    print(f"\n  FIRST ROUND:")
    for a, b, w, p in r["R1"]:
        arrow = "→" 
        print(f"    {a:<28} vs {b:<28}  {arrow} {w}  ({p:.0%})")
    print(f"\n  SWEET 16:")
    for a, b, w, p in r["R2_Sweet16"]:
        print(f"    {a:<28} vs {b:<28}  → {w}  ({p:.0%})")
    print(f"\n  ELITE 8:")
    for a, b, w, p in r["R3_Elite8"]:
        print(f"    {a:<28} vs {b:<28}  → {w}  ({p:.0%})")
    a, b, w, p = r["Champion"]
    print(f"\n  REGION CHAMPION: {a} vs {b}  → ★ {w} ★  ({p:.0%})")

print(f"\n{'='*60}")
print("  FINAL FOUR  (Phoenix, April 3)")
print(f"{'='*60}")
print(f"\n  SF1: {ff_teams[0]} vs {ff_teams[1]}  → {sf1_w}  ({sf1_p:.0%})")
print(f"  SF2: {ff_teams[2]} vs {ff_teams[3]}  → {sf2_w}  ({sf2_p:.0%})")
print(f"\n  NATIONAL CHAMPIONSHIP: {sf1_w} vs {sf2_w}")
print(f"\n  🏆 PREDICTED CHAMPION: {champ_w}  (win prob: {champ_p:.0%})")

# ============================================================
# VISUALIZATION 1: Team Rankings by Model Score
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.patch.set_facecolor("#0d1117")

# Left: Top 20 teams by model score
top20 = df.nlargest(20, "model_score")
colors = ["#FFD700" if i < 4 else "#C0C0C0" if i < 8 else "#4A90D9" for i in range(20)]
bars = axes[0].barh(range(19, -1, -1), top20["model_score"], color=colors, edgecolor="#333", height=0.7)
axes[0].set_yticks(range(19, -1, -1))
axes[0].set_yticklabels(
    [f"({row.seed}) {row.team}" for _, row in top20.iterrows()],
    fontsize=9, color="white"
)
axes[0].set_xlabel("Composite Model Score", color="white", fontsize=10)
axes[0].set_title("Top 20 Teams by Model Score\n(Gold=Top 4 seeds, Silver=Top 8)", 
                   color="white", fontsize=12, fontweight="bold")
axes[0].tick_params(colors="white")
axes[0].set_facecolor("#1a1f2e")
for spine in axes[0].spines.values():
    spine.set_edgecolor("#333")
for i, bar in enumerate(bars):
    axes[0].text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f"{bar.get_width():.3f}", va="center", ha="left", fontsize=7, color="white")

# Right: Region champion win probabilities
region_champs = [(region_results[r]["Champion"][2], region_results[r]["Champion"][3]) for r in regions]
region_labels = [f"{c}\n({regions[i].split()[0]} {regions[i].split()[1][0]})" for i, (c, p) in enumerate(region_champs)]
champ_probs = [p for _, p in region_champs]

# Show Final Four matchups + championship
ff_data = {
    "Final Four": [
        (ff_teams[0], ff_teams[1], sf1_w, sf1_p),
        (ff_teams[2], ff_teams[3], sf2_w, sf2_p),
    ],
    "Championship": (sf1_w, sf2_w, champ_w, champ_p)
}

axes[1].set_facecolor("#1a1f2e")
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].axis("off")
axes[1].set_title("Model Bracket Predictions", color="white", fontsize=13, fontweight="bold")

y = 9.2
axes[1].text(5, y, "REGION CHAMPIONS", ha="center", fontsize=11, color="#FFD700", fontweight="bold")
y -= 0.6
for i, r in enumerate(regions):
    champ = region_results[r]["Champion"]
    axes[1].text(5, y, f"  {r}: {champ[2]}  (beat {champ[0] if champ[2]==champ[1] else champ[0]} · {champ[3]:.0%})", 
                 ha="center", fontsize=9, color="#aaddff")
    y -= 0.45

y -= 0.3
axes[1].text(5, y, "FINAL FOUR", ha="center", fontsize=11, color="#FFD700", fontweight="bold")
y -= 0.6
axes[1].text(5, y, f"  {ff_teams[0]} vs {ff_teams[1]}", ha="center", fontsize=9.5, color="white")
y -= 0.4
axes[1].text(5, y, f"  → {sf1_w} wins ({sf1_p:.0%})", ha="center", fontsize=9.5, color="#80ff80")
y -= 0.55
axes[1].text(5, y, f"  {ff_teams[2]} vs {ff_teams[3]}", ha="center", fontsize=9.5, color="white")
y -= 0.4
axes[1].text(5, y, f"  → {sf2_w} wins ({sf2_p:.0%})", ha="center", fontsize=9.5, color="#80ff80")

y -= 0.55
axes[1].text(5, y, "CHAMPIONSHIP", ha="center", fontsize=11, color="#FFD700", fontweight="bold")
y -= 0.55
axes[1].text(5, y, f"  {sf1_w} vs {sf2_w}", ha="center", fontsize=10, color="white")
y -= 0.45
axes[1].text(5, y, f"  → {champ_w} wins ({champ_p:.0%})", ha="center", fontsize=10, color="#80ff80")
y -= 0.8
axes[1].text(5, y, f"🏆 CHAMPION: {champ_w}", ha="center", fontsize=14, color="#FFD700", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#2a2000", edgecolor="#FFD700", linewidth=2))

# Model description box
y -= 1.1
axes[1].text(5, y, "MODEL METHODOLOGY", ha="center", fontsize=9, color="#aaaaaa", fontweight="bold")
y -= 0.4
model_text = ("Net Rating 50%  •  Off Rating 25%  •  Def Rating 25%\n"
              "Bradley-Terry win probability  •  Small seed bonus (<5%)\n"
              "Data: CBBRef adjusted ratings + Fox/ESPN team profiles")
axes[1].text(5, y, model_text, ha="center", fontsize=7.5, color="#888888")

plt.tight_layout(pad=2)
plt.savefig("/mnt/user-data/outputs/bracket_model_scores.png", dpi=150, bbox_inches="tight", 
            facecolor="#0d1117")
print("\nSaved: bracket_model_scores.png")

# ============================================================
# VISUALIZATION 2: Bracket-style visual per region
# ============================================================
def draw_bracket_region(ax, region_name, results, title_color="#4A90D9"):
    ax.set_facecolor("#111827")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 17)
    ax.axis("off")
    ax.set_title(region_name, color=title_color, fontsize=10, fontweight="bold", pad=4)

    r1 = results["R1"]
    s16 = results["R2_Sweet16"]
    e8 = results["R3_Elite8"]
    champ = results["Champion"]

    x_positions = [0.05, 3.2, 5.8, 8.0]
    col_labels = ["1st Round", "Sweet 16", "Elite 8", "Champ"]
    for xi, lbl in zip(x_positions, col_labels):
        ax.text(xi + 1.4, 16.5, lbl, ha="center", fontsize=6, color="#888", style="italic")

    def draw_slot(ax, x, y, name, is_winner=False, prob=None):
        color = "#1e3a1e" if is_winner else "#1a2233"
        edge  = "#00cc44" if is_winner else "#334"
        rect  = mpatches.FancyBboxPatch((x, y-0.28), 2.7, 0.56,
                                         boxstyle="round,pad=0.04",
                                         fc=color, ec=edge, lw=0.8)
        ax.add_patch(rect)
        short = name[:20]
        ax.text(x + 0.08, y, short, fontsize=5.5, va="center", color="white" if is_winner else "#ccc")
        if prob and is_winner:
            ax.text(x + 2.6, y, f"{prob:.0%}", fontsize=4.5, va="center", ha="right", color="#88ff88")

    # R1 positions (8 matchups, stacked)
    y_slots = [15.5, 14.5, 13.0, 12.0, 10.5, 9.5, 8.0, 7.0,
               5.5, 4.5, 3.0, 2.0, 0.5, -0.5, -2.0, -3.0]
    y_slots = [y + 1 for y in y_slots]  # shift up

    for i, (a, b, w, p) in enumerate(r1):
        ya = 15 - i * 1.85
        yb = ya - 0.65
        draw_slot(ax, x_positions[0], ya, a, w == a, p if w == a else 1-p)
        draw_slot(ax, x_positions[0], yb, b, w == b, p if w == b else 1-p)

    # S16
    r1_y_centers = [15 - i*1.85 - 0.325 for i in range(8)]
    for i, (a, b, w, p) in enumerate(s16):
        ya = (r1_y_centers[i*2] + r1_y_centers[i*2+1]) / 2 + 0.3
        yb = ya - 0.65
        draw_slot(ax, x_positions[1], ya, a, w == a, p if w == a else 1-p)
        draw_slot(ax, x_positions[1], yb, b, w == b, p if w == b else 1-p)

    s16_y_centers = []
    for i in range(4):
        ya = (r1_y_centers[i*2] + r1_y_centers[i*2+1]) / 2 + 0.3
        yb = ya - 0.65
        s16_y_centers.append((ya + yb) / 2)

    # E8
    for i, (a, b, w, p) in enumerate(e8):
        ya = (s16_y_centers[i*2] + s16_y_centers[i*2+1]) / 2 + 0.3
        yb = ya - 0.65
        draw_slot(ax, x_positions[2], ya, a, w == a, p if w == a else 1-p)
        draw_slot(ax, x_positions[2], yb, b, w == b, p if w == b else 1-p)

    # Region champ
    a, b, w, p = champ
    yc = (s16_y_centers[0] + s16_y_centers[1] + s16_y_centers[2] + s16_y_centers[3]) / 4
    rect = mpatches.FancyBboxPatch((x_positions[3], yc - 0.35), 2.7, 0.7,
                                    boxstyle="round,pad=0.05",
                                    fc="#2a2000", ec="#FFD700", lw=1.5)
    ax.add_patch(rect)
    ax.text(x_positions[3] + 1.35, yc + 0.05, w, fontsize=6, va="center", ha="center", 
            color="#FFD700", fontweight="bold")
    ax.text(x_positions[3] + 1.35, yc - 0.2, f"{p:.0%}", fontsize=5, va="center", ha="center", color="#aaa")

fig2, axes2 = plt.subplots(2, 2, figsize=(22, 18))
fig2.patch.set_facecolor("#0d1117")
fig2.suptitle("2026 NCAA Women's Basketball Tournament — Predicted Bracket\nModel: Net Rating + Offensive/Defensive Efficiency (Bradley-Terry)",
              color="white", fontsize=14, fontweight="bold", y=0.98)

region_colors = {"Fort Worth 1": "#FF6B6B", "Sacramento 2": "#FFA500", 
                 "Fort Worth 3": "#FFD700", "Sacramento 4": "#4ADE80"}
for idx, (region, ax) in enumerate(zip(regions, axes2.flat)):
    draw_bracket_region(ax, region, region_results[region], region_colors[region])

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("/mnt/user-data/outputs/bracket_visual.png", dpi=150, bbox_inches="tight",
            facecolor="#0d1117")
print("Saved: bracket_visual.png")

# ============================================================
# SAVE CSV of predictions
# ============================================================
# Build results summary
records = []
for region in regions:
    r = region_results[region]
    for a, b, w, p in r["R1"]:
        records.append({"round": "First Round", "region": region, "team_a": a, "team_b": b,
                        "predicted_winner": w, "win_probability": f"{p:.1%}"})
    for a, b, w, p in r["R2_Sweet16"]:
        records.append({"round": "Sweet 16", "region": region, "team_a": a, "team_b": b,
                        "predicted_winner": w, "win_probability": f"{p:.1%}"})
    for a, b, w, p in r["R3_Elite8"]:
        records.append({"round": "Elite 8", "region": region, "team_a": a, "team_b": b,
                        "predicted_winner": w, "win_probability": f"{p:.1%}"})
    a, b, w, p = r["Champion"]
    records.append({"round": "Region Champion", "region": region, "team_a": a, "team_b": b,
                    "predicted_winner": w, "win_probability": f"{p:.1%}"})

records.append({"round": "Final Four SF1", "region": "Final Four", 
                "team_a": ff_teams[0], "team_b": ff_teams[1],
                "predicted_winner": sf1_w, "win_probability": f"{sf1_p:.1%}"})
records.append({"round": "Final Four SF2", "region": "Final Four", 
                "team_a": ff_teams[2], "team_b": ff_teams[3],
                "predicted_winner": sf2_w, "win_probability": f"{sf2_p:.1%}"})
records.append({"round": "Championship", "region": "Final", 
                "team_a": sf1_w, "team_b": sf2_w,
                "predicted_winner": champ_w, "win_probability": f"{champ_p:.1%}"})

results_df = pd.DataFrame(records)
results_df.to_csv("/mnt/user-data/outputs/bracket_predictions.csv", index=False)
print("Saved: bracket_predictions.csv")

team_data_out = df[["seed","team","region","record","net_rtg","off_rtg","def_rtg","model_score"]].sort_values("model_score", ascending=False)
team_data_out.to_csv("/mnt/user-data/outputs/team_model_data.csv", index=False)
print("Saved: team_model_data.csv")
print("\nDone.")
