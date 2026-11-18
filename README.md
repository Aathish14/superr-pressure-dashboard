# SupeRR Pressure Dashboard — T20 Pressure-Proxy Scouting

A reproducible, auditable methodology for ranking T20 batters and bowlers by changes in outcome rates under observable pressure proxies, computed from ball-by-ball event data.

**Live app:** https://superr-pressure-dashboard.streamlit.app/

## Repository contents

| File | Purpose |
|------|---------|
| `dashboard.py` | Streamlit entry point — interactive rankings, charts, methodology, and formulas |
| `pressure_proxy.py` | Analysis module — `analyze_all()`, `rank_batters()`, `rank_bowlers()`, `sensitivity()` |
| `requirements.txt` | Python dependencies: `streamlit`, `pandas`, `numpy`, `plotly` |
| `requirements-dashboard.txt` | Same dependencies (alternate filename) |
| `submission_outputs/` | Pre-computed IPL batter/bowler ranking CSVs |
| `.gitignore` | Ignores everything except listed dashboard assets |

**Note:** Raw input files (`ball_by_ball.csv`, `matches.csv`, `competitions.csv`) are **not** tracked in this repository. The dashboard loads them from the working directory via `pressure_proxy.analyze_all()`.

---

## Purpose and scope

The objective is to identify players whose measured outcomes **improve under defined pressure proxies** relative to their own ordinary-performance baselines — avoiding ranking by reputation or aggregate volume alone.

- **Primary recommendations:** IPL data (Rajasthan Royals scouting context)
- **Cross-format stress check:** T20I analyzed independently (not pooled with IPL)
- **Nature of results:** Descriptive scouting evidence, not causal estimates of a general "clutch" skill

---

## Data & preparation

The supplied match records contain:

| Cohort | Matches | Delivery-event rows |
|--------|---------|---------------------|
| IPL | 371 | 86,668 |
| T20I | 482 | 102,887 |

**Exclusions:** Matches with revised-overs or revised-target metadata are removed (5 IPL, 29 T20I). Only regulation innings 1 and 2 are retained; Super Overs and other T20 competitions are omitted.

**Integrity decisions:**
- `extras1_runs` behaves as an event-category code, not an extras-run amount; it is never summed as runs.
- 36 nonzero `extras_2_runs` values are unresolved; that field is not used as a run total.
- The `runs` field does not reproduce innings totals in reviewed scorecard examples, and no innings totals or provider-documented legal-ball flag are available. Exact score, required run rate, and economy are therefore **not reconstructed**.
- Bowler wicket credit is limited to: Bowled, Caught, Caught and Bowled, Hit Wicket, LBW, Stump. A non-striker run-out is not counted as a striker dismissal.

---

## Pressure-state definitions

Pressure is operationalized from auditable event-sequence fields. None of these states identifies exact match closeness.

| Role | Pressure indicator | Ordinary comparison |
|------|-------------------|---------------------|
| Batter | Overs 1–6 with ≥2 wickets lost before the row, OR overs 16–20 | All other striker rows in the same competition and cohort |
| Bowler | Partnership event-row index > 25; any wicket resets the index | Rows outside pressure, within same over phase (powerplay 1–6 / middle 7–15 / death 16–20) |

- Wickets-before-row excludes a dismissal recorded on the current event.
- Batter faced-row exposure excludes rows identified as wides.
- The partnership index is zero-based and counts event rows, not legal balls.

---

## Outcome metrics & ranking formulas

### Batting

For batter $i$ and state $s \in \{P, O\}$, the non-dismissal indicator is 1 unless the striker is dismissed on that row. The boundary indicator is 1 when the provider's four or six flag is set.

Shrinkage toward the state-level pooled rate (30 pseudo-rows):

$$\widehat p_{i,s}^{(m)} = \frac{x_{i,s}^{(m)} + \kappa \, p_s^{(m)}}{n_{i,s} + \kappa}, \qquad \kappa = 30$$

where $x_{i,s}^{(m)}$ is the outcome count, $n_{i,s}$ its row exposure, and $p_s^{(m)}$ the pooled rate for that state within the format.

Ranking axes are the pressure-minus-ordinary differences:

- $\Delta S_i = \widehat p_{i,P}^{(S)} - \widehat p_{i,O}^{(S)}$ (survival change)
- $\Delta B_i = \widehat p_{i,P}^{(B)} - \widehat p_{i,O}^{(B)}$ (boundary change)

Players positive on both axes are ordered by Pareto front; ties use the weaker standardized change and then pressure exposure.

### Bowling

For bowler $i$, over phase $g$, and state $s$:

$$\widehat r_{i,g,s} = \frac{w_{i,g,s} + \kappa \, \lambda_{g,s}}{n_{i,g,s} + \kappa}, \qquad \kappa = 30$$

where $w_{i,g,s}$ are credited wickets, $n_{i,g,s}$ delivery-row exposure, and $\lambda_{g,s}$ the pooled phase/state rate.

Phase-specific pressure-minus-ordinary changes are aggregated using each bowler's pressure-row exposure:

$$\Delta W_i = \frac{\sum_g n_{i,g,P} (\widehat r_{i,g,P} - \widehat r_{i,g,O})}{\sum_g n_{i,g,P}}$$

Bowler rates and $\Delta W_i$ are reported as **credited wickets per 100 delivery rows**, not per legal ball. Bowlers are ranked by $\Delta W_i$.

---

## Fair comparison design

Eligibility is applied separately by role and format:

- At least **30 pressure rows** across at least **3 distinct matches**
- At least **30 ordinary rows**

This removes very small or one-match samples. IPL and T20I state pools, estimates, and candidate filters are calculated independently.

| Cohort | Eligible batters | Eligible bowlers |
|--------|------------------|------------------|
| IPL | 126 | 129 |
| T20I | 198 | 186 |

---

## Results

### Top 10 IPL batters (pressure proxies)

| Rank | Batter | Pressure rows | Matches | Survival O/P | ΔS (pp) | Boundary O/P | ΔB (pp) |
|------|--------|---------------|---------|--------------|---------|--------------|---------|
| 1 | Ramandeep Singh | 93 | 17 | 92.04% → 93.39% | +1.34 | 12.56% → 25.02% | +12.45 |
| 2 | Vijay Shankar | 142 | 14 | 94.05% → 95.85% | +1.80 | 12.35% → 24.29% | +11.93 |
| 3 | Ajinkya Rahane | 71 | 10 | 94.82% → 96.89% | +2.07 | 17.29% → 27.50% | +10.21 |
| 4 | Cameron Green | 202 | 26 | 95.42% → 97.79% | +2.36 | 18.44% → 22.32% | +3.88 |
| 5 | Glenn Maxwell | 182 | 24 | 92.47% → 92.86% | +0.39 | 19.89% → 31.02% | +11.13 |
| 6 | Sarfaraz Khan | 91 | 12 | 93.68% → 94.10% | +0.42 | 15.85% → 26.26% | +10.40 |
| 7 | Cooper Connolly | 60 | 4 | 95.83% → 96.52% | +0.69 | 22.08% → 30.86% | +8.78 |
| 8 | Glenn Phillips | 43 | 6 | 90.00% → 91.59% | +1.60 | 14.03% → 21.60% | +7.57 |
| 9 | Shubham Dubey | 94 | 13 | 93.29% → 93.44% | +0.15 | 16.25% → 24.01% | +7.75 |
| 10 | Mayank Agarwal | 32 | 4 | 94.01% → 94.94% | +0.93 | 17.35% → 20.60% | +3.25 |

Ramandeep Singh leads the Pareto-ordered list with 93 pressure rows across 17 matches; his shrunk survival rate changes by +1.34 points and his boundary rate by +12.45 points.

### Top 10 IPL bowlers (partnership proxy)

| Rank | Bowler | P rows | Matches | Wickets (P) | Rate O/P | ΔW / 100 |
|------|--------|--------|---------|-------------|----------|----------|
| 1 | Yash Punja | 67 | 6 | 6 | 2.55 / 7.77 | +5.22 |
| 2 | Harpreet Brar | 123 | 22 | 12 | 3.55 / 8.36 | +4.81 |
| 3 | Mitchell Marsh | 66 | 10 | 8 | 4.55 / 8.96 | +4.41 |
| 4 | Sai Kishore | 148 | 16 | 14 | 3.95 / 8.35 | +4.40 |
| 5 | Dwayne Bravo | 78 | 9 | 8 | 3.93 / 7.86 | +3.93 |
| 6 | Liam Livingstone | 82 | 14 | 6 | 2.82 / 6.73 | +3.90 |
| 7 | FazalHaq Farooqi | 63 | 11 | 4 | 1.95 / 5.76 | +3.81 |
| 8 | Kuldeep Sen | 43 | 8 | 4 | 3.93 / 7.67 | +3.74 |
| 9 | Shashank Singh | 36 | 7 | 3 | 3.16 / 6.87 | +3.71 |
| 10 | Lockie Ferguson | 122 | 18 | 10 | 3.56 / 7.08 | +3.52 |

Yash Punja leads with a +5.22 credited-wicket-per-100-row difference, based on 67 pressure rows across 6 matches.

---

## Sensitivity & format check

Threshold sensitivity compares alternative top-ten lists with the primary IPL list. Jaccard = intersection / union.

| Analysis | Alternative | Overlap | Jaccard |
|----------|-------------|---------|---------|
| Batter pressure state | Death from over 15 | 5/10 | 0.33 |
| Batter pressure state | Collapse at 3+ wickets | 6/10 | 0.43 |
| Bowler partnership threshold | More than 30 delivery rows | 7/10 | 0.54 |

Cross-format: T20I top-ten lists overlap IPL top ten by **1/10 batters** and **0/10 bowlers** — limited cross-format transfer, not independent confirmation of IPL ordering.

---

## Reliability & limitations

- **Descriptive associations, not causal evidence** of a universal clutch trait.
- Pressure states capture innings phase and wicket/partnership context, **not exact contest closeness**.
- No leave-one-match-out analysis or bootstrap uncertainty interval was calculated; a player's rate change may be influenced by a small number of innings.
- Unverified extras encoding and missing legal-ball metadata mean denominators are delivery rows; score, chase pressure, and economy are not inferred.
- Shrinkage strength ($\kappa=30$) and eligibility cutoffs are design choices; list membership changes when thresholds change.
- IPL and T20I rankings differ substantially; recommendations are **not pooled across formats**.

---

## Running the dashboard locally

```bash
# Install dependencies
pip install -r requirements.txt

# Place raw input files in the working directory:
#   ball_by_ball.csv, matches.csv, competitions.csv

# Run the app
streamlit run dashboard.py
```

The app will be available at `http://localhost:8501`.

---

## Streamlit Community Cloud deployment

- **Repository:** `Aathish14/superr-pressure-dashboard`
- **Branch:** `main`
- **App file:** `dashboard.py`
- **Live app:** https://superr-pressure-dashboard.streamlit.app/

Before deploying, ensure `ball_by_ball.csv`, `matches.csv`, and `competitions.csv` are accessible to the app (they are gitignored in this repo).

---

## Citation

SupeRR Selector Season 2 — Round 1 Problem Statement (supplied challenge document, 2026).
