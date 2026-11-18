# PathSnap: GPS Map-Matching and Route Reconstruction

**UE24CS352A Machine Learning Hackathon, PES University**

PathSnap takes noisy GPS points recorded by a vehicle and works out **which road segment the vehicle was actually on** at every moment, then rebuilds the full route it drove. It combines a machine-learning classifier, which scores candidate roads for each GPS point, with a **Hidden Markov Model decoded by the Viterbi algorithm**, which picks the most consistent connected route over the whole trip. The system runs as a **FastAPI backend** on one laptop and a **Streamlit UI** on another.

```
raw GPS + road network
   │  project each trip to its local UTM zone (metres)
   ▼
candidate road segments within 50 m of every GPS point
   ▼
15 engineered features (distance, direction, consistency, connectivity, motion)
   ▼
HistGradientBoosting classifier → probability that each candidate is the true road
   ▼
HMM + Viterbi → most likely connected route for the whole trip
   ▼
FastAPI backend (laptop 1)  ⇄  Streamlit UI with Folium map (laptop 2)
```

---

## Results

All numbers below are produced by the scripts in this repository, and the full outputs are in [`results/`](results/). Student-test numbers are on all 10 student-test trips. The evaluator-test partition was never used for training, tuning or model selection.

### Model comparison (6 algorithm families)

5-fold GroupKFold cross-validation grouped by trip, on the 80 training trips. Each model was then refit and evaluated on the student-test trips.

| Model | Family | CV Macro F1 | Test Top-1 | Test Macro F1 | Test Log Loss | Route Continuity |
|---|---|---|---|---|---|---|
| Logistic Regression | Linear | 0.899 ± 0.008 | 0.863 | 0.906 | 0.085 | 0.943 |
| Gaussian Naive Bayes | Probabilistic | 0.794 ± 0.009 | 0.836 | 0.790 | 0.659 | 0.946 |
| Decision Tree | Tree | 0.920 ± 0.011 | 0.876 | 0.929 | 0.067 | 0.937 |
| Random Forest | Bagging ensemble | 0.924 ± 0.009 | 0.883 | 0.934 | 0.057 | 0.942 |
| **HistGradientBoosting** | **Boosting** | **0.926 ± 0.008** | 0.880 | 0.933 | **0.056** | 0.941 |
| MLP | Neural network | 0.921 ± 0.010 | 0.870 | 0.926 | 0.062 | 0.946 |

**Selected model: HistGradientBoosting.** It has the best cross-validation Macro F1 and the lowest log loss, which matters because its probabilities feed the HMM. It is also about 3× faster at inference than Random Forest.

### Bayesian hyperparameter optimisation

Optuna TPE sampler, 25 trials, objective = GroupKFold Top-1 accuracy (see [`results/tuning/`](results/tuning/)).

| | CV Macro F1 | CV Top-1 | Test Top-1 | Test Macro F1 | Test Log Loss |
|---|---|---|---|---|---|
| Untuned | 0.926 | 0.872 | 0.880 | 0.933 | 0.056 |
| Tuned | 0.925 | 0.872 | 0.880 | 0.933 | 0.064 |

Tuning did not meaningfully improve on the defaults: Top-1 was unchanged and log loss was slightly worse. The default configuration was already close to optimal for this feature set.

### Classifier only vs classifier + HMM (student-test)

| Metric | Classifier only | Classifier + HMM |
|---|---|---|
| Point accuracy | 0.882 | **0.902** |
| Route continuity | 0.943 | **0.975** |
| Route arc precision | 0.879 | **0.935** |
| Route arc recall | 0.891 | **0.899** |
| Route arc F1 | 0.883 | **0.914** |

The HMM improved point accuracy on **all 10** student-test trips, with gains ranging from +0.1 to +3.6 percentage points.

### Effect of feature engineering

A model using only the distance features picks the true road for about **55%** of GPS points. Adding direction, consistency and connectivity features raises this to about **88%**. The biggest gain comes from `heading_diff`, because almost every road is stored twice (once per direction) at the same distance, and only the direction of travel can separate the two.

| Mean feature value | True road | Wrong road |
|---|---|---|
| Heading difference (°) | 24 | 95 |
| Connected to previous road | 0.91 | 0.18 |
| Distance to GPS point (m) | 6.5 | 27.0 |

---

## Dataset

Each trip folder contains four whitespace-separated files:

| File | Columns | Meaning |
|---|---|---|
| `<id>.track` | lon, lat, time (s) | GPS points, about one per second |
| `<id>.nodes` | lon, lat | Road junctions (row number = node ID) |
| `<id>.arcs` | from_node, to_node | Directed road segments (row number = arc ID) |
| `<id>.route` | arc ID | Reference route: the arcs actually driven, in order |

| Partition | Trips | GPS points | Use |
|---|---|---|---|
| `train` | 80 | 208,417 | Training, cross-validation, tuning |
| `student_test` | 10 | 19,160 | Model comparison and final selection |
| `evaluator_test` | 10 | n/a | Final demo only, through the API |

**Findings that shaped the design:**
- **Trips are spread across the world** (25 UTM zones, from the Americas to Japan). Each trip is projected to **its own UTM zone**, keeping distance errors below 0.2%. A single national grid would distort distances by up to 4×.
- **Almost every road is stored in both directions** (A→B and B→A), so direction-of-travel features are essential.
- **Some training reference routes don't match their GPS** (in trip 54, for example, the GPS sits about 130 m from the route for most of the drive). GPS points whose reference arc is more than 50 m away are flagged `reliable = 0` and **excluded from training** (88% of training points are reliable).

The raw data is not committed. Place it as shown below, or set the `PATHSNAP_DATA` environment variable for the EDA scripts.

```
data/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/student_test/<trip_id>/...
data/evaluator_test/<trip_id>/...     ← only needed on the backend laptop
```

---

## Method

**1. Candidates and labels** (`data_processing/build_candidates.py`)
- Every road segment within **50 m** of a GPS point becomes a candidate (about 9 per point on average).
- Per-point labels are created by aligning the GPS points to the ordered reference route with **dynamic programming**. Each point is matched to exactly one route arc, the matching can only move forward along the route, and the total point-to-arc distance is minimised.
- Candidate miss rate at 50 m: **0.14%** on student-test, and 11.2% on train. The training misses come entirely from points whose reference route doesn't match the GPS (the unreliable points).

**2. Features** (`features/build_features.py`): 15 features, none of which use the label.

| Group | Features |
|---|---|
| Distance | `dist_m`, `dist_rank`, `dist_minus_min`, `dist_ratio`, `n_cands`, `proj_t` |
| Direction | `heading_diff`, `heading_cos`, `heading_valid` (heading is smoothed over ±2 s; it is switched off when the vehicle moves less than 3 m) |
| Temporal consistency | `dist_prev_m`, `dist_next_m` |
| Connectivity | `connected_prev` |
| Motion and context | `speed_mps`, `dt_prev`, `arc_length_m` |

**3. Models** (`modeling/compare_models.py`)
- Six model families are compared with GroupKFold cross-validation, so a trip never appears in both training and validation.
- Metrics: Macro F1, observation-level Top-1 accuracy, log loss, route continuity and inference time.

**4. Bayesian optimisation** (`modeling/tune_model.py`)
- Optuna TPE search over learning rate, iterations, tree size, minimum leaf size, L2 regularisation and feature subsampling.
- The selected configuration is refit on all reliable training rows and exported to `artifacts/`.

**5. HMM and Viterbi** (`hmm/viterbi.py`)

| HMM component | Definition |
|---|---|
| Hidden state | Which candidate arc the vehicle is on at each GPS point |
| Emission | `w · log P(candidate)`, where P comes from the classifier |
| Transition | `−|network distance − GPS distance| / (β · Δt)`. Network distance is the directed shortest path between the projected positions on two arcs (Dijkstra). |
| Impossible moves | A fixed large penalty instead of −∞, so the decoder can recover after a GPS jump |
| Gaps | Points with no candidates are skipped; β is scaled by the time gap |

- Decoder parameters were tuned on **12 training trips held out from classifier training**, so the probabilities behave like those on unseen data (best: β·w = 40).
- Only the product β·w affects the decoded path.

---

## Repository structure

| Path | Contents |
|---|---|
| `eda/` | Exploratory data analysis: trip summaries, data-quality checks, figures and Folium maps (see [`eda/README.md`](eda/README.md)) |
| `data_processing/build_candidates.py` | Per-trip UTM projection, candidate generation, route alignment and labels |
| `features/build_features.py` | Feature engineering |
| `modeling/compare_models.py` | Six-model comparison, CV, student-test evaluation and plots |
| `modeling/tune_model.py` | Optuna Bayesian optimisation and final model export |
| `modeling/common.py` | Shared metrics (Top-1, route continuity) and sampling |
| `models/` | Earlier model-training scripts (first iteration) |
| `hmm/viterbi.py` | HMM definition and Viterbi decoder |
| `hmm/evaluate_hmm.py` | Decoder tuning, classifier vs HMM comparison and result maps |
| `pipeline/inference.py` | Label-free prediction pipeline used by the API |
| `pipeline/scoring.py` | Server-side scoring against the reference route |
| `api/main.py` | FastAPI backend |
| `ui/app.py` | Streamlit evaluator interface |
| `visualization/maps.py` | Folium map builder |
| `artifacts/` | `model.joblib`, `model_config.json`, `hmm_config.json` (the files needed for inference) |
| `results/` | All CSV results, experiment logs, plots and maps used in the report |

**Key result files:**
- `results/models/model_comparison.csv`, `cv_folds.csv`: the model comparison table and per-fold CV results.
- `results/models/*.png`: CV Macro F1, test Top-1/F1, log loss, inference time, confusion matrices, and loss curves (MLP and HistGradientBoosting only, since only they have epoch-wise histories).
- `results/tuning/optuna_trials.csv`, `best_params.json`, `tuned_vs_untuned.csv`, `optimisation_history.png`.
- `results/hmm/student_test_hmm_summary.csv`, `student_test_hmm_per_trip.csv`, `hmm_grid_search.csv`.
- `results/maps/*.html`: interactive maps with GPS points, road network, reference route, classifier route and HMM route. Download and open them in a browser, because GitHub does not render them.

---

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/puravk376/PathSnap.git
cd PathSnap
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Run from the repo root. The full pipeline takes about 20–25 minutes on a laptop.

```bash
# 1. Exploratory data analysis
python -m eda.run_eda

# 2. Candidates and labels
python data_processing/build_candidates.py data/train        results/candidates_train.parquet 50
python data_processing/build_candidates.py data/student_test results/candidates_student_test.parquet 50

# 3. Features
python features/build_features.py results/candidates_train.parquet        data/train        results/features_train.parquet
python features/build_features.py results/candidates_student_test.parquet data/student_test results/features_student_test.parquet

# 4. Six-model comparison
python modeling/compare_models.py results/features_train.parquet results/features_student_test.parquet data/student_test

# 5. Bayesian optimisation and final model export
python modeling/tune_model.py results/features_train.parquet results/features_student_test.parquet data/student_test --trials 25

# 6. HMM decoder tuning, classifier vs HMM comparison, maps
python hmm/evaluate_hmm.py results/features_train.parquet data/train data/student_test
```

All random seeds are fixed (`42`), so reruns give the same or very close results.

---

## Running the demo (two laptops)

Both laptops must be on the same network. A phone hotspot is the most reliable option, because college Wi-Fi often blocks connections between devices.

### Laptop 1: backend

Needs the repo, `artifacts/` and `data/evaluator_test/`.

```bash
# Windows
set PATHSNAP_EVAL_DIR=data\evaluator_test
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Mac/Linux
PATHSNAP_EVAL_DIR=data/evaluator_test uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- Allow Python through the firewall, and set the Wi-Fi network to **Private** on Windows.
- Find the laptop's IP with `ipconfig` (Windows) or `ifconfig` (Mac/Linux).
- The terminal logs every request, prediction and score.

| Endpoint | Description |
|---|---|
| `GET /health` | Server status, loaded model and HMM parameters |
| `GET /trips` | Evaluator trip IDs available on the server |
| `POST /predict` | Body `{"trip_id": "00000015"}`. Runs the full pipeline and returns the GPS points, the predicted route geometry, the nearby road network and server-computed metrics. |

Interactive API docs are at `http://<laptop-1-IP>:8000/docs`.

### Laptop 2: UI

Needs only the `ui/` folder and four packages.

```bash
pip install streamlit requests folium pandas
streamlit run ui/app.py
```

1. In the sidebar, enter `http://<laptop-1-IP>:8000` and click **Connect / refresh trips**.
2. Select a trip and click **Run prediction**.

The UI shows accuracy, continuity and route F1 for both the classifier-only and HMM routes, plus a Folium map with the GPS points, the road network and the predicted routes.

To check the connection, open `http://<laptop-1-IP>:8000/health` in a browser on laptop 2.

---

## Leakage safeguards

- **No labels in features.** Every feature is computed from GPS points and road geometry only.
- **Prediction never reads `.route`.** `pipeline/inference.py` only opens `.track`, `.nodes` and `.arcs`.
- **Reference routes stay server-side.** They are read only in `pipeline/scoring.py`, after the prediction is finished, to compute metrics. They are never sent to the UI.
- **No evaluator data in development.** The evaluator-test partition was not used for training, tuning, thresholds or model selection.
- **Group-aware validation.** All cross-validation uses GroupKFold by trip, so no trip appears in both training and validation.
- **No hard-coded predictions.** Every result shown in the UI is computed live by the API.

## Limitations

- Some training reference routes are inaccurate. Affected points are excluded from training, but they reduce the training data.
- Direction features are unavailable when the vehicle is nearly stationary.
- The HMM can smoothly follow a stretch where the classifier is confidently wrong (trip 00000077 remains the hardest student-test trip, at 75% accuracy).
- The drawn route connects predicted arcs only. Arcs driven between two GPS points without a point on them are not filled in.

---

## Team

| Member | Contribution |
|---|---|
| _Pramiti Udupa [PES1UG24CS331]_ | _e.g. EDA and data-quality analysis_ |
| _Purav N Khincha [PES1UG24CS349]_ | _e.g. candidate generation, labelling, feature engineering_ |
| _Pushkar Kulkarni [PES1UG24CS350]_ | _e.g. model training and evaluation_ |
| _Prakruti B_ | _e.g. API, UI and deployment_ |

## AI assistance

_Describe here which parts were developed with AI assistance, as required by the hackathon guidelines._
