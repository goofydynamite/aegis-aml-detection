# Project Aegis: Anti-Money-Laundering Detection on Transaction Networks

Aegis ranks financial transactions by money-laundering risk so an investigation team
can review the riskiest first. It is trained and evaluated on the
[IBM AML dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
(Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering
Models*, NeurIPS 2023): 6.9 million transactions, 706k accounts, 0.05% laundering.

## Results (LI-Small, chronological hold-out)

Test period 8–10 Sept 2022: 1,384,828 transactions, 791 laundering (1 in 1,750).

| Model | PR-AUC | ROC-AUC | Precision@100 | Recall@1000 | Recall@5000 |
|---|---|---|---|---|---|
| Legacy prototype (Isolation Forest, per-transaction features) | 0.0005 | 0.456 | 0.02 | 0.004 | 0.006 |
| Isolation Forest, behavioural + cycle features | 0.0017 | 0.794 | 0.00 | 0.003 | 0.010 |
| **LightGBM, behavioural + cycle features** | **0.234** | **0.961** | **1.00** | **0.249** | **0.450** |

- **All of the top 100 alerts are laundering.** 100 random alerts would contain about 0.06.
- The top 5,000 alerts (0.36% of transactions) contain 45% of all laundering.
- Grouping the top 1,000 alerts helps: cases with 3+ alerts are 91% confirmed laundering
  (79 of 87 alerts), against 11% for single-alert cases.
- Of transactions linked to a known typology, 85% land in the top 5,000. Recall per
  typology ranges from 90% (random) to 78% (scatter-gather), in
  [reports/summary.md](reports/summary.md).
- The legacy approach from the first version of this project performs no better than random
  on real-scale data: single-transaction features cannot see network patterns.

Accuracy is not reported: predicting "not laundering" for everything scores 99.95%.

## Results on unseen data (HI-Small)

HI-Small is a separate simulated world with twice the share of laundering. Test period:
1,015,745 transactions, 1,143 laundering (1 in 889). All settings were kept from LI-Small.

| Model | PR-AUC | ROC-AUC | Precision@100 | Precision@1000 | Recall@1000 | Recall@5000 |
|---|---|---|---|---|---|---|
| **LI-Small model, applied unchanged** | **0.492** | **0.977** | **0.99** | **0.51** | **0.442** | **0.665** |
| LightGBM retrained on HI-Small | 0.568 | 0.981 | 0.99 | 0.60 | 0.521 | 0.691 |
| Legacy prototype, retrained | 0.001 | 0.518 | 0.00 | 0.01 | 0.004 | 0.016 |

- The first row is the cleanest result in this project: this model never saw HI-Small, and
  these scores were not used for any decision.
- Part of the higher PR-AUC comes from the higher base rate. Compared with random ranking,
  the lift is similar: about 440x on HI-Small, about 410x on LI-Small.
- Both datasets come from the same simulator, so this shows the model carries across
  datasets, not that it would work on real bank data.

Reproduce with:

```bash
python -m aegis all --config config.hi-small.yaml
python scripts/evaluate_transfer.py --config config.hi-small.yaml --model models/LI-Small_lightgbm.txt
```

## Live scoring API

The model's features depend on each account's history, so the service keeps that history in
memory ([src/aegis/online.py](src/aegis/online.py)) and updates it with every payment.

```bash
python -m aegis state                              # build the account memory (about 5 minutes)
uvicorn --factory --app-dir src aegis.api:load_app --port 8000
```

Interactive docs are at `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `POST /score` | Score one payment: risk, alert flag and what triggered it, reasons, cycle length, chain depth |
| `POST /score/batch` | Score up to 1,000 payments |
| `GET /accounts/{bank}/{account}` | Account history summary |
| `GET /health` | Status, accounts and payments loaded, alert threshold |

- **Matches training:** `scripts/check_online_parity.py` replays all 6.9M payments through
  the live engine and compares the last 300,000 with the training features. All 30 features
  match on every row. That check also found a same-minute look-ahead leak in the training
  windows, now fixed; scores moved very little.
- **Speed on this laptop:** median 20 ms, p95 27 ms, p99 42 ms, about 47 requests/s, one
  request at a time, with 705,905 accounts loaded.
- **Cycle rule:** the model gives cycles little weight, so a payment closing a cycle of 3+
  hops always alerts (`rules.cycle_alert_min_length` in `config.yaml`). Each alert lists what
  raised it in `triggered_by`. A new 6-hop test cycle scored only 0.029, so the rule raised it.

  | Data | Extra alerts from the rule | Of which laundering |
  |---|---|---|
  | LI-Small validation (used to choose 3 hops) | 4 | 1 |
  | LI-Small test | 7 | 2 |
  | HI-Small test, LI-Small model unchanged | 14 | 13 |

  Cycles of 2 (money straight back) were rejected: 934 extra validation alerts, 3% laundering.
  The counts are small, so the rule's precision is not well measured; its cost is about 1%
  more alerts. Reproduce with `scripts/evaluate_cycle_rule.py`.
- **Alert threshold:** the score of the 1,000th-highest validation payment, about 500 alerts a day.
- **Not production-ready:** there is no authentication, so keep it on localhost. History
  uses about 3.9 GB of RAM in one process, is lost on restart, and needs a single worker.
  Payments must arrive in time order. A real deployment would keep this state in a shared
  store such as Redis.

## How it works

```
raw CSV ──ingest──▶ transactions.parquet ──features──▶ features.parquet ──train──▶ model + reports
```

1. **Ingest** ([src/aegis/data.py](src/aegis/data.py)): typed load, deduplication, currency
   conversion to USD (rates estimated from the data's own cross-currency payments),
   integer account IDs, typology labels joined from `*_Patterns.txt`, and removal of the
   dataset's near-empty trailing days (see *Data caveats*).
2. **Features** ([src/aegis/features.py](src/aegis/features.py),
   [src/aegis/cycles.py](src/aegis/cycles.py)): 30 features, each
   computed only from the current and earlier transactions, so the same code can score a
   live stream. A test enforces this.
   - *Sender behaviour*: payment count, 24h velocity and volume, distinct receivers so far,
     time since previous payment, amount relative to the sender's own average.
   - *Receiver behaviour*: the same, from the incoming side (fan-in).
   - *Flow-through*: time since the sender last received money and the ratio of this
     payment to that inflow. These catch pass-through and stack layering.
   - *Hub signals*: how many distinct accounts paid the sender, and how many the receiver
     has paid (gather and scatter nodes).
   - *Cycles and chains*: payments are replayed in time order and each account remembers
     which accounts could have funded it in the last 96 hours, and how many hops away.
     A payment back to one of them closes a cycle of any length.
3. **Train and evaluate** ([src/aegis/model.py](src/aegis/model.py),
   [src/aegis/evaluate.py](src/aegis/evaluate.py)): 60/20/20 chronological split, LightGBM
   with early stopping on validation PR-AUC, two Isolation Forest baselines, and
   precision/recall@k metrics that match how a fixed-size investigation team works.
4. **Alerts** ([src/aegis/explain.py](src/aegis/explain.py),
   [src/aegis/cases.py](src/aegis/cases.py)): each of the top 1,000 alerts gets its top three
   plain-language reasons; alerts sharing an account are merged into cases, and risk is
   rolled up per account.
5. **Dashboard** ([app/dashboard.py](app/dashboard.py)): Streamlit app with the alert
   queue, a network view and timeline for each case, and model performance.

## Project structure

```
aegis/
├── config.yaml              # dataset, split, model and evaluation settings
├── config.hi-small.yaml     # the same settings for HI-Small
├── app/dashboard.py         # Streamlit investigator dashboard
├── src/aegis/
│   ├── __main__.py          # CLI: python -m aegis {ingest,features,train,alerts,state,all}
│   ├── config.py
│   ├── data.py              # loading, cleaning, currency conversion, typology labels
│   ├── features.py          # leakage-safe behavioural and network features
│   ├── cycles.py            # cycle and chain detection of any length
│   ├── graph.py             # two-hop path features (built, not used by the model)
│   ├── online.py            # the same features, one live payment at a time
│   ├── api.py               # FastAPI scoring service
│   ├── model.py             # time split, LightGBM, Isolation Forest
│   ├── explain.py           # plain-language reasons per alert
│   ├── cases.py             # case grouping and account roll-up
│   └── evaluate.py          # PR-AUC, precision/recall@k, per-typology recall
├── scripts/                 # feature-set, cross-dataset, live-parity and load checks
├── tests/                   # pytest suite, including a no-look-ahead test
├── data/raw/                # downloaded files (git-ignored)
├── data/processed/          # Parquet outputs (git-ignored)
├── models/                  # trained models (git-ignored)
└── reports/                 # metrics.json, summary.md, feature_importance.csv, top_alerts.csv
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,app,api]"
```

Download `LI-Small_Trans.csv` and `LI-Small_Patterns.txt` into `data/raw/` (see
[data/raw/README.md](data/raw/README.md)). Then:

```bash
python -m aegis all               # ~6-12 minutes on a laptop; or run ingest / features / train / alerts
pytest
streamlit run app/dashboard.py    # investigator dashboard
```

To use another dataset version, change `data.dataset` in `config.yaml` (e.g. `HI-Small`).

## Outputs

| File | Contents |
|---|---|
| `reports/summary.md` | Results table, recall by typology, top features |
| `reports/metrics.json` | All metrics and split details, machine-readable |
| `reports/top_alerts.csv` | The 1,000 highest-risk test transactions with accounts, scores, reasons and case |
| `reports/cases.csv` | One row per case: alerts, accounts, value, time span, confirmed laundering |
| `reports/account_risk.csv` | Risk rolled up per account |
| `reports/feature_importance.csv` | LightGBM gain per feature |
| `models/LI-Small_lightgbm.txt` | Trained model |

## Data caveats

- **Sparse tail.** After 10 Sept the raw file has only 223 transactions, 61% of them
  laundering (attempts that run past the simulation end). Left in, they let a model learn
  "a quiet period means laundering" and inflate the test PR-AUC from 0.23 to 0.34. Ingest
  drops trailing days with under 1% of a median day's volume (`data.min_day_share`).
- **Typology coverage.** Only about a quarter of laundering transactions appear in the
  patterns file. Per-typology recall uses only those, which is why it is higher than overall recall.
- **Same-minute ordering.** Timestamps are per minute. Within a minute, 24h windows follow
  file order; look-ups of an account's previous inflow or outflow skip that minute entirely.

## Known limits

- Network features add little to this model. In a fixed-tree, two-seed comparison on
  validation data, run before the same-minute fix (PR-AUC): behavioural 0.276, + cycles 0.284, + two-hop paths 0.279, both
  0.275. Seed noise is up to about 0.007, and recall@k did not change. Cycles are kept for
  their small gain and because investigators can read them; two-hop paths are left out.
- Cycle detection can only flag the final payment. Earlier legs are not a cycle yet when
  they happen. It finds about 7 of the 12 cycles in the file, including 10 and 11-hop ones.
- Most alerts form single-alert cases (851 of 897 cases), and those are mostly false alarms.
- A few very large payments dominate alert value, so the dashboard shows the median instead.
- LI-Small test scores were seen during development, though feature choices were made on
  validation data. HI-Small above is the untouched check.

## Roadmap

- Graph neural network (GraphSAGE / GAT) comparison
- LLM-written alert summaries generated from the structured evidence only
