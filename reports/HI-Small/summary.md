# Results on HI-Small

Test period: 2022-09-08 16:09:00 to 2022-09-10 23:59:00, 1,015,745 transactions, 1,143 laundering.

| Model | PR-AUC | ROC-AUC | Precision@100 | Recall@100 | Precision@500 | Recall@500 | Precision@1000 | Recall@1000 | Precision@5000 | Recall@5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy_isolation_forest | 0.0011 | 0.5177 | 0.000 | 0.000 | 0.000 | 0.000 | 0.005 | 0.004 | 0.004 | 0.016 |
| isolation_forest | 0.0025 | 0.7490 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.001 | 0.006 |
| lightgbm | 0.5682 | 0.9807 | 0.990 | 0.087 | 0.964 | 0.422 | 0.595 | 0.521 | 0.158 | 0.691 |

## LightGBM recall by typology (top alerts)

| Typology | Transactions | Caught | Recall |
|---|---|---|---|
| BIPARTITE | 52 | 46 | 0.88 |
| CYCLE | 65 | 61 | 0.94 |
| FAN-IN | 71 | 67 | 0.94 |
| FAN-OUT | 74 | 73 | 0.99 |
| GATHER-SCATTER | 147 | 137 | 0.93 |
| RANDOM | 45 | 43 | 0.96 |
| SCATTER-GATHER | 123 | 118 | 0.96 |
| STACK | 107 | 98 | 0.92 |

## Top features (LightGBM gain)

| Feature | Share |
|---|---|
| s_is_new_receiver | 35.0% |
| s_secs_since_prev_out | 10.4% |
| log_amount_usd | 7.6% |
| s_secs_since_last_in | 5.5% |
| payment_format_code | 4.9% |
| s_last_in_amount_ratio | 4.8% |
| amount_vs_sender_mean | 3.8% |
| r_secs_since_prev_in | 3.4% |
| amount_vs_receiver_mean | 2.8% |
| r_in_sum_24h | 2.4% |
| s_distinct_receivers | 2.3% |
| weekday | 1.8% |
