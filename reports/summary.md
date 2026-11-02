# Results on LI-Small

Test period: 2022-09-08 16:06:00 to 2022-09-10 23:59:00, 1,384,828 transactions, 791 laundering.

| Model | PR-AUC | ROC-AUC | Precision@100 | Recall@100 | Precision@500 | Recall@500 | Precision@1000 | Recall@1000 | Precision@5000 | Recall@5000 |
|---|---|---|---|---|---|---|---|---|---|---|
| legacy_isolation_forest | 0.0005 | 0.4562 | 0.020 | 0.003 | 0.004 | 0.003 | 0.003 | 0.004 | 0.001 | 0.006 |
| isolation_forest | 0.0017 | 0.7935 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.003 | 0.002 | 0.010 |
| lightgbm | 0.2335 | 0.9607 | 1.000 | 0.126 | 0.334 | 0.211 | 0.197 | 0.249 | 0.071 | 0.450 |

## LightGBM recall by typology (top alerts)

| Typology | Transactions | Caught | Recall |
|---|---|---|---|
| BIPARTITE | 38 | 34 | 0.89 |
| CYCLE | 19 | 17 | 0.89 |
| FAN-IN | 7 | 6 | 0.86 |
| FAN-OUT | 28 | 23 | 0.82 |
| GATHER-SCATTER | 32 | 28 | 0.88 |
| RANDOM | 20 | 18 | 0.90 |
| SCATTER-GATHER | 46 | 36 | 0.78 |
| STACK | 19 | 15 | 0.79 |

## Top features (LightGBM gain)

| Feature | Share |
|---|---|
| s_secs_since_prev_out | 13.7% |
| amount_vs_sender_mean | 8.0% |
| log_amount_usd | 7.7% |
| payment_format_code | 7.3% |
| amount_vs_receiver_mean | 5.7% |
| r_secs_since_prev_in | 5.3% |
| s_is_new_receiver | 5.2% |
| s_distinct_receivers | 4.8% |
| r_in_sum_24h | 4.6% |
| s_last_in_amount_ratio | 4.5% |
| s_secs_since_last_in | 3.9% |
| s_out_count_24h | 3.7% |
