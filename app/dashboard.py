"""Aegis investigator dashboard: streamlit run app/dashboard.py"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import polars as pl
import streamlit as st

REPORTS = Path(__file__).resolve().parent.parent / "reports"

st.set_page_config(page_title="Aegis", layout="wide")


@st.cache_data
def load():
    alerts = pl.read_csv(REPORTS / "top_alerts.csv", try_parse_dates=True,
                         schema_overrides={"from_bank": pl.String, "to_bank": pl.String})
    cases = pl.read_csv(REPORTS / "cases.csv", try_parse_dates=True)
    accounts = pl.read_csv(REPORTS / "account_risk.csv")
    importance = pl.read_csv(REPORTS / "feature_importance.csv")
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    return alerts, cases, accounts, importance, metrics


def network_figure(case_alerts: pl.DataFrame) -> go.Figure:
    g = nx.DiGraph()
    for r in case_alerts.iter_rows(named=True):
        g.add_edge(r["sender_key"], r["receiver_key"], usd=r["amount_usd"], risk=r["risk_score"])
    pos = nx.kamada_kawai_layout(g.to_undirected())

    arrows = [
        dict(ax=pos[u][0], ay=pos[u][1], x=pos[v][0], y=pos[v][1], xref="x", yref="y",
             axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.5,
             arrowcolor="#c2410c" if d["risk"] >= 0.5 else "#94a3b8", opacity=0.8)
        for u, v, d in g.edges(data=True)
    ]
    volume = [sum(d["usd"] for *_, d in g.in_edges(n, data=True))
              + sum(d["usd"] for *_, d in g.out_edges(n, data=True)) for n in g.nodes]
    top = max(volume) or 1
    fig = go.Figure(go.Scatter(
        x=[pos[n][0] for n in g.nodes], y=[pos[n][1] for n in g.nodes], mode="markers",
        marker=dict(size=[12 + 26 * v / top for v in volume], color="#60a5fa",
                    line=dict(width=1, color="#ffffff")),
        text=[f"{n}<br>${v:,.0f} moved" for n, v in zip(g.nodes, volume)], hoverinfo="text",
    ))
    fig.update_layout(annotations=arrows, showlegend=False, height=460,
                      margin=dict(l=0, r=0, t=0, b=0),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


alerts, cases, accounts, importance, metrics = load()
st.title("Aegis")
st.caption(f"Anti-money-laundering alert review, {metrics['dataset']} hold-out "
           f"({metrics['splits']['test']['rows']:,} transactions)")
overview, queue, case_view, model_view = st.tabs(["Overview", "Alert queue", "Case review", "Model"])

with overview:
    a, b, c, d = st.columns(4)
    a.metric("Alerts in queue", f"{len(alerts):,}")
    b.metric("Confirmed laundering", f"{alerts['is_laundering'].sum():,}")
    c.metric("Cases", f"{len(cases):,}")
    d.metric("Median alert value", f"${alerts['amount_usd'].median():,.0f}")
    st.subheader("Largest cases")
    st.dataframe(cases.head(15), width="stretch", hide_index=True)
    st.subheader("Highest-risk accounts")
    st.dataframe(accounts.head(15), width="stretch", hide_index=True)

with queue:
    left, right = st.columns([1, 2])
    min_risk = left.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.05)
    formats = right.multiselect("Payment method", sorted(alerts["payment_format"].unique().to_list()))
    shown = alerts.filter(pl.col("risk_score") >= min_risk)
    if formats:
        shown = shown.filter(pl.col("payment_format").is_in(formats))
    st.caption(f"{len(shown):,} alerts")
    st.dataframe(
        shown.select("case_id", "risk_score", "timestamp", "sender_key", "receiver_key", "amount_usd",
                     "payment_format", "reasons", "is_laundering", "pattern"),
        width="stretch", hide_index=True, height=560,
    )

with case_view:
    options = cases.sort("case_id")["case_id"].to_list()
    case_id = st.selectbox("Case", options, format_func=lambda i: (
        lambda r: f"Case {i}: {r['alerts'][0]} alerts, {r['accounts'][0]} accounts"
    )(cases.filter(pl.col("case_id") == i)))
    selected = alerts.filter(pl.col("case_id") == case_id).sort("timestamp")
    row = cases.filter(pl.col("case_id") == case_id)
    a, b, c, d = st.columns(4)
    a.metric("Alerts", row["alerts"][0])
    b.metric("Accounts", row["accounts"][0])
    c.metric("Value", f"${row['total_usd'][0]:,.0f}")
    d.metric("Confirmed laundering", f"{row['confirmed_laundering'][0]} of {row['alerts'][0]}")
    st.plotly_chart(network_figure(selected), width="stretch")
    st.subheader("Timeline and reasons")
    st.dataframe(
        selected.select("timestamp", "sender_key", "receiver_key", "amount_usd", "payment_format",
                        "risk_score", "reasons"),
        width="stretch", hide_index=True,
    )

with model_view:
    rows = [
        {"model": name, "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
         "precision@100": m["precision@100"], "recall@1000": m["recall@1000"],
         "recall@5000": m["recall@5000"]}
        for name, m in metrics["models"].items()
    ]
    st.dataframe(pl.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("Most important features")
    top = importance.head(15).reverse()
    fig = go.Figure(go.Bar(x=top["share"].to_list(), y=top["feature"].to_list(), orientation="h",
                           marker_color="#60a5fa"))
    fig.update_layout(height=460, margin=dict(l=0, r=0, t=0, b=0), xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
