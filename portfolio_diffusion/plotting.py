from __future__ import annotations

import matplotlib.pyplot as plt


def plot_wealth(bt_df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bt_df.index, bt_df["Generative_Markowitz_Wealth"], label="Generative Markowitz")
    ax.plot(bt_df.index, bt_df["Equal_Weight_Wealth"], label="1/N Equal Weight")
    ax.set_title("Wealth Trajectory: Generative Markowitz vs 1/N")
    ax.set_xlabel("Date")
    ax.set_ylabel("Wealth")
    ax.legend()
    ax.grid(True)
    return fig, ax


def plot_average_weights(avg_weights_df, top_n=20):
    plot_df = avg_weights_df.head(top_n).sort_values("Average_GM_Weight")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(plot_df["Industry"], plot_df["Average_GM_Weight"])
    ax.set_title(f"Top {top_n} Average Generative Markowitz Weights")
    ax.set_xlabel("Average Weight")
    ax.set_ylabel("Industry")
    ax.grid(True)
    return fig, ax


def plot_turnover(turnover_df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(turnover_df.index, turnover_df["Turnover"])
    ax.set_title("Generative Markowitz Monthly Turnover")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")
    ax.grid(True)
    return fig, ax
