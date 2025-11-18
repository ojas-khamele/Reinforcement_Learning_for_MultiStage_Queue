import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ============================================================
# CONFIG
# ============================================================
LOG_DIR = "logs"

ALGORITHMS = ["a2c", "dyna", "ppo", "r2d2"]
SEEDS = [10, 20, 30, 40, 50]

LOSS_MAP = {
    "r2d2": ["Loss"],
    "ppo": ["ActorLoss", "CriticLoss"],
    "a2c": ["ActorLoss", "CriticLoss"],
    "dyna": ["TD_Error"]
}

# Reward baseline values
BASE_MEAN = 37.34
BASE_STD = 6.83
BASE_MIN = 23.30
BASE_MAX = 48.12

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_log(algorithm, seed):
    path = os.path.join(LOG_DIR, f"{algorithm}_training_log_{seed}.csv")
    return pd.read_csv(path)

def moving_average(data, window):
    return data.groupby(data.index // window).mean()

# ============================================================
# DASH APP
# ============================================================
app = Dash(__name__)
app.title = "RL Reward & Loss Visualization"

app.layout = html.Div([
    html.H2("RL Reward & Loss Visualization Tool", style={'text-align': 'center'}),

    html.Label("Select Metric:"),
    dcc.Dropdown(
        id='metric-dropdown',
        options=[
            {'label': 'Loss', 'value': 'loss'},
            {'label': 'Reward', 'value': 'reward'}
        ],
        value='reward'
    ),

    html.Label("Select Mode:"),
    dcc.Dropdown(
        id='mode-dropdown',
        options=[
            {'label': 'Compare across seeds', 'value': 'all_seeds'},
            {'label': 'Single seed', 'value': 'single_seed'}
        ],
        value='all_seeds'
    ),

    html.Label("Select Algorithm:"),
    dcc.Dropdown(
        id='algo-dropdown',
        options=[{'label': a.upper(), 'value': a} for a in ALGORITHMS],
        value='ppo'
    ),

    html.Label("Select Seed (only for single seed mode):"),
    dcc.Dropdown(
        id='seed-dropdown',
        options=[{'label': s, 'value': s} for s in SEEDS],
        value=10
    ),

    html.Label("Averaging window:"),
    dcc.Input(id='window-input', type='number', value=50, min=1, step=1),

    html.Br(), html.Br(),

    dcc.Graph(id='rl-graph')
])

# ============================================================
# CALLBACK
# ============================================================
@app.callback(
    Output('rl-graph', 'figure'),
    Input('metric-dropdown', 'value'),
    Input('mode-dropdown', 'value'),
    Input('algo-dropdown', 'value'),
    Input('seed-dropdown', 'value'),
    Input('window-input', 'value')
)
def update_graph(metric, mode, algorithm, seed, window):
    fig = go.Figure()

    if metric == 'loss':
        # LOSS plotting
        if mode == 'all_seeds':
            for loss_col in LOSS_MAP[algorithm]:
                for s in SEEDS:
                    df = load_log(algorithm, s)
                    avg_loss = moving_average(df[loss_col], window)
                    fig.add_trace(go.Scatter(
                        y=avg_loss,
                        mode='lines',
                        name=f"{loss_col} - Seed {s}"
                    ))
            fig.update_layout(title=f"Loss Across Seeds ({algorithm.upper()})")
        else:  # single seed
            df = load_log(algorithm, seed)
            for loss_col in LOSS_MAP[algorithm]:
                avg_loss = moving_average(df[loss_col], window)
                fig.add_trace(go.Scatter(
                    y=avg_loss,
                    mode='lines',
                    name=f"{loss_col}"
                ))
            fig.update_layout(title=f"Loss Plot ({algorithm.upper()} - Seed {seed})")

    else:
        # REWARD plotting
        if mode == 'all_seeds':
            for s in SEEDS:
                df = load_log(algorithm, s)
                avg_reward = moving_average(df["Reward"], window)
                fig.add_trace(go.Scatter(
                    y=avg_reward,
                    mode='lines',
                    name=f"Seed {s}"
                ))
        else:
            df = load_log(algorithm, seed)
            avg_reward = moving_average(df["Reward"], window)
            fig.add_trace(go.Scatter(
                y=avg_reward,
                mode='lines',
                name=f"{algorithm.upper()} - Seed {seed}"
            ))

        # Add baseline mean and shaded STD
        x_points = np.arange(len(avg_reward))
        fig.add_trace(go.Scatter(
            y=[BASE_MEAN]*len(x_points),
            mode='lines',
            line=dict(dash='dash', color='black'),
            name='Baseline Mean'
        ))
        fig.add_trace(go.Scatter(
            y=[BASE_MAX]*len(x_points),
            mode='lines',
            line=dict(dash='dot', color='green'),
            name='Baseline Max'
        ))
        fig.add_trace(go.Scatter(
            y=[BASE_MIN]*len(x_points),
            mode='lines',
            line=dict(dash='dot', color='red'),
            name='Baseline Min'
        ))
        # Shaded STD area
        fig.add_trace(go.Scatter(
            y=[BASE_MEAN+BASE_STD]*len(x_points),
            fill=None,
            mode='lines',
            line_color='lightgray',
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            y=[BASE_MEAN-BASE_STD]*len(x_points),
            fill='tonexty',
            mode='lines',
            line_color='lightgray',
            fillcolor='rgba(128,128,128,0.2)',
            name='Baseline ± STD'
        ))

    fig.update_layout(
        xaxis_title=f"Step (averaged over {window})",
        yaxis_title="Loss" if metric=='loss' else "Reward",
        template="plotly_white"
    )

    return fig

# ============================================================
# RUN SERVER
# ============================================================
if __name__ == '__main__':
    app.run(debug=True)
