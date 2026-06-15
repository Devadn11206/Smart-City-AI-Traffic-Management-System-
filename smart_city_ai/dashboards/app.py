from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traci_controller import SumoController
import time
import numpy as np
import torch
from pinn_models.pinn_model import TrafficPINN
from utils.reliability import extract_state_dict, load_torch_checkpoint

REPO_ROOT = Path(PROJECT_ROOT)
RESULTS_DIR = REPO_ROOT / 'results'
EMERGENCY_DIR = RESULTS_DIR / 'emergency_metrics'
ACO_DIR = RESULTS_DIR / 'aco_results'
HYBRID_DIR = RESULTS_DIR / 'hybrid_logs'
CHART_DIR = REPO_ROOT / 'visualizations' / 'comparison_charts'


def _latest_json(directory: Path) -> dict:
    if not directory.exists():
        return {}
    files = sorted(directory.glob('*_latest.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return {}
    try:
        return json.loads(files[0].read_text(encoding='utf-8'))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _launch(command: list[str], title: str) -> None:
    subprocess.Popen(command, cwd=str(REPO_ROOT), shell=False)
    st.success(f'{title} launched in the background.')


def _port_open(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def _rl_latest_json() -> dict:
    return _latest_json(RESULTS_DIR / 'rl' / 'telemetry')


def _rl_timeseries_df() -> pd.DataFrame:
    path = RESULTS_DIR / 'rl' / 'telemetry_timeseries.csv'
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_torch_load(path: Path | str):
    """Load a trusted local checkpoint with the same path used by the backend."""
    return load_torch_checkpoint(path, map_location='cpu')


@st.cache_resource
def _load_pinn_model(model_path: str):
    p = Path(model_path)
    if not p.exists():
        st.warning(f'PINN checkpoint not found: {p}')
        return None
    try:
        model = TrafficPINN()
        ckpt = safe_torch_load(p)

        if not isinstance(ckpt, dict):
            raise TypeError(f'Invalid checkpoint format: expected dict, got {type(ckpt).__name__}')

        state_dict = extract_state_dict(ckpt)
        if not isinstance(state_dict, dict):
            raise TypeError('Checkpoint does not contain a valid state_dict')

        load_result = model.load_state_dict(state_dict, strict=False)
        if getattr(load_result, 'missing_keys', None) or getattr(load_result, 'unexpected_keys', None):
            print(
                'PINN checkpoint loaded with non-strict keys: '
                f"missing={getattr(load_result, 'missing_keys', [])}, "
                f"unexpected={getattr(load_result, 'unexpected_keys', [])}"
            )
        model.eval()
        return model
    except Exception as exc:
        st.warning(
            'PINN checkpoint could not be loaded. The dashboard will continue '\
            'without PINN overlays.'
        )
        st.exception(exc)
        return None


def _vehicle_grid(snapshot: dict, bins: int = 40):
    vehicles = snapshot.get('vehicles', {})
    if not vehicles:
        return None
    xs = np.array([v['pos'][0] for v in vehicles.values()], dtype=float)
    ys = np.array([v['pos'][1] for v in vehicles.values()], dtype=float)
    xmin, xmax = xs.min() - 40, xs.max() + 40
    ymin, ymax = ys.min() - 40, ys.max() + 40
    H, xe, ye = np.histogram2d(xs, ys, bins=bins, range=[[xmin, xmax], [ymin, ymax]])
    H = H.T
    if H.max() > 0:
        Hn = H / H.max()
    else:
        Hn = H
    return {
        'density': Hn,
        'xedges': xe,
        'yedges': ye,
        'xcenters': (xe[:-1] + xe[1:]) / 2.0,
        'ycenters': (ye[:-1] + ye[1:]) / 2.0,
        'xs': xs,
        'ys': ys,
        'vehicles': vehicles,
    }


def _pinn_overlay(snapshot: dict, horizons: list[int], bins: int = 40):
    model = _load_pinn_model(str(REPO_ROOT / 'pinn_models' / 'trained_pinn.pth'))
    grid = _vehicle_grid(snapshot, bins=bins)
    if model is None or grid is None:
        return None

    density = grid['density']
    xcenters = grid['xcenters']
    ycenters = grid['ycenters']
    t0 = float(snapshot.get('time', 0.0))

    x_min = float(xcenters.min())
    x_max = float(xcenters.max())
    x_norm = (xcenters - x_min) / max(x_max - x_min, 1e-6)

    overlays = []
    bottlenecks = []
    for h in horizons:
        t_norm = (t0 + float(h)) / 1000.0
        t_in = torch.tensor([[t_norm]], dtype=torch.float32).repeat(len(x_norm), 1)
        x_in = torch.tensor(x_norm.reshape(-1, 1), dtype=torch.float32)
        with torch.no_grad():
            rho, _ = model(t_in, x_in)
        pred_1d = rho.numpy().reshape(-1)
        if pred_1d.max() > 0:
            pred_1d = pred_1d / pred_1d.max()

        pred_2d = np.tile(pred_1d.reshape(1, -1), (len(ycenters), 1))
        conf_base = np.exp(-float(h) / 180.0)
        confidence = np.clip(conf_base * (0.35 + 0.65 * density), 0.0, 1.0)
        intensity = np.clip(0.45 * density + 0.55 * (pred_2d * confidence), 0.0, 1.0)

        thr = np.quantile(intensity, 0.92) if intensity.size else 1.0
        ys_idx, xs_idx = np.where(intensity >= thr)
        for yi, xi in zip(ys_idx.tolist(), xs_idx.tolist()):
            bottlenecks.append({
                'horizon': h,
                'x': float(xcenters[xi]),
                'y': float(ycenters[yi]),
                'risk': float(intensity[yi, xi]),
            })

        overlays.append({
            'horizon': h,
            'intensity': intensity,
            'confidence': confidence,
            'predicted': pred_2d,
        })

    return {
        'grid': grid,
        'overlays': overlays,
        'bottlenecks': pd.DataFrame(bottlenecks),
    }


def _pinn_propagation_figure(pinn_data: dict):
    grid = pinn_data['grid']
    overlays = pinn_data['overlays']
    bdf = pinn_data['bottlenecks']

    x = grid['xcenters']
    y = grid['ycenters']
    first = overlays[0]
    b0 = bdf[bdf['horizon'] == first['horizon']] if not bdf.empty else pd.DataFrame()

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=first['intensity'],
                x=x,
                y=y,
                colorscale='Turbo',
                zmin=0,
                zmax=1,
                colorbar={'title': 'Congestion Intensity'},
            ),
            go.Scatter(
                x=b0['x'] if not b0.empty else [],
                y=b0['y'] if not b0.empty else [],
                mode='markers',
                name='Future Bottlenecks',
                marker={'size': 8, 'color': 'red', 'symbol': 'x'},
            ),
        ]
    )

    frames = []
    for ov in overlays:
        hh = ov['horizon']
        b = bdf[bdf['horizon'] == hh] if not bdf.empty else pd.DataFrame()
        frames.append(
            go.Frame(
                name=str(hh),
                data=[
                    go.Heatmap(z=ov['intensity'], x=x, y=y, colorscale='Turbo', zmin=0, zmax=1),
                    go.Scatter(
                        x=b['x'] if not b.empty else [],
                        y=b['y'] if not b.empty else [],
                        mode='markers',
                        marker={'size': 8, 'color': 'red', 'symbol': 'x'},
                    ),
                ],
            )
        )

    fig.frames = frames
    fig.update_layout(
        title='Congestion Propagation (SUMO State + PINN Forecast)',
        xaxis_title='X',
        yaxis_title='Y',
        yaxis={'scaleanchor': 'x', 'scaleratio': 1},
        updatemenus=[
            {
                'type': 'buttons',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 450, 'redraw': True}, 'fromcurrent': True}],
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}],
                    },
                ],
            }
        ],
        sliders=[
            {
                'steps': [
                    {'label': f"+{f.name}s", 'method': 'animate', 'args': [[f.name], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}}]}
                    for f in frames
                ]
            }
        ],
        height=620,
    )
    return fig


def _latest_aco_summary() -> dict:
    p = ACO_DIR / 'compare_summary.json'
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _aco_pheromone_df() -> pd.DataFrame:
    return _read_csv(ACO_DIR / 'pheromones_final.csv')


def _scenario_defaults(name: str) -> dict:
    defaults = {
        'Normal traffic': {'duration': 1200, 'sim_speed': 4, 'bins': 34, 'refresh_ms': 1200},
        'Heavy congestion': {'duration': 2200, 'sim_speed': 5, 'bins': 46, 'refresh_ms': 1000},
        'Rush hour': {'duration': 2600, 'sim_speed': 6, 'bins': 52, 'refresh_ms': 900},
        'Highway bottleneck': {'duration': 1800, 'sim_speed': 6, 'bins': 48, 'refresh_ms': 900},
        'Accident response': {'duration': 1400, 'sim_speed': 4, 'bins': 44, 'refresh_ms': 1000},
        'Ambulance emergency': {'duration': 1200, 'sim_speed': 5, 'bins': 42, 'refresh_ms': 900},
        'Multiple emergencies': {'duration': 2600, 'sim_speed': 5, 'bins': 50, 'refresh_ms': 800},
    }
    return defaults.get(name, defaults['Normal traffic'])


def _build_alerts(snapshot: dict, rl_latest: dict, emergency_latest: dict) -> list[dict]:
    alerts = []
    vc = int(snapshot.get('vehicle_count', 0))
    avg_speed = float(snapshot.get('avg_speed', 0.0))
    queue = float(rl_latest.get('avg_queue', emergency_latest.get('queue', 0.0) or 0.0))
    eta = float(emergency_latest.get('eta_seconds', 0.0) or 0.0)
    err = int(snapshot.get('traci_errors', 0))
    if vc > 120:
        alerts.append({'severity': 'high', 'message': f'Heavy load detected: {vc} active vehicles'})
    if avg_speed < 2.0 and vc > 10:
        alerts.append({'severity': 'high', 'message': f'Potential gridlock: avg speed {avg_speed:.2f} m/s'})
    if queue > 12:
        alerts.append({'severity': 'medium', 'message': f'Queue pressure rising: {queue:.1f}'})
    if eta > 180:
        alerts.append({'severity': 'medium', 'message': f'Emergency ETA high: {eta:.1f}s'})
    if err > 0:
        alerts.append({'severity': 'low', 'message': f'TraCI loop reported {err} recoverable errors'})
    return alerts


st.set_page_config(page_title='Smart City AI Digital Twin', layout='wide')

theme_mode = st.sidebar.selectbox('Theme', ['Dark Futuristic', 'Default'], index=0)
fullscreen_mode = st.sidebar.checkbox('Fullscreen Digital Twin Mode', value=False)

if theme_mode == 'Dark Futuristic':
    st.markdown(
        """
        <style>
            :root {
                --bg0: #060b12;
                --bg1: #0d1622;
                --card: rgba(13, 26, 40, 0.85);
                --line: #27c1ff;
                --text: #d9ecff;
                --accent: #4df1c7;
                --warn: #ff6b6b;
            }
            .stApp { background: radial-gradient(circle at 20% 10%, #0f2236 0%, var(--bg0) 50%, #04070d 100%); color: var(--text); }
            [data-testid="stMetric"] {
                background: linear-gradient(135deg, rgba(20,40,64,.9), rgba(10,20,34,.9));
                border: 1px solid rgba(39,193,255,.35);
                border-radius: 12px;
                padding: 10px;
                box-shadow: 0 0 0 1px rgba(39,193,255,.1), 0 0 24px rgba(39,193,255,.08);
                animation: pulseGlow 3s ease-in-out infinite;
            }
            @keyframes pulseGlow {
                0% { box-shadow: 0 0 0 1px rgba(39,193,255,.08), 0 0 12px rgba(39,193,255,.05); }
                50% { box-shadow: 0 0 0 1px rgba(39,193,255,.25), 0 0 26px rgba(77,241,199,.14); }
                100% { box-shadow: 0 0 0 1px rgba(39,193,255,.08), 0 0 12px rgba(39,193,255,.05); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

if fullscreen_mode:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] { display: none; }
            .block-container { padding-top: 1rem; max-width: 100%; }
        </style>
        """,
        unsafe_allow_html=True,
    )

st.title('AI-Powered Smart City Transportation Digital Twin')
st.caption('SUMO + RL + PINN + ACO + Emergency Green Corridor analytics')

with st.sidebar:
    st.header('Control Panel')
    scenario = st.selectbox('Demo scenario', ['Normal traffic', 'Heavy congestion', 'Rush hour', 'Highway bottleneck', 'Accident response', 'Ambulance emergency', 'Multiple emergencies'])
    preset = _scenario_defaults(scenario)
    if st.button('Apply scenario preset'):
        st.session_state['duration'] = preset['duration']
        st.session_state['sim_speed'] = preset['sim_speed']
        st.session_state['overlay_bins'] = preset['bins']
        st.session_state['refresh_ms'] = preset['refresh_ms']

    duration = st.slider('Simulation duration', 20, 10000, st.session_state.get('duration', 2000), 20, key='duration')
    algorithm = st.multiselect('Algorithms', ['DQN', 'DDQN', 'PPO', 'PINN', 'ACO', 'Emergency'], default=['DDQN', 'PINN', 'ACO', 'Emergency'])
    sim_speed = st.slider('Simulation speed', 1, 10, st.session_state.get('sim_speed', 5), key='sim_speed')
    refresh_ms = st.slider('Dashboard refresh (ms)', 250, 3000, st.session_state.get('refresh_ms', 1000), 50, key='refresh_ms')
    overlay_stride = st.slider('PINN overlay update stride (SUMO steps)', 1, 30, st.session_state.get('overlay_stride', 4), 1, key='overlay_stride')
    st.write('Selected scenario:', scenario)
    st.write('Selected algorithms:', ', '.join(algorithm))
    st.markdown('---')
    st.subheader('Telemetry Service')
    telemetry_running = _port_open('127.0.0.1', 8765)
    st.metric('RL Telemetry API', 'Online' if telemetry_running else 'Offline')
    if not telemetry_running and st.button('Start RL Telemetry Server'):
        _launch([sys.executable, 'integration/rl_telemetry_server.py', '--host', '127.0.0.1', '--port', '8765'], 'RL telemetry server')

    if st.button('Run emergency scenario'):
        _launch([sys.executable, 'integration/run_emergency_system.py', '--duration', str(duration), '--pinn', 'pinn_models/trained_pinn.pth', '--metrics-output-dir', 'results/emergency_metrics', '--run-name', 'dashboard_emergency'], 'Emergency scenario')
    if st.button('Run RL comparison'):
        _launch([sys.executable, 'evaluation/compare_algorithms.py'], 'RL comparison')
    if st.button('Run full evaluation'):
        _launch([sys.executable, 'tools/final_evaluation.py'], 'Final evaluation')
    if st.button('Regenerate charts'):
        _launch([sys.executable, 'visualizations/emergency_comparison_plots.py'], 'Chart generation')

latest_emergency = _latest_json(EMERGENCY_DIR)
latest_aco = _latest_json(ACO_DIR / 'checkpoints')
final_eval = _read_csv(CHART_DIR / 'final_evaluation_results.csv')

# Initialize a singleton SumoController inside Streamlit session state
if 'sumo_controller' not in st.session_state:
    st.session_state['sumo_controller'] = SumoController()
controller: SumoController = st.session_state['sumo_controller']

# Controls for SUMO runtime
st.sidebar.markdown('---')
if controller.is_running():
    if st.sidebar.button('Pause simulation'):
        controller.pause()
    if st.sidebar.button('Stop simulation'):
        controller.stop()
else:
    if st.sidebar.button('Start simulation'):
        controller.start()


col1, col2, col3, col4 = st.columns(4)
col1.metric('Emergency vehicles', latest_emergency.get('emergency_vehicle_count', 0))
col2.metric('Active corridors', latest_emergency.get('active_corridor_tls', latest_emergency.get('active_corridors', 0)))
col3.metric('Avg speed', f"{latest_emergency.get('avg_speed', 0.0):.2f} m/s")
col4.metric('ETA', f"{latest_emergency.get('eta_seconds', 0.0):.1f} s")

left, right = st.columns(2)
with left:
    st.subheader('Emergency Analytics')
    emergency_csv = _read_csv(EMERGENCY_DIR / 'quick_test_metrics.csv')
    if not emergency_csv.empty and {'step', 'avg_speed', 'queue', 'corridor_efficiency'}.issubset(emergency_csv.columns):
        fig_em = px.line(emergency_csv, x='step', y=['avg_speed', 'queue', 'corridor_efficiency'], title='Emergency Response KPIs')
        st.plotly_chart(fig_em, use_container_width=True)
    elif latest_emergency:
        kpi_df = pd.DataFrame([
            {'metric': 'Average Speed', 'value': latest_emergency.get('avg_speed', 0.0)},
            {'metric': 'Queue', 'value': latest_emergency.get('queue', 0.0)},
            {'metric': 'Throughput', 'value': latest_emergency.get('throughput', 0.0)},
        ])
        st.plotly_chart(px.bar(kpi_df, x='metric', y='value', color='metric', title='Emergency Snapshot'), use_container_width=True)
    else:
        st.info('No emergency metrics found yet. Run the emergency scenario.')

with right:
    st.subheader('ACO / Hybrid Benchmark KPIs')
    summary = _latest_aco_summary()
    if summary:
        rows = []
        for mode, payload in summary.items():
            m = payload.get('metrics', {})
            rows.append({'mode': mode, 'avg_speed': m.get('avg_speed', 0.0), 'wait': m.get('wait', 0.0), 'queue': m.get('queue', 0.0)})
        sdf = pd.DataFrame(rows)
        st.plotly_chart(px.bar(sdf, x='mode', y='avg_speed', color='mode', title='ACO Mode Speed Comparison'), use_container_width=True)
        st.plotly_chart(px.line(sdf.melt(id_vars='mode', value_vars=['wait', 'queue']), x='mode', y='value', color='variable', title='Wait/Queue by Mode'), use_container_width=True)
    elif not final_eval.empty:
        st.dataframe(final_eval, use_container_width=True)
    else:
        st.info('No ACO summary found yet.')

st.subheader('Training & Comparison Artifacts')
artifacts = [
    HYBRID_DIR / 'hybrid_training_log.csv',
    HYBRID_DIR / 'training_reward_curves.png',
    CHART_DIR / 'comparison.png',
    CHART_DIR / 'eta_vs_efficiency.png',
    CHART_DIR / 'emergency_compare_avg_speed.png',
]
for artifact in artifacts:
    if artifact.exists() and artifact.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
        st.image(str(artifact), caption=artifact.name, use_container_width=True)

st.subheader('Live Files')
st.write('Dashboard reads current artifacts from the repo results folders. Launching a scenario from the sidebar starts a background process using the existing runners.')

# Live SUMO view
st.markdown('---')
st.header('Live SUMO Simulation')
col_map, col_stats = st.columns([3, 1])
with col_map:
    map_box = st.empty()
with col_stats:
    st.subheader('Simulation Stats')
    stats_box = st.empty()

def render_live():
    snap = controller.get_snapshot()
    vehicles = snap.get('vehicles', {})
    if vehicles:
        xs, ys, ids, speeds = [], [], [], []
        for vid, v in vehicles.items():
            x, y = v['pos']
            xs.append(x)
            ys.append(y)
            ids.append(vid)
            speeds.append(v['speed'])

        df = pd.DataFrame({'id': ids, 'x': xs, 'y': ys, 'speed': speeds})
        # simple scatter plot
        fig = px.scatter(df, x='x', y='y', color='speed', hover_name='id', size_max=8, color_continuous_scale='Turbo')
        fig.update_yaxes(scaleanchor='x', scaleratio=1)
        fig.update_layout(height=700, margin={'l':0,'r':0,'t':30,'b':0})
        map_box.plotly_chart(fig, use_container_width=True)
    else:
        map_box.info('SUMO not running or no vehicles yet. Start the simulation from the sidebar.')

    stat_df = pd.DataFrame([
        {'metric': 'Time (s)', 'value': float(snap.get('time', 0.0))},
        {'metric': 'Step', 'value': float(snap.get('step', 0))},
        {'metric': 'Vehicle Count', 'value': float(snap.get('vehicle_count', 0))},
        {'metric': 'Avg Speed (m/s)', 'value': float(snap.get('avg_speed', 0.0))},
        {'metric': 'TraCI Step Latency (ms)', 'value': float(snap.get('traci_step_latency_ms', 0.0))},
        {'metric': 'Snapshot Latency (ms)', 'value': float(snap.get('snapshot_latency_ms', 0.0))},
        {'metric': 'TraCI min (ms)', 'value': float(snap.get('traci_step_min_ms', 0.0))},
        {'metric': 'TraCI max (ms)', 'value': float(snap.get('traci_step_max_ms', 0.0))},
        {'metric': 'TraCI p95 (ms)', 'value': float(snap.get('traci_step_p95_ms', 0.0))},
        {'metric': 'Snap min (ms)', 'value': float(snap.get('snapshot_min_ms', 0.0))},
        {'metric': 'Snap max (ms)', 'value': float(snap.get('snapshot_max_ms', 0.0))},
        {'metric': 'Snap p95 (ms)', 'value': float(snap.get('snapshot_p95_ms', 0.0))},
    ])
    stats_box.plotly_chart(px.bar(stat_df, x='metric', y='value', color='metric', title='Runtime Stats'), use_container_width=True)
    # Small live latency metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('TraCI EWMA (ms)', f"{snap.get('traci_step_latency_ms', 0.0):.2f}")
    m2.metric('TraCI p95 (ms)', f"{snap.get('traci_step_p95_ms', 0.0):.2f}")
    m3.metric('Snapshot EWMA (ms)', f"{snap.get('snapshot_latency_ms', 0.0):.2f}")
    m4.metric('Snapshot p95 (ms)', f"{snap.get('snapshot_p95_ms', 0.0):.2f}")
    
    # Auto-throttle: increase effective refresh if p95 latency is high
    try:
        p95 = float(snap.get('traci_step_p95_ms', 0.0))
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Failed to parse traci p95 latency')
        p95 = 0.0
    effective_refresh_ms = max(float(st.session_state.get('refresh_ms', 1000)), min(max(p95 * 1.5, float(st.session_state.get('refresh_ms', 1000))), 5000))
    st.session_state['effective_refresh_ms'] = int(effective_refresh_ms)


def write_controls(flags: dict):
    controls_path = REPO_ROOT / 'results' / 'controls.json'
    try:
        controls_path.parent.mkdir(parents=True, exist_ok=True)
        current = {}
        if controls_path.exists():
            current = json.loads(controls_path.read_text(encoding='utf-8'))
        current.update(flags)
        controls_path.write_text(json.dumps(current, indent=2), encoding='utf-8')
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Failed to write controls.json')


st.markdown('---')
st.header('AI Control & Modules')
col_a, col_b = st.columns(2)
with col_a:
    st.subheader('RL Controls')
    rl_enable = st.checkbox('Enable RL controllers', value=False)
    rl_algo = st.selectbox('Active RL algorithm', ['DQN', 'DDQN', 'PPO'])
    if st.button('Launch RL agent (background)'):
        _launch([sys.executable, 'integration/train_hybrid.py', '--agent', rl_algo.lower(), '--episodes', '1'], f'RL {rl_algo}')
    st.write('Current RL selection:', rl_algo)
    st.write('RL enabled:', rl_enable)
    write_controls({'rl_enabled': rl_enable, 'rl_algo': rl_algo})

with col_b:
    st.subheader('PINN Controls')
    pinn_enable = st.checkbox('Enable PINN predictions', value=True)
    horizons_text = st.text_input('Forecast horizons (sec, comma-separated)', value='0,30,60,90,120')
    bins = st.slider('Overlay resolution', 20, 80, st.session_state.get('overlay_bins', 40), 2, key='overlay_bins')
    write_controls({'pinn_enabled': pinn_enable, 'pinn_horizons': horizons_text, 'pinn_bins': bins})

st.markdown('---')
st.header('RL Telemetry (Live)')
rl_latest = _rl_latest_json()
ts = _rl_timeseries_df()
rl_left, rl_right = st.columns([2, 1])
with rl_left:
    st.subheader('Rolling Reward / Queue / Wait')
    reward_col = 'last_reward' if 'last_reward' in ts.columns else 'reward'
    if not ts.empty and {'step', reward_col, 'avg_queue', 'avg_wait'}.issubset(ts.columns):
        fig_rl = px.line(ts, x='step', y=[reward_col, 'avg_queue', 'avg_wait'])
        st.plotly_chart(fig_rl, use_container_width=True)
    else:
        st.info('Waiting for RL telemetry stream in results/rl/telemetry_timeseries.csv')

with rl_right:
    st.subheader('Algorithm Status')
    st.metric('Algorithm', str(rl_latest.get('algorithm', 'not running')))
    st.metric('Last Reward', f"{float(rl_latest.get('last_reward', 0.0)):.3f}")
    st.metric('Decision Timestamp', str(rl_latest.get('decision_time', 'n/a')))

st.subheader('Actions Per Intersection')
if rl_latest and isinstance(rl_latest.get('actions'), dict) and rl_latest.get('actions'):
    actions = rl_latest['actions']
    act_df = pd.DataFrame({'intersection': list(actions.keys()), 'action': [actions[k].get('action', 0) if isinstance(actions[k], dict) else actions[k] for k in actions.keys()]})
    st.plotly_chart(px.bar(act_df, x='intersection', y='action', color='action', title='Current RL Decisions'), use_container_width=True)
else:
    st.info('No intersection-level actions in latest RL telemetry payload yet.')

st.subheader('Current Traffic-Light Phases')
snap = controller.get_snapshot()
tls = snap.get('tls', {})
if tls:
    tls_df = pd.DataFrame([{'tls': k, 'phase': v.get('phase', 0), 'state': v.get('state', '')} for k, v in tls.items()])
    st.plotly_chart(px.bar(tls_df, x='tls', y='phase', color='phase', title='Live TLS Phases'), use_container_width=True)
else:
    st.info('No TLS data available from SUMO snapshot.')

st.subheader('Decision Timeline')
if not ts.empty and {'step', 'decision_time'}.issubset(ts.columns):
    dt = ts[ts['decision_time'].notna()]
    if not dt.empty:
        hover_cols = [c for c in [reward_col, 'algorithm', 'avg_queue', 'avg_wait'] if c in dt.columns]
        fig_dt = px.scatter(dt, x='step', y='decision_time', color='algorithm' if 'algorithm' in dt.columns else None, hover_data=hover_cols)
        st.plotly_chart(fig_dt, use_container_width=True)
    else:
        st.info('No decision events in telemetry timeline.')
else:
    st.info('No decision_time column in telemetry timeline yet.')

st.markdown('---')
st.header('Alert & Notification Center')
alerts = _build_alerts(snap, rl_latest, latest_emergency)
if alerts:
    adf = pd.DataFrame(alerts)
    st.plotly_chart(px.bar(adf, x='severity', color='severity', title='Live Alert Severity Distribution'), use_container_width=True)
    for a in alerts:
        if a['severity'] == 'high':
            st.error(a['message'])
        elif a['severity'] == 'medium':
            st.warning(a['message'])
        else:
            st.info(a['message'])
else:
    st.success('No active alerts. System operating within expected thresholds.')

st.markdown('---')
st.header('PINN Congestion Overlay (Live + Forecast)')
if pinn_enable:
    try:
        horizons = [int(x.strip()) for x in horizons_text.split(',') if x.strip()]
    except Exception:
        horizons = [0, 30, 60, 90, 120]

    current_step = int(snap.get('step', 0))
    last_step = int(st.session_state.get('pinn_overlay_last_step', -10**9))
    stride = int(st.session_state.get('overlay_stride', 4))
    if (current_step - last_step) >= stride or 'pinn_overlay_cache' not in st.session_state:
        pinn_data = _pinn_overlay(snap, horizons=horizons, bins=bins)
        st.session_state['pinn_overlay_cache'] = pinn_data
        st.session_state['pinn_overlay_last_step'] = current_step
    else:
        pinn_data = st.session_state.get('pinn_overlay_cache')
    if pinn_data is None:
        st.info('PINN overlay unavailable. Ensure SUMO is running with vehicles and pinn_models/trained_pinn.pth exists.')
    else:
        grid = pinn_data['grid']
        overlays = pinn_data['overlays']
        bdf = pinn_data['bottlenecks']
        st.session_state['latest_pinn_bottlenecks'] = bdf

        current_density = grid['density']
        first_forecast = overlays[min(1, len(overlays) - 1)]
        pred = first_forecast['predicted']
        conf = first_forecast['confidence']

        p1, p2, p3 = st.columns(3)
        with p1:
            st.plotly_chart(px.imshow(current_density, origin='lower', color_continuous_scale='Viridis', title='Current Congestion Density'), use_container_width=True)
        with p2:
            st.plotly_chart(px.imshow(pred, origin='lower', color_continuous_scale='YlOrRd', title=f"Predicted Congestion (+{first_forecast['horizon']}s)"), use_container_width=True)
        with p3:
            st.plotly_chart(px.imshow(conf, origin='lower', color_continuous_scale='Blues', title='Prediction Confidence / Intensity'), use_container_width=True)

        st.plotly_chart(_pinn_propagation_figure(pinn_data), use_container_width=True)
        if not bdf.empty:
            st.plotly_chart(px.scatter(bdf, x='x', y='y', color='risk', animation_frame='horizon', title='Future Bottleneck Hotspots (Likely ACO Reroute Zones)'), use_container_width=True)
            st.caption(f"Synchronized at SUMO time={snap.get('time', 0.0):.1f}s | vehicles={snap.get('vehicle_count', 0)}")

st.markdown('---')
st.header('ACO Routing & Path Visualization')
aco_col, em_col = st.columns(2)
with aco_col:
    st.subheader('ACO Controls')
    aco_enable = st.checkbox('Enable ACO', value=False)
    pheromone = st.slider('Pheromone weight', 0.0, 5.0, 1.0)
    heuristic = st.slider('Heuristic weight', 0.0, 5.0, 1.0)
    if st.button('Run ACO benchmark'):
        _launch([sys.executable, 'evaluation/evaluate_aco.py'], 'ACO evaluation')
    write_controls({'aco_enabled': aco_enable, 'pheromone': pheromone, 'heuristic': heuristic})

with em_col:
    st.subheader('Emergency Controls')
    if st.button('Spawn Ambulance'):
        controller.spawn_vehicle('ambulance_' + str(int(time.time())), type_id='ambulance')
    if st.button('Spawn Fire Truck'):
        controller.spawn_vehicle('fire_' + str(int(time.time())), type_id='firetruck')
    if st.button('Activate emergency mode'):
        _launch([sys.executable, 'integration/run_emergency_system.py', '--duration', '60', '--pinn', 'pinn_models/trained_pinn.pth', '--metrics-output-dir', 'results/emergency_metrics', '--run-name', 'dashboard_manual'], 'Emergency')
    write_controls({'emergency_enabled': True})

pher_df = _aco_pheromone_df()
if not pher_df.empty and {'from', 'to', 'pheromone'}.issubset(pher_df.columns):
    top_p = pher_df.sort_values('pheromone', ascending=False).head(25).copy()
    top_p['edge'] = top_p['from'] + ' -> ' + top_p['to']
    st.plotly_chart(px.bar(top_p, x='edge', y='pheromone', color='pheromone', title='Pheromone Intensity Map (Top Edges)'), use_container_width=True)

    sankey_labels = pd.unique(pd.concat([top_p['from'], top_p['to']], ignore_index=True)).tolist()
    label_to_idx = {v: i for i, v in enumerate(sankey_labels)}
    sankey = go.Figure(data=[go.Sankey(
        node={'label': sankey_labels},
        link={'source': [label_to_idx[v] for v in top_p['from']], 'target': [label_to_idx[v] for v in top_p['to']], 'value': top_p['pheromone'].tolist()},
    )])
    sankey.update_layout(title='Dynamic Path / Flow Overlay (ACO Pheromone Paths)')
    st.plotly_chart(sankey, use_container_width=True)
else:
    st.info('No pheromone file found at results/aco_results/pheromones_final.csv')

st.subheader('Rerouted Vehicle Highlighting & Congestion-Aware Routing Map')
vehicles = snap.get('vehicles', {})
if vehicles:
    prev_lanes = st.session_state.get('vehicle_last_lanes', {})
    rows = []
    for vid, info in vehicles.items():
        lane = info.get('lane', '')
        rows.append({'id': vid, 'x': info['pos'][0], 'y': info['pos'][1], 'speed': info.get('speed', 0.0), 'rerouted': prev_lanes.get(vid) not in (None, lane)})
        prev_lanes[vid] = lane
    st.session_state['vehicle_last_lanes'] = prev_lanes
    vdf = pd.DataFrame(rows)
    vdf['status'] = np.where(vdf['rerouted'], 'rerouted', 'normal')
    fig_routes = px.scatter(vdf, x='x', y='y', color='status', size='speed', hover_name='id', title='Dynamic Vehicle Paths (Reroutes Highlighted)')
    bdf = st.session_state.get('latest_pinn_bottlenecks', pd.DataFrame())
    if isinstance(bdf, pd.DataFrame) and not bdf.empty:
        b0 = bdf[bdf['horizon'] == bdf['horizon'].min()]
        fig_routes.add_trace(go.Scatter(x=b0['x'], y=b0['y'], mode='markers', name='Likely Reroute Zones', marker={'symbol': 'x', 'size': 8, 'color': 'red'}))
    fig_routes.update_yaxes(scaleanchor='x', scaleratio=1)
    st.plotly_chart(fig_routes, use_container_width=True)
else:
    st.info('No live vehicles available for reroute/path overlay.')

st.markdown('---')
st.header('Signal Control & Incidents')
tls_ids = list(controller.get_tls_ids())
sel_tls = st.selectbox('Select traffic light', options=([''] + tls_ids))
if sel_tls:
    phase = st.number_input('Phase (int)', min_value=0, max_value=20, value=0)
    if st.button('Set TLS phase'):
        ok = controller.set_tls_phase(sel_tls, phase)
        st.success(f'Set phase {phase} on {sel_tls}: {ok}')

edge_id = st.text_input('Edge ID for closure / reopen')
if st.button('Close edge') and edge_id:
    controller.close_edge(edge_id)
    st.warning(f'Edge {edge_id} closed (speed throttled).')
if st.button('Open edge') and edge_id:
    controller.open_edge(edge_id)
    st.success(f'Edge {edge_id} reopened.')

st.markdown('---')
st.header('Final Demo')
if st.button('Run Full AI Smart City Demo'):
    # Orchestrate core services from one click
    if not _port_open('127.0.0.1', 8765):
        _launch([sys.executable, 'integration/rl_telemetry_server.py', '--host', '127.0.0.1', '--port', '8765'], 'RL telemetry server')
    if not controller.is_running():
        controller.start()
    for i in range(50):
        controller.spawn_vehicle(f'car_demo_{i}')
    time.sleep(1.0)
    controller.spawn_emergency_vehicle('ambulance')
    _launch([sys.executable, 'integration/run_emergency_system.py', '--duration', '120', '--pinn', 'pinn_models/trained_pinn.pth', '--metrics-output-dir', 'results/emergency_metrics', '--run-name', 'full_demo'], 'Emergency orchestrator')
    st.success('Full AI demo started: telemetry + SUMO + congestion + emergency orchestration.')

st.markdown('---')
st.header('Export Center')
report_payload = {
    'generated_at': datetime.utcnow().isoformat() + 'Z',
    'snapshot': snap,
    'rl_latest': rl_latest,
    'emergency_latest': latest_emergency,
    'alerts': alerts,
    'scenario': scenario,
    'algorithms': algorithm,
}
st.download_button(
    'Download Session Report (JSON)',
    data=json.dumps(report_payload, indent=2),
    file_name='smart_city_session_report.json',
    mime='application/json',
)

if not ts.empty:
    st.download_button(
        'Download RL Telemetry Timeline (CSV)',
        data=ts.to_csv(index=False),
        file_name='rl_telemetry_timeseries.csv',
        mime='text/csv',
    )

bdf_exp = st.session_state.get('latest_pinn_bottlenecks', pd.DataFrame())
if isinstance(bdf_exp, pd.DataFrame) and not bdf_exp.empty:
    st.download_button(
        'Download PINN Bottleneck Forecasts (CSV)',
        data=bdf_exp.to_csv(index=False),
        file_name='pinn_bottlenecks.csv',
        mime='text/csv',
    )

# Auto-refresh loop (polling)
if controller.is_running():
    render_live()
    # respect computed effective refresh (auto-throttled by p95 latency) when available
    eff = int(st.session_state.get('effective_refresh_ms', st.session_state.get('refresh_ms', 1000)))
    try:
        st.markdown(f"**Effective refresh:** {eff} ms")
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Failed to render effective refresh stat')
    time.sleep(max(float(eff) / 1000.0, 0.25))
    st.rerun()
else:
    render_live()
