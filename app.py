"""
Mathematical Concepts Visual Explorer
=====================================
Portfolio project by Yogesh Kumar Singh
PhD Scholar, IIT Kharagpur (CSE)
M.Sc. Mathematics, BHU | M.Tech CSE, IIT Kharagpur
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import scipy.stats as stats
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Math Visual Explorer",
    page_icon="∑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Sidebar navigation */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #7c3aed44;
    }
    /* Header banner */
    .page-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #7c3aed55;
        border-left: 4px solid #7c3aed;
        border-radius: 8px;
        padding: 18px 24px;
        margin-bottom: 24px;
    }
    .page-header h1 {
        margin: 0 0 4px 0;
        font-size: 1.9rem;
        background: linear-gradient(90deg, #7c3aed, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .page-header p {
        margin: 0;
        color: #94a3b8;
        font-size: 0.95rem;
    }
    /* Metric cards */
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #7c3aed44;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-card .label {
        color: #94a3b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }
    .metric-card .value {
        color: #7c3aed;
        font-size: 1.5rem;
        font-weight: 700;
    }
    /* Info boxes */
    .math-insight {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #06b6d444;
        border-left: 3px solid #06b6d4;
        border-radius: 8px;
        padding: 16px 20px;
        margin-top: 16px;
    }
    /* Sidebar profile */
    .sidebar-profile {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #7c3aed44;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "primary": "#7c3aed",
    "secondary": "#06b6d4",
    "accent": "#f59e0b",
    "danger": "#ef4444",
    "success": "#10b981",
    "muted": "#94a3b8",
}


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-profile">
            <div style="font-size:2.5rem; margin-bottom:8px;">∑</div>
            <div style="font-weight:700; font-size:1rem; color:#e2e8f0;">Yogesh Kumar Singh</div>
            <div style="color:#7c3aed; font-size:0.8rem; margin:4px 0;">PhD Scholar · IIT Kharagpur</div>
            <div style="color:#94a3b8; font-size:0.75rem;">CSE Dept.</div>
            <hr style="border-color:#7c3aed33; margin:10px 0;">
            <div style="color:#94a3b8; font-size:0.75rem;">M.Sc. Mathematics · BHU</div>
            <div style="color:#94a3b8; font-size:0.75rem;">M.Tech CSE · IIT Kharagpur</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigate")
    page = st.radio(
        label="",
        options=[
            "〜 Fourier Transform",
            "λ Eigenvalues & Eigenvectors",
            "∇ Gradient Descent",
            "⌗ Probability Distributions",
            "≈ Regression & Correlation",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        "<div style='color:#94a3b8; font-size:0.75rem; text-align:center;'>"
        "Mathematical Concepts Visual Explorer<br>"
        "<span style='color:#7c3aed;'>Portfolio · 2024</span></div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — FOURIER TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════
def page_fourier():
    st.markdown(
        """
        <div class="page-header">
            <h1>〜 Fourier Series Decomposition</h1>
            <p>Every periodic signal is a superposition of pure sinusoids — Fourier's revolutionary insight.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### Controls")
        waveform = st.selectbox(
            "Waveform Type",
            ["Square Wave", "Sawtooth Wave", "Triangle Wave"],
        )
        n_harmonics = st.slider("Number of Harmonics (N)", 1, 20, 5)
        frequency = st.slider("Fundamental Frequency (Hz)", 1, 5, 1)

    # ── Compute ──────────────────────────────────────────────────────────
    T = 1.0 / frequency  # period
    t = np.linspace(0, 2 * T, 2000)
    omega = 2 * np.pi * frequency

    # Target (ideal) waveforms
    if waveform == "Square Wave":
        target = np.sign(np.sin(omega * t))
        target[target == 0] = 1.0
    elif waveform == "Sawtooth Wave":
        target = 2 * (t * frequency - np.floor(t * frequency + 0.5))
    else:  # Triangle
        target = 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1

    # Fourier approximation
    approx = np.zeros_like(t)
    amplitudes = []
    freqs_hz = []

    if waveform == "Square Wave":
        # f(t) = (4/π) Σ sin((2k-1)ωt) / (2k-1)  for k=1..N
        for k in range(1, n_harmonics + 1):
            harmonic = 2 * k - 1
            amp = 4 / (np.pi * harmonic)
            approx += amp * np.sin(harmonic * omega * t)
            amplitudes.append(amp)
            freqs_hz.append(harmonic * frequency)

    elif waveform == "Sawtooth Wave":
        # f(t) = (2/π) Σ (-1)^(k+1) sin(kωt) / k
        for k in range(1, n_harmonics + 1):
            amp = 2 / (np.pi * k) * ((-1) ** (k + 1))
            approx += amp * np.sin(k * omega * t)
            amplitudes.append(abs(amp))
            freqs_hz.append(k * frequency)

    else:  # Triangle
        # f(t) = (8/π²) Σ (-1)^(k-1) sin((2k-1)ωt) / (2k-1)²
        for k in range(1, n_harmonics + 1):
            harmonic = 2 * k - 1
            amp = 8 / (np.pi**2 * harmonic**2) * ((-1) ** (k - 1))
            approx += amp * np.sin(harmonic * omega * t)
            amplitudes.append(abs(amp))
            freqs_hz.append(harmonic * frequency)

    # ── Charts ───────────────────────────────────────────────────────────
    with col_main:
        # Chart 1: Time domain
        fig_time = go.Figure()
        fig_time.add_trace(
            go.Scatter(
                x=t,
                y=target,
                name="Target Signal",
                line=dict(color=COLORS["danger"], width=2, dash="dash"),
                opacity=0.7,
            )
        )
        fig_time.add_trace(
            go.Scatter(
                x=t,
                y=approx,
                name=f"Fourier Approx (N={n_harmonics})",
                line=dict(color=COLORS["secondary"], width=2.5),
            )
        )
        fig_time.update_layout(
            template=PLOTLY_TEMPLATE,
            title=dict(
                text=f"<b>Time Domain</b> — {waveform} with {n_harmonics} harmonic(s)",
                font=dict(size=14),
            ),
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            height=380,
            legend=dict(orientation="h", y=-0.2),
            plot_bgcolor="#0f0f1a",
            paper_bgcolor="#1a1a2e",
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Chart 2: Frequency spectrum
        fig_freq = go.Figure()
        for i, (f_val, amp) in enumerate(zip(freqs_hz, amplitudes)):
            fig_freq.add_trace(
                go.Scatter(
                    x=[f_val, f_val],
                    y=[0, amp],
                    mode="lines",
                    line=dict(color=COLORS["primary"], width=3),
                    showlegend=False,
                )
            )
            fig_freq.add_trace(
                go.Scatter(
                    x=[f_val],
                    y=[amp],
                    mode="markers",
                    marker=dict(
                        color=COLORS["accent"],
                        size=10,
                        symbol="circle",
                        line=dict(color=COLORS["primary"], width=2),
                    ),
                    showlegend=(i == 0),
                    name="Harmonic Amplitude",
                )
            )
        fig_freq.update_layout(
            template=PLOTLY_TEMPLATE,
            title=dict(
                text="<b>Frequency Spectrum</b> — Harmonic Amplitudes",
                font=dict(size=14),
            ),
            xaxis_title="Frequency (Hz)",
            yaxis_title="|Amplitude|",
            height=320,
            plot_bgcolor="#0f0f1a",
            paper_bgcolor="#1a1a2e",
        )
        st.plotly_chart(fig_freq, use_container_width=True)

    # ── Math Insight ─────────────────────────────────────────────────────
    energy_ratio = (
        np.sum(approx**2) / np.sum(target**2) * 100
        if np.sum(target**2) > 0
        else 0
    )
    st.markdown(
        f"""
        <div class="math-insight">
            <b style="color:#06b6d4;">📐 Mathematical Insight</b><br><br>
            The <b>Fourier Series</b> represents a T-periodic function f(t) as:<br><br>
            &nbsp;&nbsp;&nbsp;&nbsp;<i>f(t) = a₀/2 + Σₙ [ aₙ cos(nωt) + bₙ sin(nωt) ]</i><br><br>
            With <b>N = {n_harmonics}</b> harmonic(s), the approximation captures
            <b style="color:#f59e0b;">{energy_ratio:.1f}%</b> of the signal's energy.
            {"<br>⚡ <b>Gibbs Phenomenon</b>: Notice the ~9% overshoot near discontinuities — this persists regardless of N and is a fundamental property of Fourier series." if waveform in ["Square Wave", "Sawtooth Wave"] and n_harmonics >= 5 else ""}
            As N → ∞, the series converges in the L² sense (mean-square convergence),
            though pointwise convergence at discontinuities requires additional conditions (Dirichlet's theorem).
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — EIGENVALUES & EIGENVECTORS
# ═══════════════════════════════════════════════════════════════════════════
def page_eigenvalues():
    st.markdown(
        """
        <div class="page-header">
            <h1>λ Eigenvalues &amp; Eigenvectors</h1>
            <p>Eigenvectors are the "natural axes" of a linear transformation — they stretch but never rotate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### Matrix A = [[a, b], [c, d]]")
        a = st.number_input("a (row 1, col 1)", value=2.0, step=0.5, format="%.2f")
        b = st.number_input("b (row 1, col 2)", value=1.0, step=0.5, format="%.2f")
        c = st.number_input("c (row 2, col 1)", value=1.0, step=0.5, format="%.2f")
        d = st.number_input("d (row 2, col 2)", value=2.0, step=0.5, format="%.2f")

        st.markdown(
            f"""
            <div class="metric-card" style="margin-top:16px;">
                <div class="label">Matrix</div>
                <div style="color:#e2e8f0; font-family:monospace; font-size:1.1rem;">
                    [{a:.1f}, {b:.1f}]<br>[{c:.1f}, {d:.1f}]
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    A = np.array([[a, b], [c, d]])
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # ── Visualization data ───────────────────────────────────────────────
    theta = np.linspace(0, 2 * np.pi, 300)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Transform the circle
    circle_pts = np.vstack([circle_x, circle_y])
    transformed_pts = A @ circle_pts

    # Grid arrows (unit vectors on a coarse grid)
    grid_range = np.arange(-2, 3, 1.0)
    gx, gy = np.meshgrid(grid_range, grid_range)
    gx_flat = gx.flatten()
    gy_flat = gy.flatten()

    with col_main:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Before Transformation", "After Transformation"],
        )

        # ── Before ───────────────────────────────────────────────────────
        # Unit circle
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                name="Unit Circle",
                line=dict(color=COLORS["secondary"], width=2),
                showlegend=True,
            ),
            row=1, col=1,
        )

        # Eigenvectors (before — just directions)
        ev_colors = [COLORS["accent"], COLORS["success"]]
        for i in range(2):
            ev = eigenvectors[:, i].real
            ev = ev / (np.linalg.norm(ev) + 1e-12)
            lam = eigenvalues[i].real
            fig.add_trace(
                go.Scatter(
                    x=[-ev[0] * 1.5, ev[0] * 1.5],
                    y=[-ev[1] * 1.5, ev[1] * 1.5],
                    mode="lines+markers",
                    name=f"v{i+1} (λ={lam:.2f})",
                    line=dict(color=ev_colors[i], width=2.5, dash="dash"),
                    marker=dict(size=8, symbol="arrow-up", angleref="previous"),
                    showlegend=True,
                ),
                row=1, col=1,
            )

        # ── After ────────────────────────────────────────────────────────
        # Transformed ellipse
        fig.add_trace(
            go.Scatter(
                x=transformed_pts[0],
                y=transformed_pts[1],
                mode="lines",
                name="Transformed Circle",
                line=dict(color=COLORS["primary"], width=2),
                showlegend=True,
            ),
            row=1, col=2,
        )

        # Transformed eigenvectors (scaled by eigenvalue)
        for i in range(2):
            ev = eigenvectors[:, i].real
            ev_norm = ev / (np.linalg.norm(ev) + 1e-12)
            lam = eigenvalues[i].real
            scale = abs(lam) * 1.2
            fig.add_trace(
                go.Scatter(
                    x=[-ev_norm[0] * scale, ev_norm[0] * scale],
                    y=[-ev_norm[1] * scale, ev_norm[1] * scale],
                    mode="lines",
                    name=f"λ{i+1}·v{i+1}",
                    line=dict(color=ev_colors[i], width=2.5, dash="dash"),
                    showlegend=True,
                ),
                row=1, col=2,
            )

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            height=420,
            plot_bgcolor="#0f0f1a",
            paper_bgcolor="#1a1a2e",
            legend=dict(orientation="h", y=-0.15, font=dict(size=11)),
        )
        # Equal aspect ratio
        for col_idx in [1, 2]:
            ax_range = max(3.5, np.max(np.abs(transformed_pts)) * 1.2)
            fig.update_xaxes(
                scaleanchor=f"y{'' if col_idx==1 else col_idx}",
                scaleratio=1,
                range=[-ax_range, ax_range],
                row=1, col=col_idx,
            )
            fig.update_yaxes(range=[-ax_range, ax_range], row=1, col=col_idx)

        st.plotly_chart(fig, use_container_width=True)

    # ── Eigenvalue cards ─────────────────────────────────────────────────
    st.markdown("#### Eigendecomposition Results")
    metric_cols = st.columns(4)

    for i in range(2):
        lam = eigenvalues[i]
        ev = eigenvectors[:, i]
        lam_str = (
            f"{lam.real:.4f}"
            if abs(lam.imag) < 1e-10
            else f"{lam.real:.3f}+{lam.imag:.3f}i"
        )
        ev_str = (
            f"[{ev[0].real:.3f}, {ev[1].real:.3f}]"
            if abs(ev[0].imag) < 1e-10
            else f"[{ev[0].real:.2f}+{ev[0].imag:.2f}i, ...]"
        )

        with metric_cols[2 * i]:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="label">Eigenvalue λ{i+1}</div>
                    <div class="value">{lam_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with metric_cols[2 * i + 1]:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="label">Eigenvector v{i+1}</div>
                    <div class="value" style="font-size:1.1rem;">{ev_str}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    det_A = np.linalg.det(A)
    trace_A = np.trace(A)
    st.markdown(
        f"""
        <div class="math-insight" style="margin-top:20px;">
            <b style="color:#06b6d4;">📐 Mathematical Insight</b><br><br>
            For matrix A, <b>Av = λv</b> — eigenvectors maintain their direction under the transformation.
            <br><br>
            Key identities: &nbsp;&nbsp;
            <b>det(A)</b> = λ₁ · λ₂ = <span style="color:#f59e0b;">{det_A:.4f}</span> &nbsp;|&nbsp;
            <b>tr(A)</b> = λ₁ + λ₂ = <span style="color:#f59e0b;">{trace_A:.4f}</span>
            <br><br>
            Eigenvalues of λ₁ = <b>{eigenvalues[0].real:.3f}</b>
            {"(real)" if abs(eigenvalues[0].imag) < 1e-10 else "(complex — rotation present)"} and
            λ₂ = <b>{eigenvalues[1].real:.3f}</b>
            {"(real)" if abs(eigenvalues[1].imag) < 1e-10 else "(complex)"}.
            {"The matrix is <b>symmetric</b> (A=Aᵀ), so eigenvectors are orthogonal — guaranteed by the Spectral Theorem." if abs(b-c) < 1e-10 else "The matrix is <b>asymmetric</b>; eigenvectors may not be orthogonal."}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — GRADIENT DESCENT
# ═══════════════════════════════════════════════════════════════════════════
def _loss_and_grad(fn_name, x, y):
    """Return (loss, grad_x, grad_y) for the given function."""
    if fn_name == "Bowl (x²+y²)":
        loss = x**2 + y**2
        gx = 2 * x
        gy = 2 * y
    elif fn_name == "Rosenbrock":
        # f(x,y) = (1-x)² + 100(y-x²)²
        loss = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        gx = -2 * (1 - x) + 100 * 2 * (y - x**2) * (-2 * x)
        gy = 100 * 2 * (y - x**2)
    else:  # Himmelblau
        # f(x,y) = (x²+y-11)² + (x+y²-7)²
        loss = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
        gx = 2 * (x**2 + y - 11) * 2 * x + 2 * (x + y**2 - 7)
        gy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * 2 * y
    return loss, gx, gy


def page_gradient_descent():
    st.markdown(
        """
        <div class="page-header">
            <h1>∇ Gradient Descent on a Loss Surface</h1>
            <p>Follow the negative gradient downhill — the workhorse of modern machine learning optimization.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### Controls")
        fn_name = st.selectbox(
            "Loss Function",
            ["Bowl (x²+y²)", "Rosenbrock", "Himmelblau"],
        )
        lr = st.slider("Learning Rate (α)", 0.001, 0.5, 0.05, step=0.001, format="%.3f")
        max_iter = st.slider("Max Iterations", 10, 500, 100)
        x0 = st.slider("Start X", -3.0, 3.0, -2.0, step=0.1)
        y0 = st.slider("Start Y", -3.0, 3.0, 2.0, step=0.1)

    # ── Gradient descent ─────────────────────────────────────────────────
    path_x = [x0]
    path_y = [y0]
    path_loss = []
    cx, cy = x0, y0

    for _ in range(max_iter):
        loss_val, gx, gy = _loss_and_grad(fn_name, cx, cy)
        path_loss.append(loss_val)
        if np.isnan(loss_val) or loss_val > 1e8:
            break
        cx = cx - lr * gx
        cy = cy - lr * gy
        path_x.append(cx)
        path_y.append(cy)
        if loss_val < 1e-8:
            break

    final_loss, _, _ = _loss_and_grad(fn_name, cx, cy)
    path_loss.append(final_loss)
    actual_iter = len(path_loss) - 1

    # ── Surface data ─────────────────────────────────────────────────────
    x_range = np.linspace(-3.5, 3.5, 120)
    y_range = np.linspace(-3.5, 3.5, 120)
    XX, YY = np.meshgrid(x_range, y_range)
    ZZ, _, _ = _loss_and_grad(fn_name, XX, YY)
    ZZ = np.clip(ZZ, 0, 500)

    # Path z-values
    path_z = [_loss_and_grad(fn_name, px, py)[0] for px, py in zip(path_x, path_y)]
    path_z = np.clip(path_z, 0, 500)

    with col_main:
        chart_col1, chart_col2 = st.columns(2)

        # Chart 1: 3D surface
        with chart_col1:
            fig3d = go.Figure()
            fig3d.add_trace(
                go.Surface(
                    x=x_range,
                    y=y_range,
                    z=ZZ,
                    colorscale="Viridis",
                    opacity=0.85,
                    showscale=False,
                    name="Loss Surface",
                )
            )
            fig3d.add_trace(
                go.Scatter3d(
                    x=path_x,
                    y=path_y,
                    z=list(path_z),
                    mode="lines+markers",
                    line=dict(color=COLORS["danger"], width=5),
                    marker=dict(size=3, color=COLORS["accent"]),
                    name="Descent Path",
                )
            )
            fig3d.add_trace(
                go.Scatter3d(
                    x=[path_x[0]],
                    y=[path_y[0]],
                    z=[path_z[0]],
                    mode="markers",
                    marker=dict(size=8, color=COLORS["success"], symbol="circle"),
                    name="Start",
                )
            )
            fig3d.add_trace(
                go.Scatter3d(
                    x=[path_x[-1]],
                    y=[path_y[-1]],
                    z=[path_z[-1]],
                    mode="markers",
                    marker=dict(size=8, color=COLORS["danger"], symbol="diamond"),
                    name="End",
                )
            )
            fig3d.update_layout(
                template=PLOTLY_TEMPLATE,
                title="<b>3D Loss Surface</b>",
                height=420,
                paper_bgcolor="#1a1a2e",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="Loss",
                    bgcolor="#0f0f1a",
                ),
                legend=dict(font=dict(size=10), x=0, y=1),
            )
            st.plotly_chart(fig3d, use_container_width=True)

        # Chart 2: 2D contour
        with chart_col2:
            fig2d = go.Figure()
            fig2d.add_trace(
                go.Contour(
                    x=x_range,
                    y=y_range,
                    z=ZZ,
                    colorscale="Viridis",
                    showscale=False,
                    ncontours=30,
                    opacity=0.9,
                )
            )
            fig2d.add_trace(
                go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode="lines+markers",
                    line=dict(color=COLORS["danger"], width=2.5),
                    marker=dict(size=4, color=COLORS["accent"]),
                    name="Descent Path",
                )
            )
            fig2d.add_trace(
                go.Scatter(
                    x=[path_x[0]], y=[path_y[0]],
                    mode="markers",
                    marker=dict(size=10, color=COLORS["success"], symbol="star"),
                    name="Start",
                )
            )
            fig2d.add_trace(
                go.Scatter(
                    x=[path_x[-1]], y=[path_y[-1]],
                    mode="markers",
                    marker=dict(size=10, color=COLORS["danger"], symbol="diamond"),
                    name="End",
                )
            )
            fig2d.update_layout(
                template=PLOTLY_TEMPLATE,
                title="<b>2D Contour Map</b>",
                xaxis_title="x",
                yaxis_title="y",
                height=420,
                plot_bgcolor="#0f0f1a",
                paper_bgcolor="#1a1a2e",
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig2d, use_container_width=True)

        # Chart 3: Convergence curve
        fig_conv = go.Figure()
        iter_range = list(range(len(path_loss)))
        fig_conv.add_trace(
            go.Scatter(
                x=iter_range,
                y=path_loss,
                mode="lines",
                line=dict(color=COLORS["primary"], width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba(124,58,237,0.15)",
                name="Loss",
            )
        )
        fig_conv.update_layout(
            template=PLOTLY_TEMPLATE,
            title="<b>Convergence Curve</b> — Loss vs. Iteration",
            xaxis_title="Iteration",
            yaxis_title="Loss Value",
            height=300,
            plot_bgcolor="#0f0f1a",
            paper_bgcolor="#1a1a2e",
        )
        st.plotly_chart(fig_conv, use_container_width=True)

    # ── Stats ─────────────────────────────────────────────────────────────
    s1, s2, s3, s4 = st.columns(4)
    def stat_card(col, label, val):
        col.markdown(
            f"""<div class="metric-card"><div class="label">{label}</div>
            <div class="value">{val}</div></div>""",
            unsafe_allow_html=True,
        )

    stat_card(s1, "Final Loss", f"{final_loss:.6f}")
    stat_card(s2, "Iterations", str(actual_iter))
    stat_card(s3, "Final X", f"{cx:.4f}")
    stat_card(s4, "Final Y", f"{cy:.4f}")

    insights = {
        "Bowl (x²+y²)": "The bowl function is <b>strongly convex</b> — gradient descent always converges to the global minimum at (0,0). Convergence rate is geometric: ‖xₖ-x*‖ ≤ (1-2αm)ᵏ‖x₀-x*‖.",
        "Rosenbrock": "The <b>Rosenbrock function</b> (banana valley) f = (1-x)² + 100(y-x²)² is the classic hard optimization benchmark. The global minimum is at (1,1). High curvature ratio makes vanilla gradient descent slow — motivating Adam, momentum methods.",
        "Himmelblau": "<b>Himmelblau's function</b> has <i>four</i> global minima at ≈(3,2),(−2.8,3.1),(−3.8,−3.3),(3.6,−1.8). It demonstrates why non-convex optimization is hard — which minimum you find depends entirely on initialization.",
    }
    st.markdown(
        f"""
        <div class="math-insight">
            <b style="color:#06b6d4;">📐 Mathematical Insight</b><br><br>
            Update rule: <b>x_{'{k+1}'} = x_k − α ∇f(x_k)</b> where α = {lr} (learning rate).<br><br>
            {insights[fn_name]}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — PROBABILITY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════
def page_distributions():
    st.markdown(
        """
        <div class="page-header">
            <h1>⌗ Probability Distribution Explorer</h1>
            <p>The mathematical language of uncertainty — from Gaussian bells to heavy-tailed Gammas.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ctrl, col_main = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### Select Distribution")
        dist_name = st.selectbox(
            "Distribution",
            ["Normal", "Poisson", "Binomial", "Exponential", "Beta", "Gamma"],
        )

        st.markdown("#### Parameters")

        if dist_name == "Normal":
            mu = st.slider("Mean (μ)", -5.0, 5.0, 0.0, 0.1)
            sigma = st.slider("Std Dev (σ)", 0.1, 5.0, 1.0, 0.1)
            dist = stats.norm(loc=mu, scale=sigma)
            x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 500)
            label = f"N(μ={mu:.1f}, σ={sigma:.1f})"
            is_discrete = False
            definition = "f(x) = (1/√(2πσ²)) · exp(−(x−μ)²/(2σ²))"

        elif dist_name == "Poisson":
            lam = st.slider("Rate (λ)", 0.5, 20.0, 5.0, 0.5)
            dist = stats.poisson(mu=lam)
            x = np.arange(0, int(lam + 4 * np.sqrt(lam)) + 1)
            label = f"Poisson(λ={lam:.1f})"
            is_discrete = True
            definition = "P(X=k) = e^(−λ) · λᵏ / k!   (k = 0,1,2,...)"

        elif dist_name == "Binomial":
            n_trials = st.slider("Trials (n)", 5, 100, 20)
            p_prob = st.slider("Probability (p)", 0.01, 0.99, 0.5, 0.01)
            dist = stats.binom(n=n_trials, p=p_prob)
            x = np.arange(0, n_trials + 1)
            label = f"Bin(n={n_trials}, p={p_prob:.2f})"
            is_discrete = True
            definition = "P(X=k) = C(n,k) · pᵏ · (1−p)^(n−k)"

        elif dist_name == "Exponential":
            lam_e = st.slider("Rate (λ)", 0.1, 5.0, 1.0, 0.1)
            dist = stats.expon(scale=1.0 / lam_e)
            x = np.linspace(0, 6.0 / lam_e, 500)
            label = f"Exp(λ={lam_e:.1f})"
            is_discrete = False
            definition = "f(x) = λ · e^(−λx)   (x ≥ 0)"

        elif dist_name == "Beta":
            alpha_b = st.slider("Shape α", 0.5, 10.0, 2.0, 0.5)
            beta_b = st.slider("Shape β", 0.5, 10.0, 5.0, 0.5)
            dist = stats.beta(a=alpha_b, b=beta_b)
            x = np.linspace(0.001, 0.999, 500)
            label = f"Beta(α={alpha_b:.1f}, β={beta_b:.1f})"
            is_discrete = False
            definition = "f(x) = xᵅ⁻¹(1−x)^(β−1) / B(α,β)   (x ∈ [0,1])"

        else:  # Gamma
            k_shape = st.slider("Shape (k)", 0.5, 10.0, 2.0, 0.5)
            theta_scale = st.slider("Scale (θ)", 0.1, 5.0, 1.0, 0.1)
            dist = stats.gamma(a=k_shape, scale=theta_scale)
            x = np.linspace(0.001, k_shape * theta_scale * 5, 500)
            label = f"Gamma(k={k_shape:.1f}, θ={theta_scale:.1f})"
            is_discrete = False
            definition = "f(x) = xᵏ⁻¹ e^(−x/θ) / (θᵏ Γ(k))   (x ≥ 0)"

    # ── Compute pdf/pmf and cdf ───────────────────────────────────────────
    if is_discrete:
        pmf_vals = dist.pmf(x)
        cdf_vals = dist.cdf(x)
    else:
        pdf_vals = dist.pdf(x)
        cdf_vals = dist.cdf(x)

    # Moments
    mean_val, var_val = dist.mean(), dist.var()
    skew_val = dist.stats(moments="s")
    kurt_val = dist.stats(moments="k")

    with col_main:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            fig_pdf = go.Figure()
            if is_discrete:
                fig_pdf.add_trace(
                    go.Bar(
                        x=x, y=pmf_vals,
                        name="PMF",
                        marker_color=COLORS["primary"],
                        marker_line_color=COLORS["secondary"],
                        marker_line_width=1.5,
                        opacity=0.85,
                    )
                )
            else:
                fig_pdf.add_trace(
                    go.Scatter(
                        x=x, y=pdf_vals,
                        mode="lines",
                        fill="tozeroy",
                        fillcolor=f"rgba(124,58,237,0.25)",
                        line=dict(color=COLORS["primary"], width=2.5),
                        name="PDF",
                    )
                )
                # Mark mean
                mean_y = dist.pdf(mean_val) if not is_discrete else dist.pmf(int(mean_val))
                fig_pdf.add_vline(
                    x=mean_val,
                    line_dash="dash",
                    line_color=COLORS["accent"],
                    annotation_text=f"μ={mean_val:.2f}",
                    annotation_font_color=COLORS["accent"],
                )

            fig_pdf.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"<b>{'PMF' if is_discrete else 'PDF'}</b> — {label}",
                xaxis_title="x",
                yaxis_title="Probability",
                height=380,
                plot_bgcolor="#0f0f1a",
                paper_bgcolor="#1a1a2e",
            )
            st.plotly_chart(fig_pdf, use_container_width=True)

        with chart_col2:
            fig_cdf = go.Figure()
            fig_cdf.add_trace(
                go.Scatter(
                    x=x, y=cdf_vals,
                    mode="lines",
                    fill="tozeroy",
                    fillcolor=f"rgba(6,182,212,0.2)",
                    line=dict(color=COLORS["secondary"], width=2.5),
                    name="CDF",
                )
            )
            fig_cdf.update_layout(
                template=PLOTLY_TEMPLATE,
                title=f"<b>CDF</b> — {label}",
                xaxis_title="x",
                yaxis_title="F(x) = P(X ≤ x)",
                height=380,
                plot_bgcolor="#0f0f1a",
                paper_bgcolor="#1a1a2e",
            )
            st.plotly_chart(fig_cdf, use_container_width=True)

    # ── Moments ───────────────────────────────────────────────────────────
    st.markdown("#### Statistical Moments")
    m1, m2, m3, m4 = st.columns(4)
    for col, lbl, val in [
        (m1, "Mean (μ)", f"{mean_val:.4f}"),
        (m2, "Variance (σ²)", f"{var_val:.4f}"),
        (m3, "Skewness", f"{float(skew_val):.4f}"),
        (m4, "Excess Kurtosis", f"{float(kurt_val):.4f}"),
    ]:
        col.markdown(
            f"""<div class="metric-card"><div class="label">{lbl}</div>
            <div class="value">{val}</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="math-insight" style="margin-top:20px;">
            <b style="color:#06b6d4;">📐 Mathematical Definition</b><br><br>
            <b>{dist_name} Distribution:</b> &nbsp;<code>{definition}</code><br><br>
            <b>Skewness</b> = E[(X−μ)³]/σ³ measures asymmetry.
            <b>Excess Kurtosis</b> = E[(X−μ)⁴]/σ⁴ − 3 measures tail heaviness relative to Normal.
            {"A kurtosis of 0 = mesokurtic (Normal-like). Positive = leptokurtic (heavy tails). Negative = platykurtic (light tails)." if abs(float(kurt_val)) < 20 else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 5 — REGRESSION & CORRELATION
# ═══════════════════════════════════════════════════════════════════════════


def page_regression():
    st.title("≈ Regression & Correlation")
    st.info("🔧 OLS regression explorer coming in next commit")


# --- Router ---
if page == "∼ Fourier Transform":
    page_fourier()
elif page == "λ Eigenvalues & Eigenvectors":
    page_eigenvalues()
elif page == "∇ Gradient Descent":
    page_gradient_descent()
elif page == "⌗ Probability Distributions":
    page_distributions()
elif page == "≈ Regression & Correlation":
    page_regression()
