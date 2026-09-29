"""
Visual UI Components & Professional Plotly Charts
Constructs consistent dark-navy cybersecurity & AI research styled charts and widgets.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


NAVY_LAYOUT = dict(
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0B132B",
    font=dict(color="#F8FAFC", family="Plus Jakarta Sans, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1", size=12)
    ),
    yaxis=dict(
        gridcolor="#1E293B",
        zerolinecolor="#334155",
        tickfont=dict(color="#94A3B8"),
        title_font=dict(color="#CBD5E1", size=12)
    ),
    legend=dict(
        bgcolor="rgba(15, 23, 42, 0.8)",
        bordercolor="#1E293B",
        borderwidth=1,
        font=dict(color="#CBD5E1")
    )
)


def create_feature_gauge_bars(scores: dict) -> str:
    """
    Renders custom HTML gauges/progress bars for ad feature scores.
    """
    labels = [
        ("Urgency", scores.get("urgency", 0.5)),
        ("Emotional Intensity", scores.get("emotional_intensity", 0.5)),
        ("Discount Emphasis", scores.get("discount_emphasis", 0.5)),
        ("Call-To-Action Strength", scores.get("cta_strength", 0.5)),
        ("Persuasive Intensity", scores.get("persuasive_intensity", 0.5)),
        ("Visual Intensity", scores.get("visual_intensity", 0.5)),
    ]
    
    html = '<div style="background:#0F172A; border:1px solid #1E293B; border-radius:12px; padding:20px;">'
    for label, val in labels:
        pct = int(round(val * 100))
        # Gradient color based on value
        color = "#00F0FF" if pct > 70 else ("#38BDF8" if pct > 40 else "#6366F1")
        blocks_fill = "█" * int(pct // 10)
        blocks_empty = "░" * (10 - int(pct // 10))
        
        html += f"""
        <div style="margin-bottom: 14px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-size:0.85rem;">
                <span style="color:#CBD5E1; font-weight:600;">{label}</span>
                <span style="color:{color}; font-family:'JetBrains Mono', monospace; font-weight:700;">
                    {blocks_fill}{blocks_empty} {pct}%
                </span>
            </div>
            <div style="background:#1E293B; height:8px; border-radius:4px; overflow:hidden;">
                <div style="background:linear-gradient(90deg, #0284C7, {color}); height:100%; width:{pct}%; border-radius:4px;"></div>
            </div>
        </div>
        """
    html += '</div>'
    return html


def plot_uplift_distribution(df_res: pd.DataFrame, title: str = "Estimated Incremental Impact Distribution") -> go.Figure:
    """
    Creates distribution histogram of estimated incremental uplift across users.
    """
    fig = go.Figure()
    
    impacts_pct = df_res["est_impact_pct"].values
    mean_val = np.mean(impacts_pct)
    
    fig.add_trace(go.Histogram(
        x=impacts_pct,
        nbinsx=35,
        marker=dict(
            color="#38BDF8",
            line=dict(color="#00F0FF", width=1)
        ),
        opacity=0.85,
        name="Users"
    ))
    
    # Add vertical mean indicator line
    fig.add_vline(
        x=mean_val,
        line_width=2,
        line_dash="dash",
        line_color="#00F0FF",
        annotation_text=f"Mean: +{mean_val:.1f}% pts",
        annotation_position="top right",
        annotation_font=dict(color="#00F0FF", family="JetBrains Mono")
    )
    
    fig.update_layout(
        NAVY_LAYOUT,
        title=dict(text=title, font=dict(color="#F8FAFC", size=15)),
        xaxis_title="Estimated Incremental Impact (Percentage Points)",
        yaxis_title="Anonymous User Count",
        height=360,
        showlegend=False
    )
    return fig


def plot_ad_comparison(df_ad_a: pd.DataFrame, df_ad_b: pd.DataFrame, label_a: str, label_b: str) -> go.Figure:
    """
    Overlays incremental impact distributions of two distinct ads over the same population.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_ad_a["est_impact_pct"],
        nbinsx=30,
        name=label_a,
        marker_color="rgba(0, 240, 255, 0.65)",
        opacity=0.75
    ))
    
    fig.add_trace(go.Histogram(
        x=df_ad_b["est_impact_pct"],
        nbinsx=30,
        name=label_b,
        marker_color="rgba(99, 102, 241, 0.65)",
        opacity=0.75
    ))
    
    fig.update_layout(
        NAVY_LAYOUT,
        barmode="overlay",
        title=dict(text="Comparative Incremental Response: Ad A vs. Ad B", font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Estimated Incremental Impact (Percentage Points)",
        yaxis_title="User Count",
        height=380
    )
    return fig


def plot_privacy_utility_frontier(df_curve: pd.DataFrame) -> go.Figure:
    """
    Draws dual-axis Privacy Protection vs. Utility & Error curve.
    """
    fig = go.Figure()
    
    # Model Utility trace (ROC AUC)
    fig.add_trace(go.Scatter(
        x=df_curve["Privacy Protection (%)"],
        y=df_curve["Model Utility (ROC AUC)"],
        mode="lines+markers",
        name="Model Utility (AUC)",
        line=dict(color="#00F0FF", width=3),
        marker=dict(size=8, symbol="circle")
    ))
    
    # Estimation Error trace (MAE) on secondary axis
    fig.add_trace(go.Scatter(
        x=df_curve["Privacy Protection (%)"],
        y=df_curve["Estimation Error (MAE)"],
        mode="lines+markers",
        name="Estimation Error (MAE)",
        line=dict(color="#F43F5E", width=3, dash="dot"),
        marker=dict(size=8, symbol="diamond"),
        yaxis="y2"
    ))
    
    layout = dict(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B132B",
        font=dict(color="#F8FAFC", family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            title="Privacy Protection Guarantee (%) [Higher = More Noise / Lower ε]",
            gridcolor="#1E293B",
            tickfont=dict(color="#94A3B8"),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        yaxis=dict(
            title="Model Utility (ROC AUC)",
            gridcolor="#1E293B",
            tickfont=dict(color="#00F0FF"),
            title_font=dict(color="#00F0FF", size=12)
        ),
        yaxis2=dict(
            title="Estimation Error (MAE)",
            overlaying="y",
            side="right",
            gridcolor="#1E293B",
            tickfont=dict(color="#F43F5E"),
            title_font=dict(color="#F43F5E", size=12)
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
            x=0.03, y=0.97
        ),
        title=dict(text="Empirical Privacy vs. Utility Frontier", font=dict(color="#F8FAFC", size=15)),
        height=380
    )
    fig.update_layout(layout)
    return fig


def plot_qini_curve(qini_data: dict, qini_score: float) -> go.Figure:
    """
    Renders the Qini / Cumulative Uplift curve against the random baseline.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=qini_data["population_fraction"],
        y=qini_data["model_uplift"],
        mode="lines+markers",
        name=f"ALTRIX Causal Uplift (Qini={qini_score})",
        line=dict(color="#00F0FF", width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=qini_data["population_fraction"],
        y=qini_data["random_baseline"],
        mode="lines",
        name="Random Policy Baseline",
        line=dict(color="#64748B", width=2, dash="dash")
    ))
    
    fig.update_layout(
        NAVY_LAYOUT,
        title=dict(text="Cumulative Uplift Curve (Qini Metric)", font=dict(size=15, color="#F8FAFC")),
        xaxis_title="Proportion of Targeted Anonymous Population",
        yaxis_title="Cumulative Incremental Conversions",
        height=360
    )
    return fig


def plot_federated_convergence(history: list) -> go.Figure:
    """
    Renders multi-round convergence curves (Loss & Accuracy) for Federated Learning.
    """
    fig = go.Figure()
    if not history:
        return fig

    rounds = [h["round"] for h in history]
    losses = [h["global_loss"] for h in history]
    accuracies = [h["global_accuracy"] * 100 for h in history]

    fig.add_trace(go.Scatter(
        x=rounds,
        y=losses,
        mode="lines+markers",
        name="Global Loss (Binary Cross-Entropy)",
        line=dict(color="#00F0FF", width=3),
        marker=dict(size=8, symbol="circle")
    ))

    fig.add_trace(go.Scatter(
        x=rounds,
        y=accuracies,
        mode="lines+markers",
        name="Global Accuracy (%)",
        line=dict(color="#10B981", width=3, dash="dash"),
        marker=dict(size=8, symbol="square"),
        yaxis="y2"
    ))

    layout = dict(
        paper_bgcolor="#0F172A",
        plot_bgcolor="#0B132B",
        font=dict(color="#F8FAFC", family="Plus Jakarta Sans, sans-serif"),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            title="Communication Round (t)",
            gridcolor="#1E293B",
            tickmode="linear",
            tick0=1,
            dtick=1,
            tickfont=dict(color="#94A3B8"),
            title_font=dict(color="#CBD5E1", size=12)
        ),
        yaxis=dict(
            title="Global Cross-Entropy Loss",
            gridcolor="#1E293B",
            tickfont=dict(color="#00F0FF"),
            title_font=dict(color="#00F0FF", size=12)
        ),
        yaxis2=dict(
            title="Global Test Accuracy (%)",
            overlaying="y",
            side="right",
            gridcolor="#1E293B",
            tickfont=dict(color="#10B981"),
            title_font=dict(color="#10B981", size=12)
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="#1E293B",
            borderwidth=1,
            x=0.03, y=0.97
        ),
        title=dict(text="Federated Learning Convergence Across Communication Rounds", font=dict(color="#F8FAFC", size=14)),
        height=340
    )
    fig.update_layout(layout)
    return fig


def plot_dp_weight_perturbation(raw_weights: list, perturbed_weights: list, feature_names: list) -> go.Figure:
    """
    Renders grouped bar chart comparing raw vs DP-perturbed model parameters.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=feature_names,
        y=raw_weights,
        name="Raw Baseline Weights (ε = ∞)",
        marker_color="#38BDF8",
        opacity=0.85
    ))

    fig.add_trace(go.Bar(
        x=feature_names,
        y=perturbed_weights,
        name="Differentially Private Weights (Current ε)",
        marker_color="#F43F5E",
        opacity=0.85
    ))

    fig.update_layout(
        NAVY_LAYOUT,
        barmode="group",
        title=dict(text="Parameter Perturbation: Unperturbed vs. Differentially Private Weights", font=dict(size=14, color="#F8FAFC")),
        xaxis_title="Behavioral Feature Weight Dimension",
        yaxis_title="Model Parameter Coefficient Value",
        height=320
    )
    return fig
