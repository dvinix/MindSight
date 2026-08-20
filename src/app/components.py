from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    import streamlit as st
except ImportError:
    st = None


def load_custom_css():
    if st is None:
        return
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #1C1917;
            background-color: #F7F6F2;
        }

        .stApp {
            background-color: #F7F6F2;
            color: #1C1917;
        }

        .main-header {
            background: #FFFFFF;
            border: 1px solid #E5E2DC;
            border-radius: 8px;
            padding: 20px 24px;
            margin-bottom: 20px;
        }

        .hero-title {
            font-size: 1.6rem;
            font-weight: 700;
            color: #1C1917;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .hero-subtitle {
            color: #78716C;
            font-size: 0.9rem;
            margin-top: 4px;
            font-weight: 400;
        }

        .card {
            background: #FFFFFF;
            border: 1px solid #E5E2DC;
            border-radius: 8px;
            padding: 16px 20px;
            margin-bottom: 16px;
        }

        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E2DC;
            border-radius: 8px;
            padding: 14px 16px;
            text-align: left;
        }

        .metric-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #78716C;
            font-weight: 600;
            margin-bottom: 2px;
        }

        .metric-value {
            font-size: 1.4rem;
            font-weight: 600;
            color: #1C1917;
            font-family: 'IBM Plex Mono', monospace;
        }

        .badge-risk-high {
            background: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FCA5A5;
            border-radius: 4px;
            padding: 4px 10px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }

        .badge-risk-low {
            background: #F0FDF4;
            color: #166534;
            border: 1px solid #86EFAC;
            border-radius: 4px;
            padding: 4px 10px;
            font-weight: 600;
            font-size: 0.8rem;
            display: inline-block;
        }

        .chip-risk {
            background: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
            border-radius: 4px;
            padding: 2px 7px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            display: inline-block;
            margin: 2px;
        }

        .chip-protective {
            background: #F0FDF4;
            color: #166534;
            border: 1px solid #BBF7D0;
            border-radius: 4px;
            padding: 2px 7px;
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.8rem;
            display: inline-block;
            margin: 2px;
        }

        .crisis-box {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-left: 3px solid #D97706;
            border-radius: 6px;
            padding: 12px 16px;
            margin-top: 16px;
        }

        .disclaimer-banner {
            background: #F5F4F0;
            border: 1px solid #E5E2DC;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 0.8rem;
            color: #57534E;
            margin-top: 14px;
        }

        .stButton>button {
            background-color: #1C1917;
            color: #FFFFFF;
            border: 1px solid #1C1917;
            border-radius: 6px;
            font-weight: 500;
            padding: 8px 18px;
            font-size: 0.9rem;
            box-shadow: none;
            transition: background-color 0.15s ease;
        }

        .stButton>button:hover {
            background-color: #44403C;
            border-color: #44403C;
            color: #FFFFFF;
        }

        .stTextArea textarea {
            background-color: #FFFFFF;
            border: 1px solid #D6D3CD;
            border-radius: 6px;
            color: #1C1917;
            font-family: 'Inter', sans-serif;
            font-size: 0.92rem;
        }

        .stTextArea textarea:focus {
            border-color: #1C1917;
            box-shadow: 0 0 0 1px #1C1917;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            background-color: #FFFFFF;
            border-color: #D6D3CD;
            border-radius: 6px;
            color: #1C1917;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_header():
    if st is None:
        return
    st.markdown(
        """
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h1 class="hero-title">MindSight</h1>
                    <div class="hero-subtitle">
                        Conversational Analytics & Screening Aid for Psychological Distress
                    </div>
                </div>
                <div style="font-size: 0.75rem; color: #78716C; font-family: 'IBM Plex Mono', monospace;">
                    Model: Dreaddit Reddit Corpus | Recall-Prioritized
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_gauge_chart(confidence: float, threshold: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'color': "#1C1917", 'family': "IBM Plex Mono", 'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#A8A29E", 'tickfont': {'color': "#78716C", 'size': 10}},
            'bar': {'color': "#292524", 'thickness': 0.22},
            'bgcolor': "#F5F4F0",
            'borderwidth': 1,
            'bordercolor': "#E5E2DC",
            'steps': [
                {'range': [0, threshold * 100], 'color': "#F0FDF4"},
                {'range': [threshold * 100, 100], 'color': "#FEF2F2"}
            ],
            'threshold': {
                'line': {'color': "#D97706", 'width': 2},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=20, b=10),
        height=160
    )
    return fig


def render_explanation_chips(explanations: List[Dict[str, Any]]):
    if st is None:
        return
    if not explanations:
        st.markdown('<div style="font-size: 0.85rem; color: #78716C;">No significant token attributions identified.</div>', unsafe_allow_html=True)
        return

    html_chips = []
    for item in explanations:
        word = item.get("word", "")
        weight = item.get("weight", item.get("contribution", 0.0))
        if weight >= 0:
            html_chips.append(f'<span class="chip-risk">+{weight:.2f} {word}</span>')
        else:
            html_chips.append(f'<span class="chip-protective">{weight:.2f} {word}</span>')

    st.markdown(f'<div style="margin-top: 6px;">{" ".join(html_chips)}</div>', unsafe_allow_html=True)


def render_feature_bar_chart(features: Dict[str, Any]):
    keys = list(features.keys())
    values = list(features.values())
    clean_keys = [k.replace("_", " ").title() for k in keys]

    fig = go.Figure(go.Bar(
        x=values,
        y=clean_keys,
        orientation='h',
        marker=dict(
            color=['#44403C' if v >= 0 else '#57534E' for v in values],
            line=dict(color='#E5E2DC', width=1)
        )
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#57534E", family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=150,
        xaxis=dict(gridcolor="#E5E2DC", zerolinecolor="#D6D3CD"),
        yaxis=dict(gridcolor="#E5E2DC")
    )
    return fig


def get_preset_samples() -> Dict[str, str]:
    return {
        "High Stress & Anxiety (Dreaddit Sample)": (
            "I've been feeling completely overwhelmed and anxious for weeks. Every morning I wake up terrified of what "
            "the day will bring. I can't concentrate at work, my chest feels constantly tight, and I feel like nobody understands "
            "how utterly exhausted and hopeless I am right now."
        ),
        "Academic & Work Burnout": (
            "I have three major deadlines this week and I haven't slept more than 3 hours a night. I feel completely "
            "trapped and paralyzed by fear of failing. I don't know who to talk to and I am breaking down."
        ),
        "Neutral Daily Routine": (
            "Woke up today, had some coffee and went for a 20-minute morning jog around the park. Looking forward to "
            "reading the new book I ordered online and finishing some light grocery shopping in the afternoon."
        ),
        "Positive Recovery & Support": (
            "Today was a really peaceful and uplifting day. I finally took time to meditate, reconnect with an old friend, "
            "and felt truly grateful for the progress and healing I've made over the last few months."
        )
    }


def render_crisis_resources():
    if st is None:
        return
    st.markdown(
        """
        <div class="crisis-box">
            <div style="font-weight: 600; color: #92400E; font-size: 0.85rem; margin-bottom: 2px;">
                Crisis & Emotional Support Helplines
            </div>
            <div style="color: #78350F; font-size: 0.8rem; line-height: 1.45;">
                If you or someone you know is in acute distress, free confidential help is available 24/7:
                <ul style="margin: 4px 0 0 16px; padding: 0;">
                    <li><b>US & Canada:</b> Call or text <b>988</b> (Suicide & Crisis Lifeline)</li>
                    <li><b>Crisis Text Line:</b> Text <b>HOME</b> to <b>741741</b></li>
                    <li><b>India (Tele-MANAS):</b> Call <b>14416</b> or <b>1800-891-4416</b></li>
                    <li><b>International:</b> <a href="https://findahelpline.com" target="_blank" style="color: #B45309;">findahelpline.com</a></li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
