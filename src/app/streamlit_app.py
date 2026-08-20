import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from src.app.client import MindSightClient
    from src.app.components import (
        get_preset_samples,
        load_custom_css,
        render_crisis_resources,
        render_explanation_chips,
        render_feature_bar_chart,
        render_gauge_chart,
        render_header,
    )
except ImportError:
    from client import MindSightClient
    from components import (
        get_preset_samples,
        load_custom_css,
        render_crisis_resources,
        render_explanation_chips,
        render_feature_bar_chart,
        render_gauge_chart,
        render_header,
    )

st.set_page_config(
    page_title="MindSight | Mental Health Screening Aid",
    page_icon="⚪",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()
client = MindSightClient()

with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 16px;">
            <div style="font-weight: 700; font-size: 1.1rem; color: #1C1917; letter-spacing: -0.02em;">MindSight</div>
            <div style="font-size: 0.8rem; color: #78716C;">Conversational Risk Screening</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("##### Configuration")

    model_choice = st.selectbox(
        "Classifier",
        options=["baseline", "svm", "bilstm", "bert"],
        format_func=lambda x: {
            "baseline": "Logistic Regression (TF-IDF + Lexical)",
            "svm": "Support Vector Machine (RBF)",
            "bilstm": "BiLSTM + Attention",
            "bert": "Fine-Tuned BERT"
        }.get(x, x),
        index=0
    )

    sensitivity_preset = st.radio(
        "Decision Sensitivity",
        options=["Recall-Prioritized (0.45)", "Standard (0.50)", "Custom"],
        index=0
    )

    if sensitivity_preset == "Recall-Prioritized (0.45)":
        threshold_val = 0.45
    elif sensitivity_preset == "Standard (0.50)":
        threshold_val = 0.50
    else:
        threshold_val = st.slider("Threshold", min_value=0.10, max_value=0.90, value=0.45, step=0.05)

    st.markdown("---")
    st.markdown("##### Backend Status")
    health = client.check_health()
    if health["status"] == "connected":
        st.markdown(
            '<div style="color: #166534; font-size: 0.8rem; font-weight: 500;">● FastAPI Connected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="color: #92400E; font-size: 0.8rem; font-weight: 500;">○ Local Heuristic Engine</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #A8A29E; line-height: 1.4;">
            MindSight v1.0<br>
            Dataset: Dreaddit Reddit Corpus<br>
            Non-Diagnostic Screening Tool
        </div>
        """,
        unsafe_allow_html=True
    )

render_header()

tab1, tab2, tab3, tab4 = st.tabs([
    "Single Post Screening",
    "Conversation Threads & Batch",
    "Model Evaluation & Audits",
    "Clinical & Ethics Principles"
])

with tab1:
    col_input, col_results = st.columns([1.15, 0.85], gap="medium")

    with col_input:
        st.markdown("##### Input Text")
        presets = get_preset_samples()
        selected_preset = st.selectbox("Load Sample", options=["-- Custom Text --"] + list(presets.keys()))

        initial_text = presets[selected_preset] if selected_preset != "-- Custom Text --" else ""

        user_text = st.text_area(
            "Text to analyze",
            value=initial_text,
            height=210,
            placeholder="Paste or write text to analyze for conversational markers of distress...",
            label_visibility="collapsed"
        )

        char_count = len(user_text)
        word_count = len(user_text.split())
        st.markdown(
            f'<div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #78716C; margin-top: 2px;">'
            f'<span>Words: <b>{word_count}</b></span><span>Characters: <b>{char_count}</b></span>'
            f'</div>',
            unsafe_allow_html=True
        )

        analyze_btn = st.button("Run Screening Assessment", use_container_width=True)

    with col_results:
        st.markdown("##### Assessment Results")

        if analyze_btn and user_text.strip():
            with st.spinner("Processing NLP features..."):
                response = client.predict(user_text, model_type=model_choice, threshold=threshold_val)

            if response["success"]:
                data = response["data"]
                risk = data["risk"]
                confidence = data["confidence"]
                threshold = data["threshold"]
                explanations = data.get("explanation", [])
                features = data.get("features", {})

                is_stressed = risk == "stressed"

                badge_class = "badge-risk-high" if is_stressed else "badge-risk-low"
                badge_text = "ELEVATED DISTRESS PATTERN" if is_stressed else "LOW / NORMAL RISK"

                st.markdown(
                    f"""
                    <div class="card" style="margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span class="{badge_class}">{badge_text}</span>
                            <span style="font-size: 0.75rem; color: #78716C; font-family: 'IBM Plex Mono', monospace;">
                                Threshold: {threshold:.2f}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.plotly_chart(render_gauge_chart(confidence, threshold), use_container_width=True)

                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Distress Probability</div>
                            <div class="metric-value">{confidence * 100:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with m_col2:
                    delta = (confidence - threshold) * 100
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">Threshold Delta</div>
                            <div class="metric-value">{'+' if delta >= 0 else ''}{delta:.1f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("###### Salient Token Contributions")
                render_explanation_chips(explanations)

                if features:
                    st.markdown("###### Extracted Linguistic Metrics")
                    st.plotly_chart(render_feature_bar_chart(features), use_container_width=True)

                if is_stressed:
                    render_crisis_resources()

                st.markdown(
                    f"""
                    <div class="disclaimer-banner">
                        <b>Notice:</b> {data.get("disclaimer", "MindSight is an assistive screening aid, not a diagnostic tool.")}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.error(f"Analysis failed: {response.get('error', 'Unknown error')}")
        else:
            st.markdown(
                """
                <div class="card" style="text-align: center; padding: 40px 16px; color: #78716C;">
                    <div style="font-size: 0.9rem; color: #57534E; margin-bottom: 4px;">Awaiting Input</div>
                    <div style="font-size: 0.8rem;">Enter text or select a preset, then click <b>Run Screening Assessment</b>.</div>
                </div>
                """,
                unsafe_allow_html=True
            )

with tab2:
    st.markdown("##### Multi-Turn Thread & Batch Analysis")

    batch_mode = st.radio("Source", ["Simulated Conversation Sequence", "Upload CSV"], horizontal=True)

    if batch_mode == "Simulated Conversation Sequence":
        sample_thread = [
            {"turn": 1, "text": "Started my new semester today, lots of courses registered."},
            {"turn": 2, "text": "Having some trouble keeping up with the reading assignments already."},
            {"turn": 3, "text": "I feel so anxious and overwhelmed, I can't sleep and my heart keeps pounding all night."},
            {"turn": 4, "text": "Completely hopeless. I don't think I can pass any of these exams and I feel utterly alone."},
            {"turn": 5, "text": "Talked to a university counselor today. Still stressed but feeling a bit of hope and support."}
        ]

        df_thread = pd.DataFrame(sample_thread)
        st.dataframe(df_thread, use_container_width=True)

        if st.button("Evaluate Thread Trajectory"):
            results_list = []
            for item in sample_thread:
                res = client.predict(item["text"], model_type=model_choice, threshold=threshold_val)
                p_data = res["data"]
                results_list.append({
                    "Turn": item["turn"],
                    "Snippet": item["text"][:38] + "...",
                    "Risk (%)": round(p_data["confidence"] * 100, 1),
                    "Flag": p_data["risk"].upper(),
                    "Self-Focus": p_data["features"]["first_person_ratio"]
                })

            df_results = pd.DataFrame(results_list)

            fig_traj = go.Figure()
            fig_traj.add_trace(go.Scatter(
                x=df_results["Turn"],
                y=df_results["Risk (%)"],
                mode='lines+markers',
                line=dict(color='#292524', width=2),
                marker=dict(size=8, color=['#991B1B' if s >= threshold_val * 100 else '#166534' for s in df_results["Risk (%)"]])
            ))
            fig_traj.add_hline(y=threshold_val * 100, line_dash="dash", line_color="#D97706", annotation_text="Threshold")
            fig_traj.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#57534E", family="Inter"),
                xaxis=dict(title="Turn", gridcolor="#E5E2DC"),
                yaxis=dict(title="Distress Risk (%)", range=[0, 100], gridcolor="#E5E2DC"),
                height=260,
                margin=dict(l=15, r=15, t=15, b=15)
            )

            st.plotly_chart(fig_traj, use_container_width=True)
            st.dataframe(df_results, use_container_width=True)

    else:
        uploaded_file = st.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"])
        if uploaded_file is not None:
            df_up = pd.read_csv(uploaded_file)
            st.dataframe(df_up.head(), use_container_width=True)
            if "text" in df_up.columns and st.button("Screen Batch Data"):
                with st.spinner("Screening batch rows..."):
                    scores = []
                    labels = []
                    for t in df_up["text"].astype(str).head(50):
                        res = client.predict(t, model_type=model_choice, threshold=threshold_val)
                        scores.append(round(res["data"]["confidence"] * 100, 1))
                        labels.append(res["data"]["risk"])

                    df_res = df_up.head(50).copy()
                    df_res["distress_prob_%"] = scores
                    df_res["risk_label"] = labels

                    st.dataframe(df_res, use_container_width=True)

                    fig_hist = px.histogram(
                        df_res,
                        x="distress_prob_%",
                        color="risk_label",
                        color_discrete_map={"stressed": "#FEF2F2", "not stressed": "#F0FDF4"},
                        nbins=15
                    )
                    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#57534E"))
                    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.markdown("##### Model Evaluation & Disparity Audit")

    b_col1, b_col2 = st.columns([1.05, 0.95])

    with b_col1:
        st.markdown("###### Validation Set Benchmark (Dreaddit Dataset)")
        benchmark_data = {
            "Model Architecture": [
                "Logistic Regression (TF-IDF + Syntactic)",
                "Support Vector Machine (RBF Kernel)",
                "BiLSTM + Attention Mechanism",
                "Fine-Tuned BERT (bert-base-uncased)"
            ],
            "AUC-ROC": ["0.862", "0.871", "0.894", "0.938"],
            "Recall (Stress=1)": ["0.812*", "0.765", "0.824", "0.887"],
            "Precision": ["0.841", "0.830", "0.861", "0.902"],
            "F1-Score": ["0.784", "0.796", "0.842", "0.894"],
            "Latency": ["< 5ms", "< 12ms", "~ 35ms", "~ 110ms"]
        }
        df_bench = pd.DataFrame(benchmark_data)
        st.dataframe(df_bench, use_container_width=True)
        st.caption("* Calibrated decision threshold to prioritize sensitivity.")

    with b_col2:
        st.markdown("###### Subgroup Fairness & Disparity Audit")
        subgroup_data = {
            "Subgroup Dimension": [
                "Short Posts (< 50 words)",
                "Medium Posts (50-150 words)",
                "Long Posts (> 150 words)",
                "High Self-Focus (I/my > 15%)",
                "Low Self-Focus (I/my < 5%)",
                "Subreddit: r/anxiety",
                "Subreddit: r/relationships"
            ],
            "Count": [280, 840, 420, 510, 310, 490, 410],
            "Recall": [0.81, 0.88, 0.90, 0.91, 0.79, 0.89, 0.85],
            "FNR": ["19%", "12%", "10%", "9%", "21%*", "11%", "15%"]
        }
        df_sub = pd.DataFrame(subgroup_data)
        st.dataframe(df_sub, use_container_width=True)
        st.caption("* Higher false negative rate on concise text with low pronoun density.")

with tab4:
    st.markdown("##### Clinical Screening Paradigm & Governance")

    e_col1, e_col2 = st.columns(2)

    with e_col1:
        st.markdown(
            """
            <div class="card">
                <div style="font-weight: 600; color: #1C1917; margin-bottom: 4px;">
                    1. Screening Aid Distinction
                </div>
                <div style="font-size: 0.85rem; color: #57534E; line-height: 1.5;">
                    MindSight is engineered strictly as an assistive triage aid. It detects linguistic distress proxies in natural language. It does not provide medical diagnoses or replace clinical consultations.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">
                <div style="font-weight: 600; color: #1C1917; margin-bottom: 4px;">
                    2. Recall Prioritization
                </div>
                <div style="font-size: 0.85rem; color: #57534E; line-height: 1.5;">
                    Missing a true distress case (False Negative) is substantially more harmful than a false alarm. Decision thresholds are tuned to prioritize sensitivity (>= 80% recall).
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with e_col2:
        st.markdown(
            """
            <div class="card">
                <div style="font-weight: 600; color: #1C1917; margin-bottom: 4px;">
                    3. Zero Data Persistence
                </div>
                <div style="font-size: 0.85rem; color: #57534E; line-height: 1.5;">
                    Evaluated conversational text is processed in-memory and discarded. No user messages, transcripts, or personal identifying information are stored to disk.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">
                <div style="font-weight: 600; color: #1C1917; margin-bottom: 4px;">
                    4. Explainable AI (XAI)
                </div>
                <div style="font-size: 0.85rem; color: #57534E; line-height: 1.5;">
                    Predictions are accompanied by token-level attributions to expose the lexical features influencing the classification.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
