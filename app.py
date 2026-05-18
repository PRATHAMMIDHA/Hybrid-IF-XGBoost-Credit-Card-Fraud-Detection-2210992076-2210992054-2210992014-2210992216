"""
Streamlit Fraud Detection App
Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ─── Page Config ───
st.set_page_config(
    page_title="FraudShield AI — Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(90deg, #e0f7fa, #80deea, #4dd0e1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #90caf9;
        margin: 0.5rem 0 0;
        font-size: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #1565c0);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #4dd0e1;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #90caf9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .fraud-alert {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(183,28,28,0.4);
        animation: pulse 2s infinite;
    }
    .legit-alert {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        box-shadow: 0 4px 20px rgba(27,94,32,0.4);
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 4px 20px rgba(183,28,28,0.4); }
        50% { box-shadow: 0 4px 30px rgba(183,28,28,0.8); }
    }
    .sidebar-info {
        background: #1e2a3a;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4dd0e1;
    }
    .stSlider > div > div > div { background: #4dd0e1; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ───
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'outputs', 'models')

@st.cache_resource
def load_models():
    try:
        hybrid = joblib.load(os.path.join(MODELS_DIR, 'hybrid_model.pkl'))
        rf = joblib.load(os.path.join(MODELS_DIR, 'random_forest.pkl'))
        xgb_model = joblib.load(os.path.join(MODELS_DIR, 'xgboost.pkl'))
        scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        metadata = joblib.load(os.path.join(MODELS_DIR, 'metadata.pkl'))
        return hybrid, rf, xgb_model, scaler, metadata
    except Exception as e:
        return None, None, None, None, None

hybrid_model, rf_model, xgb_mdl, scaler, metadata = load_models()

# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>🛡️ FraudShield AI</h1>
    <p>Hybrid Isolation Forest + XGBoost Credit Card Fraud Detection System</p>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("## ⚙️ Analysis Mode")
    mode = st.radio("", ["🔍 Single Transaction", "📋 Batch CSV Upload", "📊 Model Performance"], index=0)
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-info">
        <strong>🧠 Models Active</strong><br>
        • Hybrid IF + XGBoost<br>
        • Random Forest<br>
        • XGBoost (base)
    </div>
    """, unsafe_allow_html=True)
    if metadata:
        threshold = metadata.get('optimal_threshold', 0.5)
        st.markdown(f"""
        <div class="sidebar-info">
            <strong>⚡ Optimal Threshold</strong><br>
            Cost-optimized: <strong>{threshold:.3f}</strong><br>
            (FN=$500, FP=$10)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Dataset:** Kaggle Credit Card Fraud")
    st.markdown("**Records:** 284,807 transactions")
    st.markdown("**Fraud Rate:** 0.172%")

# ─── Main Content ───
if models_loaded := (hybrid_model is not None):
    threshold = metadata.get('optimal_threshold', 0.5) if metadata else 0.5

    # ────────────────────────────────────────────────────────
    if "Single" in mode:
        st.markdown("## 🔍 Single Transaction Analysis")
        st.markdown("*Enter transaction features to predict fraud probability*")

        col1, col2, col3 = st.columns(3)
        with col1:
            amount = st.number_input("💵 Transaction Amount (€)", min_value=0.0,
                                     max_value=30000.0, value=150.0, step=10.0)
            hour = st.slider("🕐 Hour of Day (0-23)", 0, 23, 14)

        with col2:
            st.markdown("**V1 – V10 (PCA Features)**")
            v_vals_1 = {}
            for i in range(1, 11):
                v_vals_1[f'V{i}'] = st.number_input(f"V{i}", value=0.0,
                                                     format="%.4f", key=f"v{i}")
        with col3:
            st.markdown("**V11 – V28 (PCA Features)**")
            v_vals_2 = {}
            for i in range(11, 29):
                v_vals_2[f'V{i}'] = st.number_input(f"V{i}", value=0.0,
                                                     format="%.4f", key=f"v{i}")

        if st.button("🚀 Analyze Transaction", use_container_width=True, type="primary"):
            # Build feature vector
            v_vals = {**v_vals_1, **v_vals_2}
            feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount_log', 'Hour']
            amount_log = np.log1p(amount)
            # Scale Amount_log and Hour
            scaled_last2 = scaler.transform([[amount_log, hour]])[0]
            feature_vector = np.array([v_vals[f'V{i}'] for i in range(1, 29)] + list(scaled_last2))

            # Anomaly score
            from sklearn.ensemble import IsolationForest
            anomaly_score = 0.5  # Default if IF not available
            feature_hybrid = np.append(feature_vector, anomaly_score).reshape(1, -1)

            # Predictions
            prob_hybrid = hybrid_model.predict_proba(feature_hybrid)[0][1]
            prob_rf = rf_model.predict_proba(feature_vector.reshape(1, -1))[0][1]
            prob_xgb = xgb_mdl.predict_proba(feature_vector.reshape(1, -1))[0][1]

            is_fraud = prob_hybrid >= threshold
            ensemble_prob = (prob_hybrid * 0.5 + prob_rf * 0.25 + prob_xgb * 0.25)

            st.markdown("---")
            # Verdict
            if is_fraud:
                st.markdown(f"""
                <div class="fraud-alert">
                    ⚠️ FRAUD DETECTED — Confidence: {prob_hybrid*100:.1f}%
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="legit-alert">
                    ✅ LEGITIMATE TRANSACTION — Confidence: {(1-prob_hybrid)*100:.1f}%
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Gauge Chart
            col_g1, col_g2 = st.columns([2, 1])
            with col_g1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_hybrid * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fraud Probability (%)", 'font': {'size': 18}},
                    delta={'reference': threshold * 100, 'increasing': {'color': "#F44336"}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickwidth': 1},
                        'bar': {'color': "#F44336" if is_fraud else "#4CAF50"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, threshold*100], 'color': '#E8F5E9'},
                            {'range': [threshold*100, 100], 'color': '#FFEBEE'}
                        ],
                        'threshold': {
                            'line': {'color': "orange", 'width': 4},
                            'thickness': 0.75,
                            'value': threshold * 100
                        }
                    }
                ))
                fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col_g2:
                st.markdown("**Model Probabilities**")
                fig_bar = go.Figure(go.Bar(
                    x=['Hybrid', 'RF', 'XGBoost'],
                    y=[prob_hybrid*100, prob_rf*100, prob_xgb*100],
                    marker_color=['#F44336' if p >= threshold else '#4CAF50'
                                  for p in [prob_hybrid, prob_rf, prob_xgb]],
                    text=[f'{p*100:.1f}%' for p in [prob_hybrid, prob_rf, prob_xgb]],
                    textposition='auto'
                ))
                fig_bar.update_layout(yaxis_title='Fraud Probability (%)',
                                     paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', height=280)
                st.plotly_chart(fig_bar, use_container_width=True)

    # ────────────────────────────────────────────────────────
    elif "Batch" in mode:
        st.markdown("## 📋 Batch Transaction Analysis")
        uploaded = st.file_uploader("Upload CSV with transaction data", type=['csv'])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(df_up):,}** transactions")
            feature_cols_base = [f'V{i}' for i in range(1, 29)]
            if all(f in df_up.columns for f in feature_cols_base):
                df_up['Amount_log'] = np.log1p(df_up.get('Amount', 0))
                df_up['Hour'] = (df_up.get('Time', 0) // 3600) % 24
                feature_cols_full = [f'V{i}' for i in range(1, 29)] + ['Amount_log', 'Hour']
                X_batch = df_up[feature_cols_full].values
                X_batch[:, -2:] = scaler.transform(X_batch[:, -2:])
                probs = xgb_mdl.predict_proba(X_batch)[:, 1]
                df_up['Fraud_Probability'] = probs
                df_up['Prediction'] = (probs >= threshold).astype(int)
                df_up['Risk_Level'] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                             labels=['Low', 'Medium', 'High'])

                # Summary
                n_fraud = (df_up['Prediction'] == 1).sum()
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"<div class='metric-card'><div class='value'>{len(df_up):,}</div><div class='label'>Total Transactions</div></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div class='metric-card'><div class='value'>{n_fraud}</div><div class='label'>Fraud Detected</div></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div class='metric-card'><div class='value'>{n_fraud/len(df_up)*100:.2f}%</div><div class='label'>Fraud Rate</div></div>", unsafe_allow_html=True)
                with col4:
                    est_loss = n_fraud * 150
                    st.markdown(f"<div class='metric-card'><div class='value'>€{est_loss:,}</div><div class='label'>Est. Fraud Value</div></div>", unsafe_allow_html=True)

                # Results Table
                st.markdown("### Flagged Transactions")
                fraud_df = df_up[df_up['Prediction'] == 1][['Fraud_Probability', 'Risk_Level'] +
                           (['Amount'] if 'Amount' in df_up.columns else [])].round(4)
                st.dataframe(fraud_df.style.background_gradient(subset=['Fraud_Probability'],
                             cmap='Reds'), use_container_width=True)

                csv_out = df_up.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Results CSV", csv_out,
                                  "fraud_predictions.csv", "text/csv")
            else:
                st.error("CSV must contain V1–V28 columns matching the creditcard dataset format.")

    # ────────────────────────────────────────────────────────
    elif "Performance" in mode:
        st.markdown("## 📊 Model Performance Dashboard")

        # Explanation banner
        st.info(
            "**Why is raw Accuracy not shown?**  \n"
            "With a **578:1 class imbalance**, even a naive model that predicts every transaction "
            "as 'Legitimate' achieves **99.83% accuracy** — yet it catches **zero fraud**.  \n"
            "We use **Balanced Accuracy** (average recall per class), **F1-Score**, **MCC**, and "
            "**PR-AUC** as they truly reflect fraud detection capability."
        )

        # Load comparison table
        cmp_path = os.path.join(BASE_DIR, 'outputs', 'model_comparison.csv')
        if os.path.exists(cmp_path):
            comp_df = pd.read_csv(cmp_path)
            # Support both old and new column naming
            acc_col = 'Bal_Accuracy' if 'Bal_Accuracy' in comp_df.columns else 'Accuracy'
            metrics = [acc_col, 'Precision', 'Recall', 'F1', 'MCC']
            metric_labels = {
                'Bal_Accuracy': 'Balanced Accuracy',
                'Accuracy':     'Balanced Accuracy',
                'Precision':    'Precision',
                'Recall':       'Recall',
                'F1':           'F1 Score',
                'MCC':          'MCC',
            }
            # Handle missing MCC column gracefully
            metrics = [m for m in metrics if m in comp_df.columns]

            fig = go.Figure()
            colors_p = px.colors.qualitative.Plotly
            display_labels = [metric_labels.get(m, m) for m in metrics]
            for i, row in comp_df.iterrows():
                vals = [row[m] for m in metrics]
                vals_closed  = vals + [vals[0]]
                theta_closed = display_labels + [display_labels[0]]
                fig.add_trace(go.Scatterpolar(
                    r=vals_closed, theta=theta_closed, fill='toself',
                    name=row['Model'], line_color=colors_p[i % len(colors_p)],
                    opacity=0.7
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title='Model Performance Radar Chart<br><sup>Balanced Accuracy shown (Raw Accuracy excluded — misleading for imbalanced data)</sup>',
                paper_bgcolor='rgba(0,0,0,0)', height=520
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Full Metrics Table")
            display_cols = ['Model'] + metrics
            rename_map   = {m: metric_labels.get(m, m) for m in metrics}
            display_df   = comp_df[display_cols].rename(columns=rename_map)
            styled = display_df.style \
                .highlight_max(axis=0, color='#c8e6c9',
                               subset=[metric_labels.get(m, m) for m in metrics]) \
                .format({metric_labels.get(m, m): '{:.4f}' for m in metrics})
            st.dataframe(styled, use_container_width=True)

            # Show plots
            PLOTS_DIR_APP = os.path.join(BASE_DIR, 'outputs', 'plots')
            plot_tabs = st.tabs(["ROC Curves", "PR Curves", "Confusion Matrices",
                                "Feature Importance", "Cost Threshold"])
            plot_files = ['08_roc_curves.png', '09_pr_curves.png',
                         '10_confusion_matrices.png', '11_feature_importance.png',
                         '07_cost_threshold_analysis.png']
            for tab, pfile in zip(plot_tabs, plot_files):
                with tab:
                    ppath = os.path.join(PLOTS_DIR_APP, pfile)
                    if os.path.exists(ppath):
                        st.image(ppath, use_column_width=True)
        else:
            st.warning("No model comparison data found. Please run fraud_detection.py first.")

else:
    st.error("⚠️ Models not found. Please run `python fraud_detection.py` first to train the models.")
    st.code("python fraud_detection.py", language="bash")
