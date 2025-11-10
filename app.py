import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────
# Page Configuration (must be first)
# ──────────────────────────────
st.set_page_config(
    page_title="Hybrid Explainable Predictive Maintenance — Healthcare Equipment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import Keras model loader - try standalone keras first, then tensorflow.keras
load_model = None
keras_error = None

try:
    from keras.models import load_model
except (ImportError, OSError, Exception) as e:
    keras_error = str(e)
    try:
        from tensorflow.keras.models import load_model
    except (ImportError, OSError, Exception) as e2:
        keras_error = str(e2)
        load_model = None

# Check if we successfully imported load_model
if load_model is None:
    st.error("""
    # ⚠️ TensorFlow/Keras DLL Error
    
    **The application cannot load TensorFlow/Keras due to missing DLL files.**
    
    ## Solution: Install Microsoft Visual C++ Redistributable
    
    This is a common issue on Windows. Please follow these steps:
    
    ### Step 1: Download and Install Visual C++ Redistributable
    
    1. **Download** the Microsoft Visual C++ Redistributable:
       - Direct link: https://aka.ms/vs/17/release/vc_redist.x64.exe
       - Or search for "Visual C++ Redistributable 2015-2022" on Microsoft's website
    
    2. **Run the installer** and follow the installation wizard
    
    3. **Restart your computer** (recommended)
    
    ### Step 2: Restart the Application
    
    After installing, close this window and run the app again:
    ```bash
    streamlit run app.py
    ```
    
    ---
    
    **Error Details:**
    ```
    {error}
    ```
    
    **Alternative Solutions:**
    - If the above doesn't work, try using Python 3.11 instead of 3.13
    - Or use a virtual environment with Python 3.11
    """.format(error=keras_error))
    st.stop()

# ──────────────────────────────
# Load Models & Data
# ──────────────────────────────
@st.cache_resource
def load_models():
    """Load all models and data once"""
    try:
        scaler = joblib.load("scaler.pkl")
        xgb_maint = joblib.load("xgb_maint_class.pkl")
        xgb_fail = joblib.load("xgb_failtype.pkl")
        label_encoders = joblib.load("label_encoders.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        svd = joblib.load("svd_transformer.pkl")
        kmeans = joblib.load("kmeans_failtypes.pkl")
        cluster_keywords = joblib.load("cluster_keywords.pkl")
        encoder_model = load_model("lstm_encoder.keras", compile=False)
        df = pd.read_csv("Medical_Device_Failure_dataset.csv")
        return scaler, xgb_maint, xgb_fail, label_encoders, tfidf, svd, kmeans, cluster_keywords, encoder_model, df
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

scaler, xgb_maint, xgb_fail, label_encoders, tfidf, svd, kmeans, cluster_keywords, encoder_model, df = load_models()

# ──────────────────────────────
# Sidebar
# ──────────────────────────────
with st.sidebar:
    st.title("🩺 Hybrid Explainable Predictive Maintenance")
    st.markdown("---")
    st.markdown("### Healthcare Equipment")
    st.markdown("""
    An AI-powered predictive maintenance system for healthcare equipment 
    that combines **LSTM**, **XGBoost**, and **NLP** techniques to predict 
    maintenance classes and failure types.
    
    **Features:**
    - Maintenance class prediction (1-3)
    - Failure type probability analysis
    - NLP-based failure reason extraction
    - Similar maintenance report matching
    """)
    st.markdown("---")
    st.markdown("""
    **Project by:** *Anieruth Sridhar*
    
    Part of a healthcare predictive maintenance system.
    """)
    st.markdown("---")

# ──────────────────────────────
# Main Content
# ──────────────────────────────
st.title("🩺 Hybrid Explainable Predictive Maintenance Dashboard")
st.markdown("An AI-powered system to predict maintenance risk and analyze failure trends for healthcare equipment using **LSTM + XGBoost + NLP**.")

# ──────────────────────────────
# Create Tabs
# ──────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Analytics", "💬 Failure Insights", "🌱 Sustainability"])

# ──────────────────────────────
# TAB 1: Prediction
# ──────────────────────────────
with tab1:
    st.subheader("Enter Equipment Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        device_type = st.selectbox("Device Type", sorted(df["Device_Type"].unique()))
        manufacturer = st.selectbox("Manufacturer", sorted(df["Manufacturer"].unique()))
        model_name = st.selectbox("Model", sorted(df["Model"].unique()))

    with col2:
        country = st.selectbox("Country", sorted(df["Country"].unique()))
        age = st.slider("Age (years)", 0, 30, 5)
        downtime = st.slider("Downtime (hours)", 0.0, 200.0, 10.0, step=0.1)

    with col3:
        maintenance_cost = st.number_input("Maintenance Cost ($)", min_value=0.0, value=2000.0, step=100.0)
        maintenance_frequency = st.slider("Maintenance Frequency (per year)", 0, 12, 3)
        failure_event_count = st.slider("Failure Event Count", 0, 20, 1)

    report_text = st.text_area(
        "Maintenance Report (optional — improves NLP-based failure prediction)",
        height=120,
        placeholder="Enter maintenance report text here..."
    )

    # ──────────────────────────────
    # Prediction Button
    # ──────────────────────────────
    if st.button("🚀 Predict Equipment Health", type="primary", use_container_width=True):
        with st.spinner("Processing prediction..."):
            try:
                # ──────────────────────────────
                # Prepare Structured Features
                # ──────────────────────────────
                input_struct = pd.DataFrame([{
                    "Device_Type": device_type,
                    "Manufacturer": manufacturer,
                    "Model": model_name,
                    "Country": country,
                    "Age": age,
                    "Downtime": downtime,
                    "Maintenance_Cost": maintenance_cost,
                    "Maintenance_Class": 1,  # Placeholder, will be predicted
                    "Maintenance_Frequency": maintenance_frequency,
                    "Failure_Event_Count": failure_event_count
                }])

                # Encode categorical features
                for col, le in label_encoders.items():
                    if col in input_struct.columns:
                        try:
                            input_struct[col] = le.transform(input_struct[col].astype(str))
                        except ValueError:
                            # Handle unseen categories
                            input_struct[col] = 0

                # ──────────────────────────────
                # Prepare NLP Features
                # ──────────────────────────────
                if report_text.strip():
                    tfidf_vec = tfidf.transform([report_text.lower()])
                    rpt_svd = svd.transform(tfidf_vec)
                else:
                    # Use average from similar devices if no report provided
                    mask = (df["Model"] == model_name) | (df["Manufacturer"] == manufacturer)
                    if mask.sum() > 0:
                        similar_reports = df.loc[mask, "Maintenance_Report"].fillna("").astype(str)
                        if len(similar_reports) > 0:
                            t = tfidf.transform(similar_reports)
                            rpt_svd = svd.transform(t).mean(axis=0).reshape(1, -1)
                        else:
                            rpt_svd = np.zeros((1, svd.n_components))
                    else:
                        rpt_svd = np.zeros((1, svd.n_components))

                # ──────────────────────────────
                # Combine Features & Handle Shape Mismatches
                # ──────────────────────────────
                X_all = np.hstack([input_struct.values, rpt_svd])
                expected_scaler = scaler.n_features_in_
                
                # Trim or pad to match scaler expectations
                if X_all.shape[1] > expected_scaler:
                    X_all = X_all[:, :expected_scaler]
                elif X_all.shape[1] < expected_scaler:
                    padding = np.zeros((1, expected_scaler - X_all.shape[1]))
                    X_all = np.hstack([X_all, padding])

                # Scale features
                X_scaled = scaler.transform(X_all)
                
                # ──────────────────────────────
                # LSTM Encoding
                # ──────────────────────────────
                X_lstm = X_scaled.reshape(1, 1, X_scaled.shape[1])
                encoded = encoder_model.predict(X_lstm, verbose=0)
                encoded_flat = encoded.reshape(1, -1)

                # ──────────────────────────────
                # Fix Shape Mismatch for XGBoost Models
                # ──────────────────────────────
                expected_dim_maint = xgb_maint.n_features_in_
                expected_dim_fail = xgb_fail.n_features_in_
                
                # Adjust for maintenance class model
                if encoded_flat.shape[1] < expected_dim_maint:
                    padding = np.zeros((1, expected_dim_maint - encoded_flat.shape[1]))
                    encoded_flat_maint = np.hstack([encoded_flat, padding])
                elif encoded_flat.shape[1] > expected_dim_maint:
                    encoded_flat_maint = encoded_flat[:, :expected_dim_maint]
                else:
                    encoded_flat_maint = encoded_flat
                
                # Adjust for failure type model
                if encoded_flat.shape[1] < expected_dim_fail:
                    padding = np.zeros((1, expected_dim_fail - encoded_flat.shape[1]))
                    encoded_flat_fail = np.hstack([encoded_flat, padding])
                elif encoded_flat.shape[1] > expected_dim_fail:
                    encoded_flat_fail = encoded_flat[:, :expected_dim_fail]
                else:
                    encoded_flat_fail = encoded_flat

                # ──────────────────────────────
                # Predictions
                # ──────────────────────────────
                pred_maint0 = xgb_maint.predict(encoded_flat_maint)[0]
                pred_maint = int(pred_maint0) + 1
                prob_maint = xgb_maint.predict_proba(encoded_flat_maint)[0]
                confidence = float(np.max(prob_maint)) * 100  # Cast to float to avoid float32 issues

                prob_fail = xgb_fail.predict_proba(encoded_flat_fail)[0]
                
                # ──────────────────────────────
                # Get Failure Type Keywords
                # ──────────────────────────────
                top_fail_type = int(np.argmax(prob_fail))
                failure_keywords = cluster_keywords.get(top_fail_type, [])
                if not failure_keywords:
                    failure_keywords = cluster_keywords.get(0, [])

                # ──────────────────────────────
                # Display Results
                # ──────────────────────────────
                st.header("🧠 Prediction Results")
                
                # Risk Indicator Section
                st.subheader("⚠️ Risk Assessment")
                risk_col1, risk_col2 = st.columns([1, 2])
                
                with risk_col1:
                    if pred_maint == 3:
                        st.markdown("### 🔴 High Risk")
                        st.markdown(f"**Maintenance Class:** {pred_maint}")
                    elif pred_maint == 2:
                        st.markdown("### 🟠 Moderate Risk")
                        st.markdown(f"**Maintenance Class:** {pred_maint}")
                    else:
                        st.markdown("### 🟢 Low Risk")
                        st.markdown(f"**Maintenance Class:** {pred_maint}")
                
                with risk_col2:
                    st.markdown("**Confidence Level:**")
                    # Cast to float to avoid float32 issues with progress bar
                    progress_value = float(confidence) / 100.0
                    st.progress(progress_value)
                    st.caption(f"{confidence:.2f}% confidence")

                st.markdown("---")

                # ──────────────────────────────
                # Failure Type Probabilities Chart
                # ──────────────────────────────
                col_chart, col_keywords = st.columns([2, 1])
                
                with col_chart:
                    st.subheader("📊 Failure Type Probabilities")
                    # Create bar chart with Plotly
                    fail_types = [f"Type {i}" for i in range(len(prob_fail))]
                    fig = go.Figure(data=[
                        go.Bar(
                            x=fail_types,
                            y=prob_fail,
                            marker_color=px.colors.sequential.Reds_r[:len(prob_fail)],
                            text=[f"{p*100:.2f}%" for p in prob_fail],
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(
                        title="Predicted Failure Type Probabilities",
                        xaxis_title="Failure Type",
                        yaxis_title="Probability",
                        height=400,
                        showlegend=False,
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_keywords:
                    st.subheader("🔍 Likely Failure Reasons")
                    st.markdown(f"**Top Predicted Type: {top_fail_type}** ({prob_fail[top_fail_type]*100:.2f}%)")
                    st.markdown("**Keywords:**")
                    if failure_keywords:
                        for keyword in failure_keywords[:10]:  # Show top 10 keywords
                            st.markdown(f"- {keyword}")
                    else:
                        st.info("No keywords available for this failure type.")
                    
                    # Show keywords for all failure types
                    st.markdown("---")
                    st.markdown("**All Failure Types:**")
                    for i, prob in enumerate(prob_fail):
                        if prob > 0.01:  # Only show if probability > 1%
                            keywords = cluster_keywords.get(i, [])
                            if keywords:
                                st.markdown(f"**Type {i}** ({prob*100:.2f}%):")
                                st.caption(", ".join(keywords[:5]))

                st.markdown("---")

                # ──────────────────────────────
                # Similar Maintenance Reports
                # ──────────────────────────────
                st.subheader("🧾 Similar Past Maintenance Reports")
                
                # Find similar reports using cosine similarity on TF-IDF vectors
                if report_text.strip():
                    # Use the input report text
                    query_vec = tfidf.transform([report_text.lower()])
                    all_reports = df["Maintenance_Report"].fillna("").astype(str)
                    all_vecs = tfidf.transform(all_reports)
                    similarities = cosine_similarity(query_vec, all_vecs)[0]
                    top_indices = np.argsort(similarities)[::-1][:5]
                    similar_reports = df.iloc[top_indices]["Maintenance_Report"].tolist()
                else:
                    # Use device/model matching
                    mask = (df["Model"] == model_name) | (df["Manufacturer"] == manufacturer)
                    if mask.sum() > 0:
                        similar_reports = df.loc[mask, "Maintenance_Report"].dropna().head(5).tolist()
                    else:
                        # Fallback to random reports
                        similar_reports = df["Maintenance_Report"].dropna().head(5).tolist()
                
                if similar_reports:
                    for idx, report in enumerate(similar_reports, 1):
                        with st.expander(f"Report {idx}", expanded=(idx == 1)):
                            st.write(report)
                else:
                    st.info("No similar maintenance reports found in the dataset.")

            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)

# ──────────────────────────────
# TAB 2: Analytics
# ──────────────────────────────
with tab2:
    st.subheader("📊 Equipment Maintenance Analytics")
    
    # Maintenance Class Distribution
    fig1 = px.histogram(
        df, 
        x="Maintenance_Class", 
        color="Maintenance_Class",
        title="Distribution of Maintenance Classes",
        nbins=10,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)
    
    # Downtime vs Cost
    fig2 = px.scatter(
        df, 
        x="Downtime", 
        y="Maintenance_Cost", 
        color="Maintenance_Class",
        size="Failure_Event_Count", 
        hover_data=["Device_Type", "Manufacturer", "Model"],
        title="Downtime vs Maintenance Cost by Class",
        color_discrete_sequence=['green', 'orange', 'red']
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Device Type Analysis
    col_anal1, col_anal2 = st.columns(2)
    
    with col_anal1:
        device_type_counts = df["Device_Type"].value_counts()
        fig3 = px.pie(
            values=device_type_counts.values,
            names=device_type_counts.index,
            title="Device Type Distribution"
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    with col_anal2:
        manufacturer_counts = df["Manufacturer"].value_counts().head(10)
        fig4 = px.bar(
            x=manufacturer_counts.index,
            y=manufacturer_counts.values,
            title="Top 10 Manufacturers by Device Count",
            labels={"x": "Manufacturer", "y": "Count"}
        )
        fig4.update_xaxes(tickangle=45)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Heatmap
    st.subheader("Failure Frequency Heatmap")
    heatmap_df = df.pivot_table(
        index="Manufacturer", 
        columns="Device_Type",
        values="Failure_Event_Count", 
        aggfunc="mean"
    ).fillna(0)
    
    if not heatmap_df.empty:
        fig5 = px.imshow(
            heatmap_df, 
            color_continuous_scale="Reds",
            title="Average Failure Frequency by Manufacturer & Device Type",
            labels=dict(x="Device Type", y="Manufacturer", color="Avg Failures")
        )
        st.plotly_chart(fig5, use_container_width=True)
    
    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("Total Devices", len(df))
    with stat_col2:
        st.metric("Avg Downtime (hrs)", f"{df['Downtime'].mean():.2f}")
    with stat_col3:
        st.metric("Avg Cost ($)", f"{df['Maintenance_Cost'].mean():.2f}")
    with stat_col4:
        st.metric("Avg Failures", f"{df['Failure_Event_Count'].mean():.2f}")

# ──────────────────────────────
# TAB 3: Failure Insights
# ──────────────────────────────
with tab3:
    st.subheader("💬 NLP-Based Failure Insights")
    
    # Word Cloud (if wordcloud is available, otherwise show text analysis)
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        text = " ".join(df["Maintenance_Report"].dropna().astype(str))
        if text.strip():
            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color='black', 
                colormap='Reds',
                max_words=100
            ).generate(text)
            
            fig_wc, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("Most Frequent Terms in Maintenance Reports", fontsize=16, pad=20)
            st.pyplot(fig_wc)
        else:
            st.info("No maintenance reports available for word cloud generation.")
    except ImportError:
        st.info("WordCloud library not available. Install with: `pip install wordcloud`")
        
        # Alternative: Show top keywords from reports
        st.subheader("Top Keywords from Maintenance Reports")
        all_text = " ".join(df["Maintenance_Report"].dropna().astype(str).str.lower())
        words = all_text.split()
        from collections import Counter
        word_freq = Counter(words)
        top_words = word_freq.most_common(30)
        
        if top_words:
            words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
            fig_words = px.bar(
                words_df.head(20),
                x='Frequency',
                y='Word',
                orientation='h',
                title="Top 20 Most Frequent Words in Maintenance Reports"
            )
            st.plotly_chart(fig_words, use_container_width=True)
    
    # Failure Type Analysis
    st.subheader("Failure Type Cluster Analysis")
    
    if cluster_keywords:
        cluster_col1, cluster_col2 = st.columns(2)
        
        with cluster_col1:
            st.markdown("### Failure Type Keywords")
            for cluster_id, keywords in list(cluster_keywords.items())[:5]:
                if keywords:
                    with st.expander(f"Cluster {cluster_id} - {len(keywords)} keywords"):
                        st.write(", ".join(keywords[:20]))
        
        with cluster_col2:
            # Show distribution of maintenance classes
            maint_class_dist = df["Maintenance_Class"].value_counts().sort_index()
            fig_class = px.bar(
                x=maint_class_dist.index,
                y=maint_class_dist.values,
                title="Maintenance Class Distribution",
                labels={"x": "Maintenance Class", "y": "Count"},
                color=maint_class_dist.index,
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_class, use_container_width=True)

# ──────────────────────────────
# TAB 4: Sustainability Metrics
# ──────────────────────────────
with tab4:
    st.subheader("🌱 Operational & Sustainability Indicators")
    
    # Key Metrics
    avg_downtime = df["Downtime"].mean()
    avg_cost = df["Maintenance_Cost"].mean()
    avg_failures = df["Failure_Event_Count"].mean()
    total_devices = len(df)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Average Downtime (hrs)", f"{avg_downtime:.1f}")
    with metric_col2:
        st.metric("Avg. Maintenance Cost ($)", f"{avg_cost:.2f}")
    with metric_col3:
        st.metric("Avg. Failures per Device", f"{avg_failures:.2f}")
    with metric_col4:
        st.metric("Total Devices", total_devices)
    
    st.markdown("---")
    
    # Sustainability Progress
    st.subheader("Operational Efficiency Metrics")
    
    # Calculate efficiency score (lower downtime and failures = better)
    max_downtime = df["Downtime"].max()
    max_failures = df["Failure_Event_Count"].max()
    efficiency_score = 100 - ((avg_downtime / max_downtime * 50) + (avg_failures / max_failures * 50))
    efficiency_score = max(0, min(100, efficiency_score))
    
    st.markdown("**Overall Equipment Efficiency Score:**")
    progress_value = float(efficiency_score) / 100.0
    st.progress(progress_value)
    st.caption(f"{efficiency_score:.1f}% - Reducing downtime and failures improves sustainability and cost efficiency in healthcare equipment usage.")
    
    # Cost Analysis
    st.subheader("Cost Analysis")
    cost_col1, cost_col2 = st.columns(2)
    
    with cost_col1:
        cost_by_class = df.groupby("Maintenance_Class")["Maintenance_Cost"].mean()
        fig_cost = px.bar(
            x=cost_by_class.index,
            y=cost_by_class.values,
            title="Average Maintenance Cost by Class",
            labels={"x": "Maintenance Class", "y": "Average Cost ($)"},
            color=cost_by_class.index,
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig_cost, use_container_width=True)
    
    with cost_col2:
        cost_by_device = df.groupby("Device_Type")["Maintenance_Cost"].mean().sort_values(ascending=False)
        fig_cost_device = px.bar(
            x=cost_by_device.values,
            y=cost_by_device.index,
            orientation='h',
            title="Average Cost by Device Type",
            labels={"x": "Average Cost ($)", "y": "Device Type"}
        )
        st.plotly_chart(fig_cost_device, use_container_width=True)
    
    # Age vs Performance
    st.subheader("Device Age vs Performance")
    age_perf = df.groupby("Age").agg({
        "Downtime": "mean",
        "Failure_Event_Count": "mean",
        "Maintenance_Cost": "mean"
    }).reset_index()
    
    fig_age = px.scatter(
        age_perf,
        x="Age",
        y="Downtime",
        size="Failure_Event_Count",
        color="Maintenance_Cost",
        hover_data=["Failure_Event_Count"],
        title="Device Age Impact on Downtime and Costs",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_age, use_container_width=True)
    
    # Recommendations
    st.subheader("💡 Sustainability Recommendations")
    st.info("""
    **Key Recommendations:**
    - **Proactive Maintenance**: Devices with higher failure rates should have increased maintenance frequency
    - **Cost Optimization**: Focus on preventive maintenance for high-cost device types
    - **Downtime Reduction**: Implement predictive maintenance schedules to minimize operational disruptions
    - **Resource Planning**: Allocate maintenance resources based on device age and historical failure patterns
    """)

# ──────────────────────────────
# Footer
# ──────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>Hybrid Explainable Predictive Maintenance — Healthcare Equipment | Built by Anieruth Sridhar</small>
</div>
""", unsafe_allow_html=True)
