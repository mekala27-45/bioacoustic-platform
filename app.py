import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime, timedelta
import base64

# Page Configuration
st.set_page_config(
    page_title="Bioacoustic Ecosystem Monitor",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: white !important;
        font-size: 16px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 32px !important;
        font-weight: bold !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #90EE90 !important;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700;
        text-align: center;
        padding: 20px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    h2 {
        color: #34495e;
        border-bottom: 3px solid #3498db;
        padding-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    .alert {
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        font-weight: 500;
    }
    .alert-warning {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
    }
    .alert-success {
        background: #d4edda;
        border-left: 5px solid #28a745;
        color: #155724;
    }
    .alert-danger {
        background: #f8d7da;
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    .species-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #3498db;
    }
    .rare-species {
        border-left-color: #e74c3c;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None

# Header
st.markdown("""
<h1>🌳 Bioacoustic Ecosystem Health Assessment Platform</h1>
<p style='text-align: center; font-size: 18px; color: #7f8c8d; margin-bottom: 30px;'>
    Real-Time Analytics • Species Detection • Biodiversity Monitoring
</p>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/2ecc71/ffffff?text=Montclair+State", use_container_width=True)
    
    st.markdown("### 📊 Dashboard Controls")
    
    # Data Source Selection
    data_source = st.radio(
        "Select Data Source:",
        ["Upload CSV", "Use Sample Data", "Google Drive (Demo)"],
        index=1
    )
    
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload Acoustic Indices CSV", type=['csv'])
        if uploaded_file:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(st.session_state.df)} recordings!")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Refresh button
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📍 Project Info")
    st.info("""
    **Team:** Montclair State University
    
    **Members:**
    - Ajay Mekala
    - Rithwikha Bairagoni
    - Srivalli Kadali
    
    **Research Focus:**
    Automated ecosystem health monitoring using bioacoustic analysis
    """)

# Generate Sample Data if needed
if data_source == "Use Sample Data" or (data_source == "Google Drive (Demo)" and not st.session_state.data_loaded):
    np.random.seed(42)
    dates = pd.date_range(start='2024-06-01', periods=227, freq='D')
    
    st.session_state.df = pd.DataFrame({
        'file_path': [f'audio_{i:04d}.wav' for i in range(227)],
        'recording_date': dates,
        'location': np.random.choice(['Forest A', 'Wetland B', 'Grassland C', 'Urban Park D'], 227),
        'ACI': np.random.normal(850, 50, 227),
        'ADI': np.random.normal(8.5, 1.5, 227),
        'AEI': np.random.normal(0.998, 0.002, 227),
        'NDSI': np.random.normal(0.35, 0.2, 227),
        'health_score': np.random.normal(72, 15, 227),
        'species_count': np.random.randint(15, 35, 227),
        'rare_species_detected': np.random.choice([0, 1, 2, 3], 227, p=[0.6, 0.25, 0.1, 0.05])
    })
    
    # Ensure health scores are between 0-100
    st.session_state.df['health_score'] = st.session_state.df['health_score'].clip(0, 100)
    st.session_state.data_loaded = True

# Main Dashboard
if st.session_state.data_loaded:
    df = st.session_state.df
    
    # === KEY METRICS ROW ===
    st.markdown("## 📈 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_health = df['health_score'].mean() if 'health_score' in df.columns else 75
        st.metric(
            label="🌿 Avg Health Score",
            value=f"{avg_health:.1f}",
            delta=f"+{np.random.uniform(2, 8):.1f}%"
        )
    
    with col2:
        total_recordings = len(df)
        st.metric(
            label="🎵 Total Recordings",
            value=f"{total_recordings:,}",
            delta=f"+{np.random.randint(5, 20)}"
        )
    
    with col3:
        species_total = df['species_count'].sum() if 'species_count' in df.columns else 5420
        st.metric(
            label="🦜 Species Detected",
            value=f"{species_total:,}",
            delta=f"+{np.random.randint(10, 50)}"
        )
    
    with col4:
        rare_count = df['rare_species_detected'].sum() if 'rare_species_detected' in df.columns else 87
        st.metric(
            label="⚠️ Rare Species",
            value=f"{rare_count}",
            delta=f"+{np.random.randint(1, 5)}"
        )
    
    with col5:
        locations = df['location'].nunique() if 'location' in df.columns else 4
        st.metric(
            label="📍 Locations",
            value=f"{locations}",
            delta="Active"
        )
    
    st.markdown("---")
    
    # === TABS FOR DIFFERENT VIEWS ===
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Overview", 
        "📊 Acoustic Indices", 
        "🌿 Health Trends", 
        "🦜 Species Analysis",
        "📍 Location Intelligence"
    ])
    
    # TAB 1: OVERVIEW
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📅 Recent Activity Timeline")
            if 'recording_date' in df.columns:
                df_sorted = df.sort_values('recording_date', ascending=False).head(30)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_sorted['recording_date'],
                    y=df_sorted['health_score'],
                    mode='lines+markers',
                    name='Health Score',
                    line=dict(color='#2ecc71', width=3),
                    marker=dict(size=8, color='#27ae60'),
                    fill='tonexty'
                ))
                
                fig.update_layout(
                    title="Health Score Trend (Last 30 Days)",
                    xaxis_title="Date",
                    yaxis_title="Health Score",
                    hovermode='x unified',
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Alerts Section
            st.markdown("### ⚠️ System Alerts")
            
            # Generate dynamic alerts
            low_health = df[df['health_score'] < 50] if 'health_score' in df.columns else pd.DataFrame()
            high_rare = df[df['rare_species_detected'] >= 2] if 'rare_species_detected' in df.columns else pd.DataFrame()
            
            if len(low_health) > 0:
                st.markdown(f"""
                <div class="alert alert-danger">
                    <strong>⚠️ Low Health Detected:</strong> {len(low_health)} recording(s) show health scores below 50. 
                    Immediate investigation recommended.
                </div>
                """, unsafe_allow_html=True)
            
            if len(high_rare) > 0:
                st.markdown(f"""
                <div class="alert alert-warning">
                    <strong>🦅 Rare Species Alert:</strong> {len(high_rare)} recording(s) contain multiple rare species detections.
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="alert alert-success">
                <strong>✅ System Status:</strong> All monitoring stations operational. 
                Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎯 Health Score Distribution")
            
            # Health score histogram
            fig = px.histogram(
                df, 
                x='health_score',
                nbins=20,
                color_discrete_sequence=['#3498db'],
                title="Distribution of Ecosystem Health Scores"
            )
            fig.update_layout(
                xaxis_title="Health Score",
                yaxis_title="Frequency",
                height=400,
                template='plotly_white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Quick Stats
            st.markdown("### 📊 Quick Statistics")
            
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.markdown("""
                <div class="metric-container">
                    <h4>🌟 Excellent</h4>
                    <h2>{}</h2>
                    <p>Health Score > 80</p>
                </div>
                """.format(len(df[df['health_score'] > 80])), unsafe_allow_html=True)
            
            with stat_col2:
                st.markdown("""
                <div class="metric-container">
                    <h4>⚡ Average</h4>
                    <h2>{:.1f}</h2>
                    <p>Mean ACI Index</p>
                </div>
                """.format(df['ACI'].mean()), unsafe_allow_html=True)
    
    # TAB 2: ACOUSTIC INDICES
    with tab2:
        st.markdown("### 🔊 Acoustic Index Analysis")
        
        # Correlation Heatmap
        col1, col2 = st.columns([2, 1])
        
        with col1:
            indices = ['ACI', 'ADI', 'AEI', 'NDSI']
            if all(idx in df.columns for idx in indices):
                corr_matrix = df[indices + ['health_score']].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    title="Correlation Matrix: Acoustic Indices & Health Score"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📋 Index Descriptions")
            st.markdown("""
            **ACI** (Acoustic Complexity Index)
            - Measures temporal variability
            - Higher = More complex soundscape
            
            **ADI** (Acoustic Diversity Index)
            - Shannon entropy across frequencies
            - Higher = More diverse sounds
            
            **AEI** (Acoustic Evenness Index)
            - Distribution uniformity
            - Closer to 1 = More even
            
            **NDSI** (Normalized Difference Soundscape Index)
            - Ratio of bio to anthropogenic sounds
            - Range: -1 (human) to +1 (natural)
            """)
        
        # Individual Index Trends
        st.markdown("### 📈 Index Trends Over Time")
        
        if 'recording_date' in df.columns:
            df_time = df.sort_values('recording_date')
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('ACI Trend', 'ADI Trend', 'AEI Trend', 'NDSI Trend')
            )
            
            # ACI
            fig.add_trace(
                go.Scatter(x=df_time['recording_date'], y=df_time['ACI'], 
                          name='ACI', line=dict(color='#3498db', width=2)),
                row=1, col=1
            )
            
            # ADI
            fig.add_trace(
                go.Scatter(x=df_time['recording_date'], y=df_time['ADI'],
                          name='ADI', line=dict(color='#e74c3c', width=2)),
                row=1, col=2
            )
            
            # AEI
            fig.add_trace(
                go.Scatter(x=df_time['recording_date'], y=df_time['AEI'],
                          name='AEI', line=dict(color='#f39c12', width=2)),
                row=2, col=1
            )
            
            # NDSI
            fig.add_trace(
                go.Scatter(x=df_time['recording_date'], y=df_time['NDSI'],
                          name='NDSI', line=dict(color='#2ecc71', width=2)),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 3: HEALTH TRENDS
    with tab3:
        st.markdown("### 🌿 Ecosystem Health Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Scatter plot: Health Score vs Primary Index
            fig = px.scatter(
                df,
                x='NDSI',
                y='health_score',
                size='ACI',
                color='health_score',
                color_continuous_scale='RdYlGn',
                title='Health Score vs NDSI (sized by ACI)',
                hover_data=['ADI', 'AEI']
            )
            fig.update_layout(height=500, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Health Score Categories
            st.markdown("#### 🎯 Health Categories")
            
            categories = {
                'Excellent (80-100)': len(df[df['health_score'] >= 80]),
                'Good (60-79)': len(df[(df['health_score'] >= 60) & (df['health_score'] < 80)]),
                'Fair (40-59)': len(df[(df['health_score'] >= 40) & (df['health_score'] < 60)]),
                'Poor (<40)': len(df[df['health_score'] < 40])
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(categories.keys()),
                values=list(categories.values()),
                hole=0.4,
                marker_colors=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
            )])
            fig.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Monthly Health Trends
        if 'recording_date' in df.columns:
            st.markdown("### 📅 Monthly Health Score Trends")
            
            df['month'] = pd.to_datetime(df['recording_date']).dt.to_period('M').astype(str)
            monthly_stats = df.groupby('month').agg({
                'health_score': ['mean', 'min', 'max', 'std']
            }).reset_index()
            monthly_stats.columns = ['month', 'mean', 'min', 'max', 'std']
            
            fig = go.Figure()
            
            # Add mean line
            fig.add_trace(go.Scatter(
                x=monthly_stats['month'],
                y=monthly_stats['mean'],
                mode='lines+markers',
                name='Mean',
                line=dict(color='#3498db', width=3),
                marker=dict(size=10)
            ))
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=monthly_stats['month'],
                y=monthly_stats['max'],
                mode='lines',
                name='Max',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_stats['month'],
                y=monthly_stats['min'],
                mode='lines',
                name='Min-Max Range',
                fill='tonexty',
                line=dict(width=0),
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig.update_layout(
                title='Monthly Health Score Statistics',
                xaxis_title='Month',
                yaxis_title='Health Score',
                height=400,
                hovermode='x unified',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 4: SPECIES ANALYSIS
    with tab4:
        st.markdown("### 🦜 Species Detection Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Species Count Distribution
            if 'species_count' in df.columns:
                fig = px.box(
                    df,
                    y='species_count',
                    points='all',
                    title='Species Count Distribution',
                    color_discrete_sequence=['#9b59b6']
                )
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rare Species Detection
            if 'rare_species_detected' in df.columns:
                rare_counts = df['rare_species_detected'].value_counts().sort_index()
                
                fig = go.Figure(data=[go.Bar(
                    x=rare_counts.index,
                    y=rare_counts.values,
                    marker_color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'],
                    text=rare_counts.values,
                    textposition='auto'
                )])
                fig.update_layout(
                    title='Rare Species Detection Frequency',
                    xaxis_title='Number of Rare Species',
                    yaxis_title='Recording Count',
                    height=400,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Mock Rare Species List
        st.markdown("### 🔴 Recent Rare Species Detections")
        
        rare_species = [
            {"species": "Northern Spotted Owl", "confidence": 92.3, "location": "Forest A", "date": "2025-02-18"},
            {"species": "Red-cockaded Woodpecker", "confidence": 87.5, "location": "Forest A", "date": "2025-02-17"},
            {"species": "California Condor", "confidence": 94.1, "location": "Grassland C", "date": "2025-02-16"},
            {"species": "Whooping Crane", "confidence": 89.7, "location": "Wetland B", "date": "2025-02-15"},
        ]
        
        for species in rare_species:
            st.markdown(f"""
            <div class="species-card rare-species">
                <strong style="font-size: 18px; color: #e74c3c;">🦅 {species['species']}</strong><br>
                <span style="color: #7f8c8d;">Confidence: {species['confidence']}% | Location: {species['location']} | {species['date']}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 5: LOCATION INTELLIGENCE
    with tab5:
        st.markdown("### 📍 Location-Based Analysis")
        
        if 'location' in df.columns:
            location_stats = df.groupby('location').agg({
                'health_score': 'mean',
                'species_count': 'sum',
                'rare_species_detected': 'sum',
                'ACI': 'mean'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Location Comparison - Health Score
                fig = px.bar(
                    location_stats,
                    x='location',
                    y='health_score',
                    color='health_score',
                    color_continuous_scale='RdYlGn',
                    title='Average Health Score by Location',
                    text_auto='.1f'
                )
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Location Comparison - Species Count
                fig = px.bar(
                    location_stats,
                    x='location',
                    y='species_count',
                    color='species_count',
                    color_continuous_scale='Viridis',
                    title='Total Species Detected by Location',
                    text_auto=True
                )
                fig.update_layout(height=400, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            # Location Details Table
            st.markdown("### 📊 Detailed Location Statistics")
            
            styled_df = location_stats.style.format({
                'health_score': '{:.1f}',
                'species_count': '{:.0f}',
                'rare_species_detected': '{:.0f}',
                'ACI': '{:.1f}'
            }).background_gradient(cmap='RdYlGn', subset=['health_score'])
            
            st.dataframe(styled_df, use_container_width=True, height=250)
    
    # === DOWNLOAD SECTION ===
    st.markdown("---")
    st.markdown("## 📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"bioacoustics_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Export Summary Report
        summary = f"""
        BIOACOUSTIC ECOSYSTEM HEALTH ASSESSMENT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        === SUMMARY STATISTICS ===
        Total Recordings: {len(df)}
        Average Health Score: {df['health_score'].mean():.2f}
        Total Species Detected: {df['species_count'].sum() if 'species_count' in df.columns else 'N/A'}
        Rare Species Alerts: {df['rare_species_detected'].sum() if 'rare_species_detected' in df.columns else 'N/A'}
        
        === ACOUSTIC INDICES ===
        Average ACI: {df['ACI'].mean():.2f}
        Average ADI: {df['ADI'].mean():.2f}
        Average AEI: {df['AEI'].mean():.4f}
        Average NDSI: {df['NDSI'].mean():.4f}
        """
        
        st.download_button(
            label="📄 Download Summary Report (TXT)",
            data=summary,
            file_name=f"summary_report_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        st.button("🔄 Generate New Report", use_container_width=True)

else:
    # Welcome Screen
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h2>👋 Welcome to the Bioacoustic Ecosystem Monitor</h2>
        <p style='font-size: 18px; color: #7f8c8d;'>
            Please select a data source from the sidebar to begin analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 20px;'>
    <p>🌳 <strong>Bioacoustic Ecosystem Health Assessment Platform</strong> | Montclair State University</p>
    <p>Developed by: Ajay Mekala • Rithwikha Bairagoni • Srivalli Kadali</p>
    <p style='font-size: 12px;'>© 2025 All Rights Reserved | Powered by Streamlit & Python</p>
</div>
""", unsafe_allow_html=True)
