# 🌳 Bioacoustic Ecosystem Health Assessment Platform

A professional, production-ready web application for real-time bioacoustic analysis, ecosystem health monitoring, and rare species detection.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🚀 Features

### Real-Time Analytics Dashboard
- **Live Health Score Monitoring** - Track ecosystem health scores (0-100 scale)
- **Acoustic Indices Visualization** - ACI, ADI, AEI, NDSI analysis
- **Species Detection** - Real-time rare species alerts
- **Location Intelligence** - Geographic comparison and trends
- **Interactive Charts** - Powered by Plotly for dynamic visualizations

### Key Capabilities
✅ Upload CSV data or connect to Google Drive
✅ 5 comprehensive dashboard tabs
✅ Real-time data refresh
✅ Export reports (CSV, TXT)
✅ Responsive design for mobile/tablet
✅ Professional gradient UI with custom styling

## 📋 Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Google Drive account (for Drive integration)

## 🛠️ Installation

### Option 1: Local Setup

1. **Clone or Download the Repository**
```bash
# Create a new directory
mkdir bioacoustic-platform
cd bioacoustic-platform

# Copy app.py and requirements.txt to this directory
```

2. **Create Virtual Environment**
```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the Application**
```bash
streamlit run app.py
```

5. **Open in Browser**
The app will automatically open at `http://localhost:8501`

### Option 2: Deploy to Streamlit Cloud (FREE)

1. **Push to GitHub**
```bash
git init
git add app.py requirements.txt
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. **Deploy on Streamlit Cloud**
- Go to https://share.streamlit.io/
- Sign in with GitHub
- Click "New app"
- Select your repository
- Set main file: `app.py`
- Click "Deploy"

Your app will be live at: `https://share.streamlit.io/<username>/<repo>/app.py`

## 📁 Project Structure

```
bioacoustic-platform/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore file
└── data/                  # (Optional) Sample data directory
    └── acoustic_indices.csv
```

## 📊 Data Format

The application expects a CSV file with the following columns:

```csv
file_path,recording_date,location,ACI,ADI,AEI,NDSI,health_score,species_count,rare_species_detected
audio_0001.wav,2024-06-01,Forest A,850.5,8.2,0.998,0.35,72.5,25,1
audio_0002.wav,2024-06-02,Wetland B,842.3,7.8,0.997,0.42,78.2,28,0
...
```

### Column Descriptions:
- **file_path**: Audio file name or path
- **recording_date**: Date of recording (YYYY-MM-DD)
- **location**: Recording location name
- **ACI**: Acoustic Complexity Index
- **ADI**: Acoustic Diversity Index
- **AEI**: Acoustic Evenness Index
- **NDSI**: Normalized Difference Soundscape Index
- **health_score**: Ecosystem health score (0-100)
- **species_count**: Number of species detected
- **rare_species_detected**: Count of rare species (0, 1, 2, 3+)

## 🔐 Google Drive Integration (Optional)

To connect to Google Drive:

1. **Enable Google Drive API**
   - Go to https://console.cloud.google.com/
   - Create a new project
   - Enable Google Drive API
   - Create OAuth 2.0 credentials
   - Download `credentials.json`

2. **Add Credentials**
   - Place `credentials.json` in the project root
   - First run will open browser for authentication

3. **Update app.py** (if needed)
```python
# Add at the top of app.py
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# Add authentication function
def authenticate_gdrive():
    # Authentication code here
    pass
```

## 📈 Using the Application

### 1. Dashboard Overview
- View key metrics: Health Score, Total Recordings, Species Count, Rare Species
- See recent activity timeline
- Monitor system alerts

### 2. Acoustic Indices Tab
- Analyze correlation heatmap
- View individual index trends
- Understand acoustic patterns

### 3. Health Trends Tab
- Scatter plots: Health Score vs NDSI
- Health category pie chart (Excellent, Good, Fair, Poor)
- Monthly statistics and trends

### 4. Species Analysis Tab
- Species count distribution
- Rare species detection frequency
- Recent rare species alerts

### 5. Location Intelligence Tab
- Compare health scores across locations
- Analyze species diversity by area
- Export location-specific reports

## 🎨 Customization

### Change Color Scheme
Edit the CSS in `app.py` around line 30:
```python
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #your-color1, #your-color2);
    }
    ...
</style>
""", unsafe_allow_html=True)
```

### Add New Charts
Use Plotly Express or Plotly Graph Objects:
```python
import plotly.express as px

fig = px.scatter(df, x='ACI', y='health_score')
st.plotly_chart(fig, use_container_width=True)
```

## 🚨 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Module Not Found
```bash
pip install -r requirements.txt --upgrade
```

### Data Not Loading
- Check CSV format matches expected columns
- Verify file encoding is UTF-8
- Ensure no missing values in critical columns

## 📝 Sample Data

The app includes built-in sample data. To use your own:

1. Prepare CSV file matching the format above
2. Select "Upload CSV" in sidebar
3. Click "Browse files" and select your CSV
4. Data will load automatically

## 🔒 Security Notes

- Never commit `credentials.json` or API keys to Git
- Add sensitive files to `.gitignore`
- Use environment variables for production secrets
- Enable authentication for production deployments

## 📞 Support & Contact

**Development Team:**
- Ajay Mekala - Data Science Lead
- Rithwikha Bairagoni - Ecosystem Analytics
- Srivalli Kadali - Data Engineering

**Institution:** Montclair State University

**Course:** Research Methods in Computing

## 🎓 Academic Use

This project is part of academic research on bioacoustic ecosystem monitoring. 

**Citation:**
```
Mekala, A., Bairagoni, R., Kadali, S. (2025). 
Bioacoustic Ecosystem Health Assessment Platform. 
Montclair State University.
```

## 📄 License

MIT License - See LICENSE file for details

## 🚀 Future Enhancements

- [ ] Real-time audio upload and processing
- [ ] ML model inference integration
- [ ] WebSocket for live streaming data
- [ ] Email alerts for critical events
- [ ] Mobile app companion
- [ ] API endpoints for external systems
- [ ] Advanced filtering and search
- [ ] Historical trend predictions

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Plotly for interactive visualizations
- Montclair State University for research support

---

**Built with ❤️ using Python & Streamlit**

For issues or questions, please open an issue on GitHub or contact the development team.
