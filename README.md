# EchoPulse 🎥

**Understand Any Video — Just by reading the comments.**

EchoPulse is an interactive Streamlit app that analyzes YouTube comments to generate:
- **Educational summaries** of video content  
- **Sentiment analysis** and **toxicity filtering**  
- **Topic clustering** of viewer discussions  
- **Automated watch/skip recommendations**  

## 🚀 Features

- **Comment Extraction**  
  Fetches comments directly from YouTube using `yt-dlp` and `youtube-comment-downloader`.  
- **Toxicity Filtering**  
  Combines ML-based detection (`unitary/toxic-bert`) with keyword heuristics.  
- **Sentiment Analysis**  
  Leverages `pysentimiento` for fine-grained sentiment scoring.  
- **Topic Modeling**  
  Uses BERTopic to cluster comments into thematic topics.  
- **Multi-Model Summarization**  
  Supports Mistral-7B, FLAN-T5, DistilBART, and BART-Large fallbacks.  
- **Interactive Dashboard**  
  Built with Streamlit: dynamic metrics, charts, and expandable topic views.

## 📦 Prerequisites

- **Python ≥ 3.8**  
- **Git**  
- **At least 14–18 GB of free disk space** (for model weights and offload folders)  
- **Windows users** may need to adjust virtual memory to load models (see Troubleshooting).

## 🔧 Installation

1. **Navigate to your desired parent directory**  
   ```bash
   cd /path/to/your/projects
   ```
2. **Clone the repository**  
   ```bash
   git clone https://github.com/DrAlzahraniInternships/youtube-comments-insights
   cd youtube-comments-insights
   ```
3. **Create & activate a virtual environment** (optional but recommended)  
   ```bash
   python -m venv venv
   ```
   **For Linux & MacOS users**
   ```bash
   source venv/bin/activate       # Linux/macOS
   ```  
   ```bash
   venv\Scripts\activate        # Windows
   ```
4. **Install Python dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the app with Streamlit:

```bash
streamlit run echopulse.py
```

- **Enter** a YouTube video URL in the input box  
- **Adjust** toxicity threshold and number of topic clusters via the sidebar  
- **Click** “🚀 Analyze Video” to start processing  

The app will display:
- Key metrics (positive/negative sentiment, toxicity level)  
- A concise educational summary  
- A recommendation to **Watch**, **Skip**, or **Mixed**  
- Sentiment distribution pie chart  
- Top discussion topics and sample comments  

## 🛠 Troubleshooting

If you encounter build errors or model-loading crashes on Windows:

1. Open **Environment Variables**  
2. Go to **Performance → Settings → Advanced → Virtual Memory → Change**  
3. **Uncheck** “Automatically manage paging file size”  
4. Set **Custom size**:  
   - Initial and maximum size: 15 000–20 000 MB  
5. **Restart** your PC  

This increases your system’s paging file to accommodate large model offloads.

---

*Happy analyzing!*  
