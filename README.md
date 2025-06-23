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

## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/).
2. **Docker**: [Install Docker](https://www.docker.com/).
3. **Linux/macOS**: No extra setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker’s WSL integration as per [Docker’s guide](https://docs.docker.com/desktop/windows/wsl/).

---

### Step 1: Remove the existing code directory completely

If your local EchoPulse folder is out-of-date or corrupted, remove it first:

```bash
rm -rf EchoPulse
```

### Step 2: Clone the Repository

Clone the EchoPulse GitHub repository:

```bash
git clone https://github.com/Shwejan/EchoPulse.git
```

### Step 3: Navigate to the Repository

```bash
cd EchoPulse
```

### Step 4: Pull the Latest Version

```bash
git pull origin main
```

### Step 5: Make Docker Scripts Executable

```bash
chmod +x docker-launch.sh docker-cleanup.sh
```

### Step 6: Build and Run the Docker Container

```bash
./docker-launch.sh
```

### Step 7: Access EchoPulse

Open your browser to:

```
http://localhost:8501
```

The Streamlit app will be running on port 8501.

### Step 8: Stop and Remove the Docker Container

When you’re done, clean up resources:

```bash
./docker-cleanup.sh
```

- **Enter** a YouTube video URL in the input box  
- **Click** “🚀 Analyze Video” to start processing  

The app will display:
- Key metrics (positive/negative sentiment, toxicity level)  
- A concise educational summary  
- A recommendation to **Watch**, **Skip**, or **Mixed**  
- Sentiment distribution pie chart  
- Top discussion topics and sample comments  

