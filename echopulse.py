# EchoPulse: Understand Any Video — Without Watching It
# "Know what a video says — just by reading the comments."

import os
import subprocess
import json
import streamlit as st
st.set_page_config(       # ← must come right after import
    page_title="EchoPulse",
    page_icon="🎥",
    layout="wide",
)
import torch
import pandas as pd
import numpy as np
from bertopic import BERTopic
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import warnings
from youtube_comment_downloader import YoutubeCommentDownloader
from transformers import pipeline, pipeline as hf_pipeline, AutoModelForCausalLM
from pysentimiento import create_analyzer
from typing import List, Dict
import re

warnings.filterwarnings('ignore')


# ─── Toxicity Model ────────────────────────────────────────────────────
toxicity_pipe = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    return_all_scores=True
    device=-1  
)

# ─── Multiple Summarization Options ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_summarization_models():
    models = {}
    try:
        models['mistral'] = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.1",
            device_map="auto",
            load_in_4bit=True,
            torch_dtype=torch.float16,
            offload_folder="offload",      # <— new folder on disk
            offload_state_dict=True,
            max_new_tokens=150,
        )
        st.sidebar.success("✅ Mistral loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Mistral failed: {e}")
        models['mistral'] = None

    
    # Option 2: Try a smaller, more reliable model
    try:
        models['flan_t5'] = hf_pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=0 if torch.cuda.is_available() else -1,
            max_length=200
        )
        print("✅ FLAN-T5 model loaded successfully")
    except Exception as e:
        print(f"❌ FLAN-T5 model failed: {e}")
        models['flan_t5'] = None
    
    # Option 3: DistilBART (lightweight summarization)
    try:
        models['distilbart'] = hf_pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            device=0 if torch.cuda.is_available() else -1,
            max_length=150,
            min_length=60
        )
        print("✅ DistilBART model loaded successfully")
    except Exception as e:
        print(f"❌ DistilBART model failed: {e}")
        models['distilbart'] = None
    
    # Option 4: Simple BERT-based summarization
    try:
        models['bart_large'] = hf_pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
            max_length=150,
            min_length=60
        )
        print("✅ BART-Large model loaded successfully")
    except Exception as e:
        print(f"❌ BART-Large model failed: {e}")
        models['bart_large'] = None
    
    return models

# Initialize models once
_summarization_models = init_summarization_models()

class SummarizerAgent:
    def __init__(self):
        self.models = _summarization_models

    def _extract_meaningful_content(self, title, description, topics):
        """Extract what the video is actually about from title, description, and meaningful topics"""
        
        # Clean and extract key concepts from title
        title_words = re.findall(r'\b[A-Za-z]{3,}\b', title.lower())
        title_concepts = [w for w in title_words if w not in [
            'the', 'and', 'for', 'with', 'from', 'this', 'that', 'video', 'youtube',
            'explained', 'tutorial', 'guide', 'how', 'why', 'what', 'when', 'where'
        ]]
        
        # Extract key information from description
        description_concepts = []
        if description and len(description) > 50:
            # Look for sentences that explain what the video covers
            sentences = description.split('.')[:3]  # First 3 sentences usually have the main content
            for sentence in sentences:
                words = re.findall(r'\b[A-Za-z]{4,}\b', sentence.lower())
                relevant_words = [w for w in words if w not in [
                    'this', 'that', 'will', 'video', 'channel', 'please', 'like', 'subscribe',
                    'comment', 'share', 'watch', 'youtube', 'content', 'today', 'show'
                ]]
                description_concepts.extend(relevant_words[:3])
        
        # Extract meaningful topics from comments
        topic_concepts = []
        if topics:
            for topic in topics[:4]:
                terms = topic.get("terms", [])
                meaningful_terms = [
                    term for term in terms[:5] 
                    if len(term) > 3 and term.lower() not in [
                        'video', 'comment', 'people', 'think', 'really', 'pretty', 'actually',
                        'watch', 'like', 'good', 'great', 'love', 'hate', 'best', 'worst',
                        'youtube', 'channel', 'content', 'stuff', 'thing', 'things'
                    ]
                ]
                topic_concepts.extend(meaningful_terms[:2])
        
        # Combine and deduplicate concepts
        all_concepts = title_concepts[:3] + description_concepts[:3] + topic_concepts[:4]
        unique_concepts = []
        seen = set()
        for concept in all_concepts:
            if concept not in seen and len(concept) > 3:
                unique_concepts.append(concept)
                seen.add(concept)
        
        return unique_concepts[:6]  # Top 6 most relevant concepts

    def _determine_video_subject(self, title, description, concepts):
        """Determine what the video is actually teaching or explaining"""
        
        title_lower = title.lower()
        desc_lower = description.lower() if description else ""
        
        # Try to understand the main subject from context
        if concepts:
            primary_subject = concepts[0]
            secondary_subjects = concepts[1:3] if len(concepts) > 1 else []
            
            # Make it more readable
            subject_desc = primary_subject.replace('_', ' ').replace('-', ' ')
            
            # Add context based on other concepts
            if secondary_subjects:
                context_terms = [c.replace('_', ' ').replace('-', ' ') for c in secondary_subjects]
                subject_desc += f" and {context_terms[0]}"
                if len(context_terms) > 1:
                    subject_desc += f" along with {context_terms[1]}"
            
            return subject_desc
        
        # Fallback: extract from title more intelligently
        if 'explained' in title_lower:
            # Find what's being explained
            parts = title.split(' - ')
            if len(parts) > 1:
                return parts[0].replace('Explained', '').strip()
        
        # Extract main nouns from title
        words = title.split()
        main_words = [w for w in words if len(w) > 4 and w[0].isupper()]
        if main_words:
            return ' '.join(main_words[:2])
        
        return "the main topic"

    def _analyze_viewer_understanding(self, sentiment_data, topics, concepts):
        """Analyze what viewers understood and how they responded"""
        
        pos = sentiment_data["positive_percentage"]
        neg = sentiment_data["negative_percentage"]
        
        # Determine how well viewers received the explanation
        if pos > 75:
            reception = "Viewers found the explanation very clear and informative"
            detail = "with many appreciating the detailed breakdown"
        elif pos > 60:
            reception = "Most viewers understood the content well"
            detail = "finding the explanation helpful and well-structured"
        elif pos > 45:
            reception = "Viewers had mixed reactions to the explanation"
            detail = "with some finding it clearer than others"
        elif neg > 55:
            reception = "Many viewers found the content confusing or incomplete"
            detail = "expressing concerns about clarity and depth"
        else:
            reception = "Viewers are divided on how well the topic was explained"
            detail = "with varying levels of understanding"
        
        # What specific aspects are viewers discussing
        discussion_focus = []
        if topics and concepts:
            # Match topic terms with our extracted concepts to see what resonates
            for topic in topics[:3]:
                terms = topic.get("terms", [])
                for term in terms[:3]:
                    if any(concept in term.lower() or term.lower() in concept for concept in concepts):
                        if term not in discussion_focus and len(term) > 3:
                            discussion_focus.append(term)
        
        return {
            "reception": reception,
            "detail": detail,
            "discussion_focus": discussion_focus[:3]
        }

    def generate_summary(
        self,
        title: str,
        description: str,
        sentiment_data: Dict,
        topics: List[Dict],
        toxicity_data: Dict,
        comment_data: Dict
    ) -> str:
        """Generate intelligent summary that explains what the video teaches"""
        
        # Extract meaningful content concepts
        concepts = self._extract_meaningful_content(title, description, topics)
        video_subject = self._determine_video_subject(title, description, concepts)
        viewer_analysis = self._analyze_viewer_understanding(sentiment_data, topics, concepts)
        
        # Try advanced model with teacher-like prompt
        if self.models.get('mistral'):
            try:
                context = f"""Video Subject: {video_subject}
Key Concepts Covered: {', '.join(concepts[:4]) if concepts else 'main topic'}
Viewer Reception: {viewer_analysis['reception']}
What Viewers Discuss: {', '.join(viewer_analysis['discussion_focus']) if viewer_analysis['discussion_focus'] else 'the explanations provided'}
Video Description Context: {description[:300] if description else 'No description available'}"""

                prompt = f"""[INST] You are a teacher summarizing what a YouTube educational video covers based on viewer comments and reactions. 

{context}

Write a clear 6-7 sentence summary that explains:
1. What specific topic or concept the video teaches/explains
2. What viewers learned or understood from it
3. How well the explanation worked for the audience
4. What specific aspects viewers found most interesting or challenging

Write like you're explaining to a student what this video is about and whether it's worth watching for learning. Don't mention the title directly, percentages, or comment counts. Focus on the educational value and viewer learning experience. [/INST]

This video teaches"""

                result = self.models['mistral'](
                    prompt,
                    max_new_tokens=160,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.models['mistral'].tokenizer.eos_token_id,
                )
                
                generated_text = result[0]["generated_text"]
                if "[/INST]" in generated_text:
                    summary = generated_text.split("[/INST]")[-1].strip()
                    summary = re.sub(r'^(This video teaches|This video)', '', summary).strip()
                    
                    if summary and 150 < len(summary) < 600:
                        # Ensure it starts properly
                        if summary and not summary[0].isupper():
                            summary = summary[0].upper() + summary[1:]
                        
                        # Check for quality - avoid repetitive content
                        words = summary.split()
                        if len(set(words)) / len(words) > 0.7:  # Good vocabulary diversity
                            return summary.strip()
                
            except Exception as e:
                print(f"Mistral summary generation failed: {e}")

        # Rule-based fallback with teacher-like approach
        return self._generate_educational_summary(video_subject, viewer_analysis, sentiment_data, concepts)
    
    def _generate_educational_summary(self, video_subject, viewer_analysis, sentiment_data, concepts):
        """Generate educational-focused summary explaining what the video teaches"""
        
        pos = sentiment_data["positive_percentage"]
        
        # Start with what the video explains/teaches
        summary_parts = []
        
        # Opening: What does the video teach?
        if concepts and len(concepts) >= 2:
            summary_parts.append(f"This video explains {video_subject}, covering key aspects like {concepts[1]} and {concepts[2] if len(concepts) > 2 else 'related concepts'}.")
        else:
            summary_parts.append(f"This video provides an in-depth explanation of {video_subject}.")
        
        # How did viewers respond to the explanation?
        summary_parts.append(f"{viewer_analysis['reception']}, {viewer_analysis['detail']}.")
        
        # What specifically resonated with viewers?
        if viewer_analysis['discussion_focus']:
            focus_items = viewer_analysis['discussion_focus'][:2]
            summary_parts.append(f"Viewers particularly engaged with discussions about {focus_items[0]}" + 
                               (f" and {focus_items[1]}" if len(focus_items) > 1 else "") + 
                               ", showing strong interest in these specific elements.")
        else:
            if pos > 60:
                summary_parts.append("The explanations resonated well with viewers who found the content educational and worthwhile.")
            else:
                summary_parts.append("Viewers had varying levels of engagement with the explanations provided.")
        
        # Educational value assessment
        if pos > 70:
            summary_parts.append("Based on viewer feedback, this appears to be an effective educational resource that successfully breaks down complex concepts.")
        elif pos > 50:
            summary_parts.append("The video serves as a decent learning resource, though some viewers felt certain aspects could be explained more clearly.")
        else:
            summary_parts.append("While the topic is covered, viewers suggest the explanation could benefit from clearer presentation or more detailed examples.")
        
        # Final learning outcome
        if concepts:
            main_takeaway = concepts[0].replace('_', ' ').replace('-', ' ')
            summary_parts.append(f"Overall, viewers gain insights into {main_takeaway} and develop a better understanding of the subject matter.")
        else:
            summary_parts.append("The video contributes to viewers' understanding of the topic, with many walking away with new insights.")
        
        return " ".join(summary_parts)

# =============================================================================
# CORE AGENTS AND MODULES (rest remains mostly the same)
# =============================================================================

class CommentExtractor:
    @staticmethod
    def extract_comments(_video_url):
        vid = _video_url.split("v=")[-1].split("&")[0]
        downloader = YoutubeCommentDownloader()
        comments = [c["text"] for c in downloader.get_comments(vid)]

        try:
            meta_json = subprocess.check_output(
                ["yt-dlp", "--dump-single-json", _video_url],
                stderr=subprocess.DEVNULL
            )
            meta = json.loads(meta_json)
            title       = meta.get("title", vid)
            description = meta.get("description", "")
        except:
            title = vid
            description = ""

        return {
            "comments": comments,
            "video_title": title,
            "video_description": description,
            "comment_count": len(comments),
            "video_id": vid
        }

# ─── Batch helper ───────────────────────────────────────────────
def batch_process(pipe, comments, batch_size=100):
    """Run the HF pipeline in chunks to save peak RAM."""
    results = []
    for i in range(0, len(comments), batch_size):
        chunk = comments[i : i + batch_size]
        results.extend(pipe(chunk, truncation=True, padding=True))
    return results



class ToxicityFilter:
    """Enhanced toxicity filter with multiple detection methods"""
    
    def __init__(self):
        # Common toxic/inappropriate words and phrases
        self.toxic_keywords = {
            'hate_speech': ['hate', 'nazi', 'terrorist', 'kill yourself', 'kys'],
            'profanity': ['fuck', 'shit', 'bitch', 'asshole', 'damn', 'hell', 'crap'],
            'harassment': ['idiot', 'stupid', 'moron', 'loser', 'retard', 'dumb'],
            'spam': ['subscribe', 'like and subscribe', 'check out my channel', 'follow me'],
            'offensive': ['gay', 'lame', 'sucks', 'worst', 'terrible', 'awful', 'trash']
        }
    
    def _keyword_toxicity_check(self, comment):
        """Check for toxic keywords (backup method)"""
        comment_lower = comment.lower()
        toxicity_score = 0.0
        
        for category, words in self.toxic_keywords.items():
            for word in words:
                if word in comment_lower:
                    if category == 'hate_speech':
                        toxicity_score += 0.8
                    elif category == 'profanity':
                        toxicity_score += 0.4
                    elif category == 'harassment':
                        toxicity_score += 0.5
                    elif category == 'spam':
                        toxicity_score += 0.3
                    elif category == 'offensive':
                        toxicity_score += 0.3
        
        return min(toxicity_score, 1.0)  # Cap at 1.0
    
    def _is_spam_comment(self, comment):
        """Detect spam comments"""
        comment_lower = comment.lower()
        spam_indicators = [
            'subscribe', 'sub4sub', 'follow me', 'check my channel',
            'like and subscribe', 'please subscribe', 'sub to me',
            'follow for follow', 'f4f', 's4s'
        ]
        return any(indicator in comment_lower for indicator in spam_indicators)
    
    def _is_low_quality(self, comment):
        """Detect low quality comments"""
        if len(comment.strip()) < 3:  # Too short
            return True
        if comment.count('!') > 3 or comment.count('?') > 2:  # Too many punctuation
            return True
        if len(set(comment.lower().split())) < 2:  # Too repetitive
            return True
        return False

    @st.cache_data(show_spinner=False)
    def filter_comments(_self, comments, toxicity_threshold=0.3):
        if not comments:
            return {
                "clean_comments": [],
                "toxicity_percentage": 0.0,
                "filtered_count": 0,
                "filter_breakdown": {"toxic": 0, "spam": 0, "low_quality": 0}
            }
        
        # Use ML model for primary detection
        try:
            batch_preds = toxicity_pipe(
                comments,
                truncation=True,
                padding=True,
                max_length=512
            )
        except Exception as e:
            print(f"ML toxicity detection failed: {e}")
            batch_preds = [{"label": "TOXIC", "score": 0.0}] * len(comments)
        
        clean = []
        filter_stats = {"toxic": 0, "spam": 0, "low_quality": 0}
        
        for comment, preds in zip(comments, batch_preds):
            # Get ML toxicity score
            ml_tox_score = 0.0
            if isinstance(preds, list):
                ml_tox_score = next((p['score'] for p in preds if p['label']=='TOXIC'), 0.0)
            
            # Get keyword-based toxicity score
            keyword_tox_score = _self._keyword_toxicity_check(comment)
            
            # Use the higher of the two scores
            final_tox_score = max(ml_tox_score, keyword_tox_score)
            
            # Check other filter criteria
            is_spam = _self._is_spam_comment(comment)
            is_low_quality = _self._is_low_quality(comment)
            
            # Apply filters
            if final_tox_score > toxicity_threshold:
                filter_stats["toxic"] += 1
            elif is_spam:
                filter_stats["spam"] += 1
            elif is_low_quality:
                filter_stats["low_quality"] += 1
            else:
                clean.append(comment)
        
        total_filtered = sum(filter_stats.values())
        toxicity_percentage = 100 * total_filtered / (len(comments) or 1)
        
        return {
            "clean_comments": clean,
            "toxicity_percentage": round(toxicity_percentage, 1),
            "filtered_count": total_filtered,
            "filter_breakdown": filter_stats
        }

class SentimentAgent:
    """Analyzes sentiment using pysentimiento (DistilRoBERTa under the hood)."""
    def __init__(self):
        self.analyzer = create_analyzer(task="sentiment", lang="en")

    @st.cache_data(show_spinner=False)   
    def analyze_sentiment(_self, comments):
        if not comments:
            return []
            
        results = []
        for c in comments:
            try:
                res = _self.analyzer.predict(c)
                # Map PySentimiento outputs to our labels
                label_map = {"POS": "Positive", "NEG": "Negative", "NEU": "Neutral"}
                score = res.probas.get(res.output, 0.0)
                # Convert to polarity: POS→+score, NEG→–score, NEU→0.0
                polarity = score if res.output == "POS" else (-score if res.output == "NEG" else 0.0)
                results.append({"comment": c, "score": polarity, "label": label_map[res.output]})
            except:
                # Fallback for failed predictions
                results.append({"comment": c, "score": 0.0, "label": "Neutral"})
        return results

    def get_sentiment_summary(self, sentiments):
        if not sentiments:
            return {
                "average_sentiment": 0.0,
                "sentiment_distribution": Counter({"Neutral": 1}),
                "positive_percentage": 0.0,
                "negative_percentage": 0.0,
                "neutral_percentage": 100.0
            }
            
        scores = [s["score"] for s in sentiments]
        labels = [s["label"] for s in sentiments]
        total = len(labels) or 1
        return {
            "average_sentiment": np.mean(scores),
            "sentiment_distribution": Counter(labels),
            "positive_percentage": 100 * labels.count("Positive") / total,
            "negative_percentage": 100 * labels.count("Negative") / total,
            "neutral_percentage":  100 * labels.count("Neutral")  / total
        }

class TopicClusterer:
    """Groups comments into themes using BERTopic."""
    def __init__(self):
        self.model = None

    @st.cache_data(show_spinner=False)
    def extract_topics(_self, comments, n_clusters=5):
        if not comments or all(not c.strip() for c in comments):
            return []
        # fit-transform
        try:
            topic_model = BERTopic(nr_topics=n_clusters,
                                   language="english",
                                   calculate_probabilities=False)
            topics, _ = topic_model.fit_transform(comments)
            info = topic_model.get_topic_info()  # DataFrame with topic, count, and terms
            # build same output format
            topics_out = []
            for _, row in info[info.Topic >= 0].iterrows():
                tid = int(row.Topic)
                terms = [t[0] for t in topic_model.get_topic(tid)]
                size = int(row.Count)
                topics_out.append({"topic_id": tid, "terms": terms, "size": size})
            return topics_out
        except:
            return []


class VerdictAgent:
    """Enhanced recommendation system with more nuanced decisions"""
    
    def make_verdict(self, sentiment_data, topics, toxicity_data):
        """Generate more accurate watch recommendation"""
        pos_pct = sentiment_data['positive_percentage']
        neg_pct = sentiment_data['negative_percentage']
        neu_pct = sentiment_data['neutral_percentage']
        toxicity_pct = toxicity_data['toxicity_percentage']
        
        # Calculate engagement quality
        total_comments = len(toxicity_data.get('clean_comments', []))
        engagement_score = min(total_comments / 50, 2.0)  # Normalize engagement
        
        # Calculate sentiment strength (how polarized opinions are)
        sentiment_strength = abs(pos_pct - neg_pct)
        
        # Enhanced decision logic
        if pos_pct >= 75 and toxicity_pct < 10:
            verdict = "Watch"
            confidence = "High"
            reason = "Overwhelmingly positive reception with clean discussion"
            
        elif pos_pct >= 65 and toxicity_pct < 15 and sentiment_strength > 20:
            verdict = "Watch"
            confidence = "High"
            reason = "Strong positive consensus among viewers"
            
        elif pos_pct >= 60 and neg_pct < 25 and toxicity_pct < 20:
            verdict = "Watch"
            confidence = "Medium"
            reason = "Generally positive with manageable criticism"
            
        elif pos_pct >= 50 and toxicity_pct < 25 and engagement_score > 1.0:
            verdict = "Watch"
            confidence = "Medium"
            reason = "Decent reception with good viewer engagement"
            
        elif neg_pct >= 70 or toxicity_pct > 40:
            verdict = "Skip"
            confidence = "High"
            reason = "Predominantly negative or highly toxic discussion"
            
        elif neg_pct >= 55 and sentiment_strength > 25:
            verdict = "Skip"
            confidence = "High"
            reason = "Strong negative consensus from viewers"
            
        elif neg_pct >= 45 and toxicity_pct > 25:
            verdict = "Skip"
            confidence = "Medium"
            reason = "Significant criticism with concerning toxicity levels"
            
        elif sentiment_strength < 15 and neu_pct > 40:
            verdict = "Mixed"
            confidence = "Medium"
            reason = "Neutral reception - depends on personal preferences"
            
        elif sentiment_strength < 20 and abs(pos_pct - neg_pct) < 10:
            verdict = "Mixed"
            confidence = "Medium"
            reason = "Sharply divided opinions among viewers"
            
        elif pos_pct > neg_pct and toxicity_pct < 30:
            verdict = "Watch"
            confidence = "Low"
            reason = "Slightly positive but with notable concerns"
            
        else:
            verdict = "Mixed"
            confidence = "Low"
            reason = "Unclear consensus - review detailed analysis"
        
        # Calculate overall score
        score = pos_pct - neg_pct - (toxicity_pct * 0.5)
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'reason': reason,
            'score': round(score, 1),
            'engagement_level': 'High' if engagement_score > 1.5 else ('Medium' if engagement_score > 0.8 else 'Low')
        }

# =============================================================================
# STREAMLIT DASHBOARD
# =============================================================================

def create_dashboard():
    st.title("🎥 EchoPulse")
    st.subheader("Understand Any Video — Just by reading the comments")
    st.markdown("*Know what a video says — just by reading the comments.*")

    # Show available models
    available_models = [name for name, model in _summarization_models.items() if model is not None]
    if available_models:
        st.sidebar.success(f"✅ Loaded models: {', '.join(available_models)}")
    else:
        st.sidebar.warning("⚠️ Using rule-based summarization (no ML models available)")

    video_url = st.text_input(
        "Enter YouTube Video URL:",
        placeholder="https://youtube.com/watch?v=example"
    )
    toxicity_threshold = st.sidebar.slider(
        "Toxicity Filter Threshold", 0.1, 1.0, 0.3, 0.1
    )
    n_topics = st.sidebar.selectbox(
        "Number of Topic Clusters", [3,4,5,6,7], index=2
    )

    if st.button("🚀 Analyze Video"):
        if video_url:
            st.cache_data.clear()    # only if you want to bust cache on each run
            analyze_video(video_url, toxicity_threshold, n_topics)
        else:
            st.error("Please enter a YouTube video URL")

def analyze_video(video_url, toxicity_threshold, n_topics):
    """Main analysis pipeline"""

    with st.spinner("🔍 Extracting comments..."):
        extractor = CommentExtractor()
        comment_data = extractor.extract_comments(video_url)

    st.success(f"✅ Extracted {comment_data['comment_count']} comments")

    # Initialize agents
    toxicity_filter = ToxicityFilter()
    sentiment_agent = SentimentAgent()
    topic_clusterer = TopicClusterer()
    summarizer = SummarizerAgent()
    verdict_agent = VerdictAgent()

    with st.spinner("🧹 Filtering toxic content..."):
        filter_results = toxicity_filter.filter_comments(
            comment_data['comments'],
            toxicity_threshold
        )

    with st.spinner("😊 Analyzing sentiment..."):
        sentiments = sentiment_agent.analyze_sentiment(filter_results['clean_comments'])
        sentiment_summary = sentiment_agent.get_sentiment_summary(sentiments)

    with st.spinner("🧠 Extracting topics..."):
        topics = topic_clusterer.extract_topics(filter_results['clean_comments'], n_topics)

    # Generate summary & verdict
    with st.spinner("📝 Generating summary..."):
        # Pass the filter_results as comment_data to the summarizer
        enhanced_comment_data = {**comment_data, **filter_results}
        summary = summarizer.generate_summary(
            comment_data["video_title"],
            comment_data["video_description"],
            sentiment_summary,
            topics,
            filter_results,
            enhanced_comment_data  # Pass the enhanced data
        )
        verdict = verdict_agent.make_verdict(sentiment_summary, topics, filter_results)

    # Display results
    display_results(
        comment_data,
        filter_results,
        sentiment_summary,
        sentiments,
        topics,
        summary,
        verdict
    )


def display_results(comment_data, filter_results, sentiment_summary, sentiments, topics, summary, verdict):
    """Enhanced results display with better toxicity and recommendation info"""
    
    st.header("📊 Analysis Results")
    
    # Key metrics with enhanced toxicity display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📝 Summary Score",
            f"{sentiment_summary['positive_percentage']:.0f}%",
            f"{sentiment_summary['positive_percentage'] - sentiment_summary['negative_percentage']:.0f}% net positive"
        )
    
    with col2:
        # Enhanced toxicity display
        toxicity_pct = filter_results['toxicity_percentage']
        filtered_count = filter_results['filtered_count']
        
        if toxicity_pct == 0:
            toxicity_color = "🟢"
            toxicity_status = "Clean"
        elif toxicity_pct < 10:
            toxicity_color = "🟡"
            toxicity_status = "Low"
        elif toxicity_pct < 25:
            toxicity_color = "🟠"
            toxicity_status = "Moderate"
        else:
            toxicity_color = "🔴"
            toxicity_status = "High"
        
        st.metric(
            "☣️ Content Quality",
            f"{toxicity_color} {toxicity_status}",
            f"{filtered_count} comments filtered ({toxicity_pct}%)"
        )
    
    with col3:
        # Enhanced recommendation display
        verdict_colors = {"Watch": "🟢", "Mixed": "🟡", "Skip": "🔴"}
        confidence_emoji = {"High": "💪", "Medium": "👍", "Low": "🤷"}
        
        st.metric(
            "✅ Recommendation",
            f"{verdict_colors.get(verdict['verdict'], '⚪')} {verdict['verdict']}",
            f"{confidence_emoji.get(verdict['confidence'], '❓')} {verdict['confidence']} confidence"
        )
    
    with col4:
        engagement_emoji = {"High": "🔥", "Medium": "👀", "Low": "😴"}
        st.metric(
            "💬 Engagement",
            f"{engagement_emoji.get(verdict.get('engagement_level', 'Medium'), '📊')} {verdict.get('engagement_level', 'Medium')}",
            f"{comment_data['comment_count']} total comments"
        )
    
    # Enhanced toxicity breakdown
    if filter_results['filtered_count'] > 0:
        st.subheader("🛡️ Content Filtering Breakdown")
        breakdown = filter_results.get('filter_breakdown', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🚫 Toxic Comments", breakdown.get('toxic', 0))
        with col2:
            st.metric("📢 Spam Filtered", breakdown.get('spam', 0))
        with col3:
            st.metric("💭 Low Quality Removed", breakdown.get('low_quality', 0))
    
    # Main summary
    st.subheader("📋 Summary")
    st.info(summary)
    
    # Enhanced verdict details
    st.subheader("🎯 Detailed Recommendation")
    verdict_emoji = {"Watch": "✅", "Mixed": "⚡", "Skip": "❌"}
    
    recommendation_text = f"""
    **{verdict_emoji.get(verdict['verdict'], '⚪')} {verdict['verdict']}** ({verdict['confidence']} confidence)
    
    *Reasoning:* {verdict['reason']}
    
    *Overall Score:* {verdict['score']}/100 (Positive sentiment minus negative factors)
    """
    
    if verdict['verdict'] == "Watch":
        st.success(recommendation_text)
    elif verdict['verdict'] == "Skip":
        st.error(recommendation_text)
    else:
        st.warning(recommendation_text)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Sentiment Distribution")
        
        # Pie chart
        labels = list(sentiment_summary['sentiment_distribution'].keys())
        values = list(sentiment_summary['sentiment_distribution'].values())
        colors = ['#2E8B57', '#FFD700', '#DC143C']  # Green, Yellow, Red
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent'
        )])
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🧠 Top Discussion Topics")
        
        if topics:
            for i, topic in enumerate(topics[:5]):
                with st.expander(f"Topic {i+1} ({topic['size']} comments)"):
                    st.write("**Key terms:** " + ", ".join(topic['terms']))
        else:
            st.info("No clear topics identified in the comments.")
    
    # Sentiment timeline (simulated)
    if sentiments:
        timeline_df = create_sentiment_timeline(sentiments)
        if not timeline_df.empty:
            st.subheader("📊 Sentiment Timeline")
            fig = px.line(
                timeline_df,
                x='comment_index',
                y='sentiment_score',
                title='Sentiment Flow Throughout Comments',
                labels={'comment_index': 'Comment Order', 'sentiment_score': 'Sentiment Score'}
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data to build a sentiment timeline.")
      
    # Raw comments sample
    if sentiments:
        st.subheader("💬 Sample Comments")
        
        # Show positive and negative examples
        positive_comments = [s for s in sentiments if s['score'] > 0.1]
        negative_comments = [s for s in sentiments if s['score'] < -0.1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**😊 Most Positive Comments**")
            for comment in sorted(positive_comments, key=lambda x: x['score'], reverse=True)[:3]:
                st.success(f"💚 {comment['comment']}")
        
        with col2:
            st.markdown("**😞 Most Critical Comments**")
            for comment in sorted(negative_comments, key=lambda x: x['score'])[:3]:
                st.error(f"💛 {comment['comment']}")

def create_sentiment_timeline(sentiments):
    """Create timeline data for sentiment visualization"""
    timeline_data = []
    for i, sentiment in enumerate(sentiments):
        timeline_data.append({
            'comment_index': i + 1,
            'sentiment_score': sentiment['score'],
            'label': sentiment['label']
        })
    return pd.DataFrame(timeline_data)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

if __name__ == "__main__":
    create_dashboard()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **🔬 EchoPulse - Open Source Video Analysis**
    
    📋 **Features:**
    - ✅ Comment-based video analysis
    - ✅ Sentiment analysis with timeline
    - ✅ Topic clustering and theme extraction
    - ✅ Toxicity filtering
    - ✅ Automated recommendations
    - ✅ Interactive dashboard
    
    """)
