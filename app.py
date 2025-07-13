import streamlit as st
import os
import smtplib
from datetime import datetime, timedelta
from urllib.parse import urlparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import feedparser
import spacy
from newsapi.newsapi_client import NewsApiClient
from newspaper import Article, Config
from openai import OpenAI

# Import the visualization functions from our separate file
from viz_utils import create_sentiment_donut_chart, create_source_bar_chart, create_word_cloud

# --- SETUP & CONFIGURATION ---
@st.cache_resource
def setup_openai_client():
    try:
        return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception as e:
        st.error(f"Could not set up OpenAI client: {e}"); st.stop()

@st.cache_resource
def setup_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("SpaCy model 'en_core_web_sm' not found in your requirements.txt."); st.stop()

openai_client = setup_openai_client()
nlp = setup_spacy_model()

MY_API_KEY = st.secrets["NEWSAPI_KEY"]
SENDER_EMAIL = st.secrets["SENDER_EMAIL"]
SENDER_PASSWORD = st.secrets["SENDER_PASSWORD"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# --- HELPER FUNCTIONS ---

def parse_sentiment(sentiment_string):
    if "Positive" in sentiment_string: return "Positive"
    if "Negative" in sentiment_string: return "Negative"
    return "Neutral"

# (The rest of your helper functions are unchanged)
def fetch_google_news_mentions(person_name, from_date, to_date):
    mentions_found = []
    try:
        query_terms = f'"{person_name}" after:{from_date.strftime("%Y-%m-%d")} before:{to_date.strftime("%Y-%m-%d")}'
        rss_url = f"https://news.google.com/rss/search?q={query_terms.replace(' ', '%20')}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        for entry in feed.entries:
            mentions_found.append((entry.get("title", "No Title"), entry.get("link", "")))
        return mentions_found
    except Exception as e:
        st.warning(f"Could not fetch from Google News RSS: {e}"); return []

def fetch_from_newsapi(api_client, person_name, from_date, to_date):
    try:
        all_articles = api_client.get_everything(
            q=f'"{person_name}"', from_param=from_date.isoformat(), to=to_date.isoformat(),
            language='en', sort_by='relevancy', page_size=40
        )
        return [(article.get('title', 'No Title'), article.get('url')) for article in all_articles.get('articles', [])]
    except Exception as e:
        st.error(f"Error fetching from NewsAPI: {e}"); return []

def process_article(url, name_to_find):
    try:
        config = Config(); config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36'; config.request_timeout = 25
        article = Article(url, config=config); article.download(); article.parse()
        title = article.title if article.title else "Title Not Found"
        if not article.text or len(article.text) < 250: return (None, None, None)
        doc = nlp(article.text)
        found_sentences = [s.text.strip().replace('\n', ' ') for s in doc.sents if name_to_find.lower() in s.text.lower()]
        return (title, found_sentences, article.text)
    except Exception: return (None, None, None)

def get_summary_from_gpt(article_text):
    if not article_text: return "Summary could not be generated."
    system_prompt = "You are an expert news editor. Create a concise, neutral, two-sentence summary of the provided news article text."
    user_prompt = f"Please summarize the following article text:\n\n---\n\n{article_text}"
    try:
        response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0.2, max_tokens=150)
        return response.choices[0].message.content.strip()
    except Exception as e: return f"Summary generation failed: {e}"

def get_sentiment_from_gpt(person_name, sentences):
    if not sentences: return "No mentions found."
    context_text = " ".join(sentences)
    system_prompt = "You are an expert news analyst. Determine if the sentiment of a news mention towards a person is Positive, Negative, or Neutral. Base your judgment ONLY on the provided text."
    user_prompt = f"Person: {person_name}\nSentences: \"{context_text}\"\n\nFormat your response as: Sentiment: [Positive/Negative/Neutral]. Justification: [A brief, one-sentence explanation.]"
    try:
        response = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0, max_tokens=100)
        return response.choices[0].message.content.strip()
    except Exception as e: return f"Sentiment analysis failed: {e}"

def send_email_with_attachment(subject, body, recipient_email, file_path):
    if not SENDER_PASSWORD: return False
    try:
        msg = MIMEMultipart(); msg['From'] = SENDER_EMAIL; msg['To'] = recipient_email; msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        with open(file_path, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream'); part.set_payload(attachment.read())
        encoders.encode_base64(part); part.add_header('Content-Disposition', f'attachment; filename= {os.path.basename(file_path)}'); msg.attach(part)
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT); server.starttls(); server.login(SENDER_EMAIL, SENDER_PASSWORD); server.send_message(msg); server.quit()
        return True
    except Exception as e:
        st.error(f"An error occurred while sending the email: {e}"); return False

# --- STREAMLIT WEB APPLICATION INTERFACE ---
st.set_page_config(page_title="kaitlyn's news report", layout="wide", page_icon="📰")
st.title("📰 kaitlyn's daily news report")
st.markdown("""
track news mentions for any public figure + get an AI summary/sentiment report

enter a name, date, and email to get started

*refrain from entering today's date for optimal functionality*
""")


col1, col2 = st.columns(2)
with col1:
    person_name = st.text_input("👤 **Person's Full Name**", placeholder="e.g., Tom Smith")
    date_input = st.date_input("🗓️ **Date to Search**", datetime.now() - timedelta(days=1))
with col2:
    recipient_email = st.text_input("✉️ **Your Email Address (Optional)**", placeholder="Enter your email to receive the report")

if st.button("🚀 Generate Report", type="primary", use_container_width=True):
    if not person_name:
        st.warning("Please enter a person's name to start the analysis."); st.stop()

    from_date = date_input
    to_date = from_date + timedelta(days=1)
    
    results, failed_articles, sentiments_list, sources_list = {}, [], [], []
    wordcloud_text = ""

    with st.status(f"Running Analysis for '{person_name}'...", expanded=True) as status:
        status.write("🧠 **Step 1: Fetching Articles**")
        newsapi_client = NewsApiClient(api_key=MY_API_KEY)
        newsapi_articles = fetch_from_newsapi(newsapi_client, person_name, from_date, to_date)
        status.write(f"✅ Found {len(newsapi_articles)} articles from NewsAPI.")
        google_mentions = fetch_google_news_mentions(person_name, from_date, to_date)
        status.write(f"✅ Found {len(google_mentions)} mentions from Google News.")

        if not newsapi_articles and not google_mentions:
            status.update(label="Analysis failed!", state="error", expanded=True); st.error(f"No articles or mentions found for '{person_name}' on {from_date.strftime('%Y-%m-%d')}."); st.stop()
        
        if newsapi_articles:
            status.write(f"🧠 **Step 2: Analyzing {len(newsapi_articles)} Articles**")
            for i, (original_title, url) in enumerate(newsapi_articles):
                status.write(f"➡️ **Processing Article {i+1}/{len(newsapi_articles)}:** [{original_title}]({url})")
                processed_title, mentions, article_text = process_article(url, person_name)
                
                # --- MODIFIED LOGIC BLOCK ---
                if article_text:
                    # Condition: Only proceed if the person was actually mentioned in sentences.
                    if mentions:
                        status.write("   - ✅ Content parsed, mentions found. Proceeding with AI analysis.")
                        summary = get_summary_from_gpt(article_text)
                        sentiment = get_sentiment_from_gpt(person_name, mentions)
                        
                        final_title = processed_title if processed_title != "Title Not Found" else original_title
                        results[url] = {'title': final_title, 'summary': summary, 'mentions': mentions, 'sentiment': sentiment}
                        
                        sentiments_list.append(parse_sentiment(sentiment))
                        sources_list.append(urlparse(url).netloc.replace('www.', ''))
                        wordcloud_text += f" {final_title} {summary}"
                    else:
                        # This article is readable but doesn't mention the person, so we skip it.
                        status.write(f"   - ⚠️ Skipping (no specific mentions of '{person_name}' found in article text).")
                else:
                    # This article failed to download (paywall, etc.)
                    status.write("   - ⚠️ Skipping (article content is unreadable or too short).")
                    failed_articles.append((original_title, url))
        
        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

    if not results and not google_mentions:
        st.warning("No articles with direct mentions were found to analyze."); st.stop()
    
    if results: st.balloons()
    
    st.header("📊 Report at a Glance", divider='rainbow')
    donut_fig = create_sentiment_donut_chart(sentiments_list)
    bar_fig = create_source_bar_chart(sources_list)
    wordcloud_img = create_word_cloud(wordcloud_text)
    
    if results:
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            if donut_fig: st.plotly_chart(donut_fig, use_container_width=True)
        with viz_col2:
            if bar_fig: st.plotly_chart(bar_fig, use_container_width=True)
        if wordcloud_img:
            st.subheader("Keyword Cloud"); st.image(wordcloud_img, use_container_width=True)

    st.header("📄 Detailed Article Breakdown", divider='rainbow')
    report_text_content = f"News Report for {person_name} on {from_date.strftime('%A, %B %d, %Y')}\n" + "="*50 + "\n\n"
    
    if results:
        report_text_content += "--- Analyzed Articles from NewsAPI ---\n\n"
        for i, (url, data) in enumerate(results.items(), 1):
            with st.container(border=True):
                st.subheader(f"{i}. {data.get('title', 'Title Not Found')}", anchor=False); st.markdown(f"**Source:** [{url}]({url})")
                st.info(f"**AI Summary:** {data['summary']}")
                if "Positive" in data['sentiment']: st.success(f"**Sentiment:** {data['sentiment']}")
                elif "Negative" in data['sentiment']: st.error(f"**Sentiment:** {data['sentiment']}")
                else: st.warning(f"**Sentiment:** {data['sentiment']}")
                if data['mentions']:
                    with st.expander("Show mentions..."):
                        for sent in data['mentions']: st.markdown(f'- "{sent}"')
            report_text_content += f"{i}. {data.get('title', 'Title Not Found')}\n   URL: {url}\n\n   AI Summary: {data['summary']}\n\n   Sentiment Analysis: {data['sentiment']}\n\n   Mentions Found:\n"
            for sent in data['mentions']: report_text_content += f'   - "{sent}"\n'
            report_text_content += "\n"
    
    if failed_articles:
        st.subheader("Unanalyzable Articles from NewsAPI")
        st.warning("These articles were found but were likely behind a paywall or blocked by the publisher:")
        for title, url in failed_articles:
            st.markdown(f"- **{title}** ([Source]({url}))")
            
    if google_mentions:
        st.subheader(f"Mentions Found on Google News")
        st.info("Note: These links lead to Google and may require an extra click to reach the article. Analysis is not performed on these sources.")
        for title, link in google_mentions:
            st.markdown(f"- **{title}** ([Source]({link}))")

    if recipient_email and (results or google_mentions or failed_articles):
        if failed_articles:
            report_text_content += "\n--- Unanalyzable Articles from NewsAPI ---\n(Note: These links were found but could not be read)\n\n"
            for i, (title, url) in enumerate(failed_articles, 1): report_text_content += f"{i}. {title}\n   Link: {url}\n\n"
        if google_mentions:
            report_text_content += "\n--- Additional Mentions Found on Google News ---\n(Note: These links were not analyzed)\n\n"
            for i, (title, link) in enumerate(google_mentions, 1): report_text_content += f"{i}. {title}\n   Link: {link}\n\n"
        with st.spinner("Preparing and sending email report..."):
            output_filename = f"Report-{person_name.replace(' ','_')}-{from_date.strftime('%Y-%m-%d')}.txt"
            with open(output_filename, "w", encoding='utf-8') as f: f.write(report_text_content)
            email_subject = f"News & Sentiment Report for {person_name} on {from_date.strftime('%Y-%m-%d')}"
            email_body = f"Hi,\n\nPlease find the attached comprehensive news report for {person_name}."
            if send_email_with_attachment(email_subject, email_body, recipient_email, output_filename):
                st.success(f"✅ Report sent to {recipient_email}!")
            else: st.error("Failed to send email.")
            if os.path.exists(output_filename): os.remove(output_filename)
