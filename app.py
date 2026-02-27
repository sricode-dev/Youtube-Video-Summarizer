import os
import yt_dlp
from flask import Flask, render_template, request
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

app = Flask(__name__)

# =========================
# OpenRouter Client Setup
# =========================
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "http://localhost:5000",  # change in production
        "X-Title": "YouTube Metadata Summarizer"
    }
)

# =========================
# YouTube Metadata Fetcher
# =========================
def get_video_metadata(url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "title": info.get("title", ""),
        "channel": info.get("uploader", ""),
        "description": info.get("description", ""),
        "tags": info.get("tags", []) or []
    }

# =========================
# OpenRouter AI Processing
# =========================
def ai_summary_and_concepts(meta):
    prompt = f"""
You are an expert teacher.

Using ONLY the following YouTube video information,
generate a VERY DETAILED explanation as if teaching a student.

REQUIREMENTS:
1. Explain the video content in depth (like a lecture)
2. Break into clear sections with headings
3. Explain each concept in simple language
4. Use examples wherever possible
5. Write a LONG explanation (minimum 400–500 lines if possible)
6. Do NOT summarize shortly – explain fully

Video Information:
Title: {meta['title']}
Channel: {meta['channel']}
Tags: {', '.join(meta['tags'])}
Description: {meta['description']}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You explain topics in extreme detail like a teacher."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=3500
    )

    return response.choices[0].message.content

# =========================
# Flask Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    output = ""
    summary = ""
    error = ""

    if request.method == "POST":
        video_url = request.form.get("video_url")
        use_ai = request.form.get("use_ai")

        if not video_url:
            error = "Please enter a YouTube URL."
        else:
            try:
                meta = get_video_metadata(video_url)

                output = (
                    f"Title: {meta['title']}\n"
                    f"Channel: {meta['channel']}"
                )

                if use_ai:
                    summary = ai_summary_and_concepts(meta)

            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        output=output,
        summary=summary,
        error=error
    )

# =========================
# App Entry Point
# =========================
if __name__ == "__main__":
    app.run(debug=True)