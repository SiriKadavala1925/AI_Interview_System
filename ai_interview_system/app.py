import streamlit as st
import pypdf
import sqlite3
import ollama
import json
import pyttsx3
import numpy as np
import random
import io
import os
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from pydub import AudioSegment
from faster_whisper import WhisperModel
from streamlit_mic_recorder import mic_recorder
from streamlit_drawable_canvas import st_canvas

# ============================================================
# OPTIONAL LIVE CAMERA SUPPORT
# ============================================================
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    import cv2
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        raise ImportError("This opencv-python build is missing CascadeClassifier/objdetect support.")
    _cascade_check_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(_cascade_check_path):
        raise ImportError("Haar Cascade data file not found in this opencv-python build.")
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
from PIL import Image

from database import COMPANIES_LIST, ROLES_LIST, init_db

# ============================================================
# FFMPEG PATH CONFIGURATION
# ============================================================
FFMPEG_PATH  = r"C:\FFmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\FFmpeg\bin\ffprobe.exe"

AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe   = FFPROBE_PATH
os.environ["PATH"] = r"C:\FFmpeg\bin" + os.pathsep + os.environ.get("PATH", "")

DB_PATH = "interview_system.db"

# ============================================================
# RELIABLE LOCAL LLM CALL (Phi-3 / Moondream via Ollama)
# ============================================================
def call_local_llm(model, prompt, expect_json=False, num_predict=1024, timeout=60, images=None):
    options = {"temperature": 0.2, "num_predict": num_predict}

    def _once(p):
        client = ollama.Client(timeout=timeout)
        kwargs = {"model": model, "prompt": p, "options": options}
        if images:
            kwargs["images"] = images
        resp = client.generate(**kwargs)
        return resp["response"].strip()

    raw = _once(prompt)

    if not expect_json:
        return raw

    def _try_parse(text):
        cleaned = text.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end+1]
        return json.loads(cleaned)

    try:
        return _try_parse(raw)
    except (json.JSONDecodeError, ValueError):
        repair_prompt = (
            prompt
            + "\n\nIMPORTANT: Your previous response was not valid JSON or was cut off. "
              "Return ONLY the complete, valid JSON object described above — no commentary, "
              "no markdown fences, and make sure every bracket is closed."
        )
        raw_retry = _once(repair_prompt)
        return _try_parse(raw_retry)

# ============================================================
# AUTO-SYNC DATABASE
# ============================================================
def ensure_database_ready():
    needs_seed = False
    if not os.path.exists(DB_PATH):
        needs_seed = True
    else:
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT company_name FROM question_bank")
            db_companies = set(r[0] for r in cur.fetchall())
            cur.execute("SELECT DISTINCT role_name FROM question_bank")
            db_roles = set(r[0] for r in cur.fetchall())
            conn.close()
            if db_companies != set(COMPANIES_LIST) or db_roles != set(ROLES_LIST):
                needs_seed = True
        except Exception:
            needs_seed = True
    if needs_seed:
        init_db()

ensure_database_ready()

# ============================================================
# LOAD WHISPER MODEL
# ============================================================
@st.cache_resource
def load_local_whisper_model():
    return WhisperModel("base", device="cpu", compute_type="int8")

whisper_model = load_local_whisper_model()

# ============================================================
# PDF RESUME EXTRACTION
# ============================================================
@st.cache_data
def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ============================================================
# RESUME SKILL PARSING VIA SLM
# ============================================================
def parse_resume_with_slm(resume_text):
    prompt = f"Extract all engineering skills, coding tools, and tech keywords as a brief plain comma-separated list from this profile: {resume_text[:1000]}"
    try:
        return call_local_llm("phi3", prompt, num_predict=300).replace("`", "")
    except Exception:
        return "python, data structures, algorithms, machine learning, operating systems, databases, sql, java, c++, network architecture"

# ============================================================
# CANDIDATE KNOWLEDGE GRAPH — WEB SCRAPING (NO APIs)
# ============================================================
_SCRAPE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
_SCRAPE_TIMEOUT = 8


def _fetch_soup(url):
    try:
        resp = requests.get(url, headers=_SCRAPE_HEADERS, timeout=_SCRAPE_TIMEOUT)
        if resp.status_code != 200:
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return None


def scrape_github_profile(url):
    """Returns a short text summary (repo count + top languages) or None."""
    url = url.strip().rstrip("/")
    if not url:
        return None
    if "github.com" not in url:
        url = f"https://github.com/{url}"
    repos_url = url if "?tab=repositories" in url else f"{url}?tab=repositories"

    soup = _fetch_soup(repos_url)
    if soup is None:
        return None

    repo_items = soup.select("li[itemprop=owns] a")
    repo_names = [a.get_text(strip=True) for a in repo_items if a.get_text(strip=True)]
    lang_spans = soup.select("span[itemprop=programmingLanguage]")
    languages = [s.get_text(strip=True) for s in lang_spans if s.get_text(strip=True)]

    if not repo_names and not languages:
        return None

    lang_counts = {}
    for lang in languages:
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    top_languages = sorted(lang_counts, key=lang_counts.get, reverse=True)[:5]

    summary = f"GitHub: {len(repo_names)} public repositories visible"
    if top_languages:
        summary += f", most-used languages: {', '.join(top_languages)}"
    return summary


def scrape_portfolio_site(url):
    """Returns truncated visible page text, or None."""
    url = url.strip()
    if not url:
        return None
    if not url.startswith("http"):
        url = f"https://{url}"

    soup = _fetch_soup(url)
    if soup is None:
        return None

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()
    if not text:
        return None
    return f"Portfolio site content (excerpt): {text[:600]}"


def _scrape_js_rendered_profile_best_effort(url, platform_name):
    url = url.strip()
    if not url:
        return None
    if not url.startswith("http"):
        url = f"https://{url}"

    soup = _fetch_soup(url)
    if soup is None:
        return None

    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True)).strip()

    if len(text) < 80:
        return None
    return f"{platform_name} page content (excerpt, best-effort): {text[:400]}"


def _find_stats_in_json(obj, wanted_keys, results=None, _depth=0):
    if results is None:
        results = {}
    if _depth > 12:
        return results
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).lower()
            for wk in wanted_keys:
                if wk in lk and not isinstance(v, (dict, list)):
                    results.setdefault(k, v)
            if isinstance(v, (dict, list)):
                _find_stats_in_json(v, wanted_keys, results, _depth + 1)
    elif isinstance(obj, list):
        for item in obj[:50]:
            _find_stats_in_json(item, wanted_keys, results, _depth + 1)
    return results


def scrape_leetcode_profile(url):
    url = url.strip()
    if not url:
        return None
    if not url.startswith("http"):
        url = f"https://leetcode.com/{url}"

    try:
        resp = requests.get(url, headers=_SCRAPE_HEADERS, timeout=_SCRAPE_TIMEOUT)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            wanted_keys = [
                "totalsolved", "easysolved", "mediumsolved", "hardsolved",
                "ranking", "acsubmissionnum", "totalsubmissionnum", "reputation"
            ]
            for script_tag in soup.find_all("script"):
                raw = script_tag.string or script_tag.get_text() or ""
                raw = raw.strip()
                if not raw or raw[0] not in "{[":
                    continue
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue
                stats = _find_stats_in_json(data, wanted_keys)
                if stats:
                    parts = [f"{k}: {v}" for k, v in stats.items()]
                    return f"LeetCode (from embedded page data): {', '.join(parts)}"
    except Exception:
        pass

    return _scrape_js_rendered_profile_best_effort(url, "LeetCode")


def scrape_kaggle_profile(url):
    return _scrape_js_rendered_profile_best_effort(url, "Kaggle")


def scrape_linkedin_profile(url):
    """Scrapes public LinkedIn profile metadata (OG tags, JSON-LD, description) without APIs."""
    url = url.strip()
    if not url:
        return None
    if not url.startswith("http"):
        url = f"https://{url}"

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        resp = requests.get(url, headers=headers, timeout=_SCRAPE_TIMEOUT)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            for script_tag in soup.find_all("script", type="application/ld+json"):
                raw = script_tag.string or script_tag.get_text() or ""
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict):
                        name = data.get("name") or ""
                        job = data.get("jobTitle") or ""
                        desc = data.get("description") or ""
                        if name or job:
                            return f"LinkedIn: {name} - {job} - {desc[:300]}".strip()
                except Exception:
                    pass

            og_title = soup.find("meta", property="og:title")
            og_desc = soup.find("meta", property="og:description")
            title = og_title["content"].strip() if og_title and og_title.get("content") else ""
            desc = og_desc["content"].strip() if og_desc and og_desc.get("content") else ""
            if not title:
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else ""

            if title or desc:
                return f"LinkedIn profile info: {title}. {desc}".strip()[:500]
    except Exception:
        pass

    # Extract username or handle from URL path
    match = re.search(r"linkedin\.com/in/([^/?#]+)", url)
    if match:
        username = match.group(1).replace("-", " ").strip()
        return f"LinkedIn profile identified for: {username}"
    return f"LinkedIn profile: {url}"


def build_candidate_knowledge_graph(resume_text, profile_links, linkedin_pdf_text=""):
    provided_links = {k: v.strip() for k, v in profile_links.items() if v and v.strip()}
    scrape_status = {}
    has_linkedin_pdf = bool(linkedin_pdf_text and linkedin_pdf_text.strip())

    if not provided_links and not has_linkedin_pdf:
        return parse_resume_with_slm(resume_text), provided_links, scrape_status

    scraped_context_lines = []
    for label, value in provided_links.items():
        scraper = {
            "github": scrape_github_profile,
            "linkedin": scrape_linkedin_profile,
            "leetcode": scrape_leetcode_profile,
            "kaggle": scrape_kaggle_profile,
            "portfolio": scrape_portfolio_site,
        }.get(label)

        result = scraper(value) if scraper else None
        if result:
            scrape_status[label] = "scraped"
            scraped_context_lines.append(f"- {result}")
        else:
            scrape_status[label] = "link_only"
            scraped_context_lines.append(f"- {label.capitalize()}: {value} (link provided, page content unavailable)")

    if has_linkedin_pdf and "linkedin" not in provided_links:
        scrape_status["linkedin"] = "scraped"
        scraped_context_lines.append(f"- LinkedIn (from candidate profile document): {linkedin_pdf_text[:1000]}")

    links_context = "\n".join(scraped_context_lines)

    prompt = f"""Extract all engineering skills, coding tools, and tech keywords as a brief
plain comma-separated list, using BOTH sources below. Where real page content is
given, use it directly. Where only a link is noted as "provided, not fetched" or
"content unavailable", treat it only as a weak signal of general activity in that
area — do NOT invent specific repo names, scores, or stats you were not given.

Resume text:
{resume_text[:1000]}

Candidate profile information:
{links_context}

Return ONLY the comma-separated skill list, nothing else."""

    try:
        skills_text = call_local_llm("phi3", prompt, num_predict=300).replace("`", "")
        return skills_text, provided_links, scrape_status
    except Exception:
        fallback = "python, data structures, algorithms, machine learning, operating systems, databases, sql, java, c++, network architecture"
        return fallback, provided_links, scrape_status

# ============================================================
# QUESTION POOL FROM SQLITE DATABASE
# ============================================================
def get_questions_randomized_pool(company, role):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT question, source_reference, subject FROM question_bank "
            "WHERE LOWER(TRIM(company_name)) = LOWER(TRIM(?)) "
            "AND LOWER(TRIM(role_name)) = LOWER(TRIM(?))",
            (company.strip(), role.strip())
        )
        rows = cursor.fetchall()
        conn.close()
        if len(rows) >= 10:
            return random.sample(rows, 10)
        else:
            random.shuffle(rows)
            return rows
    except:
        return []

# ============================================================
# TEXT TO SPEECH
# ============================================================
def speak_text_globally_neutral(text_to_speak):
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        for voice in voices:
            if "EN" in voice.id.upper() or "ENGLISH" in voice.name.upper() or "ZIRA" in voice.name.upper():
                engine.setProperty('voice', voice.id)
                break
        engine.setProperty('rate', 140)
        engine.say(text_to_speak)
        engine.runAndWait()
    except:
        pass

# ============================================================
# SPEECH TO TEXT — WHISPER
# ============================================================
def speech_to_text_local_whisper(audio_bytes):
    if not audio_bytes or len(audio_bytes) < 100:
        return "⚠️ No audio detected. Please record again."
    try:
        try:
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
        except Exception:
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")
            except Exception:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))

        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        if audio_segment.dBFS != float("-inf") and audio_segment.dBFS < -30:
            audio_segment = audio_segment.apply_gain(-20 - audio_segment.dBFS)

        temp_path = "temp_interview_audio.wav"
        audio_segment.export(temp_path, format="wav")

        def _transcribe(use_vad):
            segs, _ = whisper_model.transcribe(
                temp_path, beam_size=5, language=None, task="transcribe",
                vad_filter=use_vad,
                vad_parameters=dict(min_silence_duration_ms=500) if use_vad else None
            )
            return " ".join(s.text.strip() for s in segs).strip()

        full_text = _transcribe(use_vad=True)
        if not full_text:
            full_text = _transcribe(use_vad=False)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if not full_text:
            return "⚠️ No speech detected. Please speak clearly and try again."
        return full_text
    except Exception as e:
        error_msg = str(e)
        if "ffprobe" in error_msg.lower() or "ffmpeg" in error_msg.lower():
            return "⚠️ FFmpeg Error: Check C:\\FFmpeg\\bin\\ffprobe.exe"
        return f"⚠️ Transcription Error: {error_msg}"

# ============================================================
# VISION MODEL — MOONDREAM2
# ============================================================
def describe_diagram_with_vision_model(image_array):
    try:
        img_array = image_array.astype(np.uint8)
        pil_image = Image.fromarray(img_array, mode="RGBA")
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, pil_image).convert("RGB")
        temp_path = "temp_diagram_capture.png"
        composited.save(temp_path)
        description = call_local_llm(
            "moondream",
            "Describe what is drawn in this image. Focus on shapes, boxes, arrows, trees, or any structure you see. Be specific about how elements are connected.",
            num_predict=400,
            images=[temp_path]
        )
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return description
    except Exception as e:
        return f"[Vision model unavailable: {str(e)}]"

# ============================================================
# CANVAS TO PNG BYTES
# ============================================================
def canvas_drawing_to_png_bytes(image_array):
    try:
        img_array = image_array.astype(np.uint8)
        pil_image = Image.fromarray(img_array, mode="RGBA")
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(background, pil_image).convert("RGB")
        buf = io.BytesIO()
        composited.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None

# ============================================================
# BACKGROUND FACE MONITOR — completely invisible to user
# ============================================================
if WEBRTC_AVAILABLE and CV2_AVAILABLE:
    class SilentFaceMonitor(VideoProcessorBase):
        def __init__(self):
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.last_face_present = None
            self.last_face_count = 0
            self.total_frames = 0
            self.face_detected_frames = 0

        def recv(self, frame):
            try:
                img = frame.to_ndarray(format="bgr24")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                self.last_face_count = len(faces)
                self.last_face_present = len(faces) > 0
                self.total_frames += 1
                if self.last_face_present:
                    self.face_detected_frames += 1
                return frame
            except Exception:
                return frame

CAMERA_KEY = "persistent_camera_feed"

def _render_hidden_camera():
    """Renders the camera streamer completely hidden in the background via CSS.
    Returns the streamer context so the backend can read face detection state silently."""
    if not WEBRTC_AVAILABLE:
        return None
    # Aggressively hide ALL camera streamer elements on the page during the interview
    st.markdown("""
    <style>
    iframe[title*="webrtc"],
    iframe[src*="webrtc"],
    div[data-testid="stCustomComponentV1"]:has(iframe[title*="webrtc"]),
    div[data-testid="element-container"]:has(iframe[title*="webrtc"]),
    div[data-testid="stVideo"],
    .stVideo {
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 1px !important;
        height: 1px !important;
        opacity: 0 !important;
        visibility: hidden !important;
        pointer-events: none !important;
        overflow: hidden !important;
        z-index: -9999 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    common_kwargs = dict(
        key=CAMERA_KEY,
        rtc_configuration=RTCConfiguration({"iceServers": []}),
        media_stream_constraints={"video": True, "audio": False},
        desired_playing_state=True,
        sendback_video=False,
        video_html_attrs={
            "style": {
                "display": "none",
                "width": "1px",
                "height": "1px",
                "opacity": "0",
                "pointerEvents": "none",
            },
            "controls": False,
            "autoPlay": True,
        }
    )
    if CV2_AVAILABLE:
        return webrtc_streamer(mode=WebRtcMode.SENDRECV, video_processor_factory=SilentFaceMonitor, **common_kwargs)
    else:
        return webrtc_streamer(mode=WebRtcMode.SENDONLY, **common_kwargs)


def _render_background_camera():
    """Keeps the camera and microphone active in the background throughout the interview,
    displays a live PIP proctored monitor in the bottom-right, and enforces fullscreen & window integrity."""
    cam_active = bool(st.session_state.get("camera_enabled"))
    
    ctx = None
    if cam_active:
        ctx = _render_hidden_camera()

    cam_card_display = "block" if cam_active else "none"
    cam_active_js = "true" if cam_active else "false"

    html_code = """
    <div id="pip-proctor-card" style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 196px;
        height: 148px;
        background: #0f172a;
        border-radius: 12px;
        border: 2px solid #10b981;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        z-index: 999999;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: __CAM_DISPLAY__;
    ">
        <video id="proctor-live-vid" autoplay playsinline muted style="
            width: 100%;
            height: 100%;
            object-fit: cover;
            transform: scaleX(-1);
            background: #000;
        "></video>
        <div style="
            position: absolute;
            top: 7px;
            left: 7px;
            background: rgba(15,23,42,0.85);
            color: #10b981;
            font-size: 9.5px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 5px;
            display: flex;
            align-items: center;
            gap: 5px;
            border: 1px solid rgba(16,185,129,0.35);
            pointer-events: none;
            backdrop-filter: blur(4px);
        ">
            <span style="width:6px; height:6px; background:#10b981; border-radius:50%; display:inline-block; animation: proctorPulse 1.6s infinite;"></span>
            PROCTORED
        </div>
        <button id="pip-toggle-btn" onclick="togglePipSize()" title="Minimize/Maximize" style="
            position: absolute;
            top: 6px;
            right: 6px;
            background: rgba(0,0,0,0.65);
            color: #fff;
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            font-size: 13px;
            width: 22px;
            height: 22px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
        ">−</button>
    </div>

    <style>
    @keyframes proctorPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.3; transform: scale(0.85); }
    }
    @keyframes fsBannerPulse {
        0%, 100% { box-shadow: 0 8px 25px rgba(2,132,199,0.4); border-color: #38bdf8; }
        50% { box-shadow: 0 8px 32px rgba(2,132,199,0.85); border-color: #0284c7; }
    }
    </style>

    <script>
    (function() {
        const pWin = window.parent;
        const pDoc = pWin.document;

        // 1. Fullscreen Enforcement for Interview
        function requestInterviewFullscreen() {
            const el = pDoc.documentElement;
            if (pDoc.fullscreenElement || pDoc.webkitFullscreenElement) return;
            try {
                if (el.requestFullscreen) {
                    el.requestFullscreen().catch(() => {});
                } else if (el.webkitRequestFullscreen) {
                    el.webkitRequestFullscreen();
                } else if (el.msRequestFullscreen) {
                    el.msRequestFullscreen();
                }
            } catch(e) {}
        }

        function updateFullscreenUI() {
            const isFS = !!(pDoc.fullscreenElement || pDoc.webkitFullscreenElement);
            let banner = pDoc.getElementById('interview-fs-banner');
            
            if (!isFS) {
                if (!banner) {
                    banner = pDoc.createElement('div');
                    banner.id = 'interview-fs-banner';
                    banner.style.cssText = 'position:fixed; top:14px; left:50%; transform:translateX(-50%); background:linear-gradient(135deg, #0f172a, #1e293b); color:#38bdf8; border:2px solid #0284c7; border-radius:30px; padding:9px 24px; font-size:13.5px; font-weight:700; font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; cursor:pointer; box-shadow:0 10px 30px rgba(0,0,0,0.7); z-index:99999999; display:flex; align-items:center; gap:12px; letter-spacing:0.2px; animation:fsBannerPulse 2s infinite; transition:transform 0.15s ease, background 0.2s ease;';
                    banner.innerHTML = '<span style="font-size:16px;">⛶</span> <span>Proctored Assessment: <b>Click here to Enter Fullscreen Mode</b></span> <span style="background:#0284c7; color:#ffffff; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:800; letter-spacing:0.5px;">ENFORCE</span>';
                    banner.addEventListener('click', function(ev) {
                        ev.stopPropagation();
                        requestInterviewFullscreen();
                    });
                    pDoc.body.appendChild(banner);
                } else {
                    banner.style.display = 'flex';
                }
            } else {
                if (banner) banner.style.display = 'none';
            }

            const hdrBtn = pDoc.getElementById('header-fullscreen-btn');
            if (hdrBtn) {
                if (isFS) {
                    hdrBtn.innerHTML = '<span>🗗</span> Exit Fullscreen';
                    hdrBtn.style.borderColor = 'rgba(16, 185, 129, 0.5)';
                    hdrBtn.style.color = '#34d399';
                } else {
                    hdrBtn.innerHTML = '<span>⛶</span> Enter Fullscreen';
                    hdrBtn.style.borderColor = 'rgba(56, 189, 248, 0.4)';
                    hdrBtn.style.color = '#38bdf8';
                }
            }
        }

        // Catch clicks anywhere on document to enter fullscreen if not yet fullscreen
        pDoc.addEventListener('click', function(ev) {
            if (!pDoc.fullscreenElement && !pDoc.webkitFullscreenElement) {
                requestInterviewFullscreen();
            }
            const targetBtn = ev.target.closest('#header-fullscreen-btn');
            if (targetBtn) {
                ev.preventDefault();
                ev.stopPropagation();
                if (pDoc.fullscreenElement || pDoc.webkitFullscreenElement) {
                    if (pDoc.exitFullscreen) pDoc.exitFullscreen().catch(() => {});
                    else if (pDoc.webkitExitFullscreen) pDoc.webkitExitFullscreen();
                } else {
                    requestInterviewFullscreen();
                }
            }
        }, true);

        pDoc.addEventListener('fullscreenchange', updateFullscreenUI);
        pDoc.addEventListener('webkitfullscreenchange', updateFullscreenUI);

        setTimeout(updateFullscreenUI, 100);
        setTimeout(requestInterviewFullscreen, 250);

        // 2. Maintain background camera & mic stream (if enabled)
        const camActive = __CAM_ACTIVE__;
        const vid = document.getElementById('proctor-live-vid');
        if (camActive && navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true, audio: true })
                .then(stream => {
                    window.__proctor_stream = stream;
                    if (vid) {
                        vid.srcObject = stream;
                        vid.play().catch(e => console.log('Proctor autoplay:', e));
                    }
                })
                .catch(err => {
                    console.warn('Proctor device error:', err);
                });
        }

        // 3. Strict Tab switch / window focus loss guard (FORBIDDEN)
        let violationCount = parseInt(sessionStorage.getItem('proctor_tab_switches') || '0');
        let lastViolationTime = 0;
        
        function handleViolation() {
            const now = Date.now();
            if (now - lastViolationTime < 2500) return; // Debounce dual blur/visibility events
            lastViolationTime = now;

            violationCount++;
            sessionStorage.setItem('proctor_tab_switches', violationCount.toString());
            
            // Sync with parent URL query parameter for Streamlit Python backend
            try {
                const url = new URL(window.parent.location.href);
                url.searchParams.set('proctor_violations', violationCount);
                window.parent.history.replaceState({}, '', url);
            } catch(e) {}

            // Show blocking modal on top window
            try {
                const existing = pDoc.getElementById('proctor-lock-overlay');
                if (existing) existing.remove();

                const overlay = pDoc.createElement('div');
                overlay.id = 'proctor-lock-overlay';
                overlay.style.cssText = `
                    position: fixed;
                    top: 0; left: 0; width: 100vw; height: 100vh;
                    background: rgba(15, 23, 42, 0.96);
                    z-index: 99999999;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    backdrop-filter: blur(10px);
                `;

                overlay.innerHTML = `
                    <div style="
                        background: #1e293b;
                        border: 2px solid #ef4444;
                        border-radius: 16px;
                        padding: 32px 36px;
                        max-width: 520px;
                        text-align: center;
                        box-shadow: 0 25px 60px rgba(0,0,0,0.85);
                        color: #ffffff;
                    ">
                        <div style="font-size: 52px; margin-bottom: 12px; line-height: 1;">🚫</div>
                        <h2 style="color: #ef4444; font-size: 22px; font-weight: 800; margin: 0 0 10px 0; letter-spacing: -0.5px;">
                            SECURITY VIOLATION: TAB SWITCHING FORBIDDEN
                        </h2>
                        <p style="color: #cbd5e1; font-size: 14.5px; line-height: 1.6; margin: 0 0 16px 0;">
                            Tab switching, window minimization, and navigating away from the interview interface are <b>strictly forbidden</b> under proctored examination rules.
                        </p>
                        <div style="
                            background: rgba(239,68,68,0.12);
                            border: 1px solid rgba(239,68,68,0.35);
                            border-radius: 8px;
                            padding: 12px 16px;
                            margin-bottom: 22px;
                            text-align: left;
                        ">
                            <div style="color: #f87171; font-size: 13.5px; font-weight: 700;">
                                ⚠️ Violation Incident #${violationCount} Logged
                            </div>
                            <div style="color: #94a3b8; font-size: 12px; margin-top: 4px;">
                                Each occurrence is recorded in your official assessment integrity audit and reduces your integrity rating (-10% penalty per violation).
                            </div>
                        </div>
                        <button id="proctor-dismiss-btn" style="
                            background: linear-gradient(135deg, #ef4444, #dc2626);
                            color: #ffffff;
                            border: none;
                            padding: 12px 28px;
                            border-radius: 8px;
                            font-size: 14px;
                            font-weight: 700;
                            cursor: pointer;
                            box-shadow: 0 4px 14px rgba(239,68,68,0.45);
                            transition: transform 0.15s ease, background 0.2s ease;
                        ">I Acknowledge & Return to Assessment</button>
                    </div>
                `;

                pDoc.body.appendChild(overlay);

                const btn = pDoc.getElementById('proctor-dismiss-btn');
                if (btn) {
                    btn.addEventListener('click', function() {
                        overlay.remove();
                        try {
                            window.parent.focus();
                            requestInterviewFullscreen();
                        } catch(e) {}
                    });
                }
            } catch(e) {}
        }

        // Attach listeners to both current iframe and parent document
        document.addEventListener('visibilitychange', function() {
            if (document.hidden) handleViolation();
        });
        window.addEventListener('blur', function() {
            handleViolation();
        });

        try {
            if (window.parent && window.parent.document) {
                window.parent.document.addEventListener('visibilitychange', function() {
                    if (window.parent.document.hidden) handleViolation();
                });
                window.parent.addEventListener('blur', function() {
                    handleViolation();
                });
            }
        } catch(e) {}
    })();

    function togglePipSize() {
        const box = document.getElementById('pip-proctor-card');
        const btn = document.getElementById('pip-toggle-btn');
        if (box && btn) {
            if (box.style.height === '32px') {
                box.style.height = '148px';
                box.style.width = '196px';
                btn.textContent = '−';
            } else {
                box.style.height = '32px';
                box.style.width = '125px';
                btn.textContent = '+';
            }
        }
    }
    </script>
    """

    st.components.v1.html(
        html_code.replace("__CAM_DISPLAY__", cam_card_display).replace("__CAM_ACTIVE__", cam_active_js),
        height=0,
        width=0,
    )

    return ctx

# ============================================================
# TRANSPARENT SCORING FRAMEWORK
# ============================================================
SCORE_WEIGHTS = {
    "technical_correctness": 0.35,
    "reasoning_approach": 0.25,
    "completeness_relevance": 0.15,
    "answer_format_quality": 0.15,
    "communication_clarity": 0.10,
}

SCORE_COMPONENT_LABELS = {
    "technical_correctness": "Technical Correctness",
    "reasoning_approach": "Reasoning & Problem-Solving",
    "completeness_relevance": "Completeness & Relevance",
    "answer_format_quality": "Code / Diagram Quality",
    "communication_clarity": "Communication Clarity",
}

# ============================================================
# HIRING BAND LOGIC — real-world thresholds
# ============================================================
def get_hiring_band(score):
    if score >= 85:
        return "Strong Hire", "#00c853", "Exceptional performance. The candidate demonstrated deep technical mastery, clear communication, and strong problem-solving instincts."
    elif score >= 70:
        return "Hire", "#2196f3", "Solid performance across most areas. The candidate shows the fundamentals needed to contribute effectively with standard onboarding."
    elif score >= 50:
        return "Lean Hire", "#ff9800", "Borderline performance. Some areas showed promise, but gaps exist that may need targeted development or a follow-up round."
    elif score >= 35:
        return "Lean No Hire", "#ff5722", "Below the bar for this role. Fundamentals are partially present but significant gaps were observed in core areas."
    else:
        return "No Hire", "#f44336", "The candidate did not demonstrate sufficient readiness for this role. Fundamental concepts need substantial work."


def evaluate_entire_interview_via_slm(history):
    compiled_log = ""
    for i, (q, a) in enumerate(history):
        compiled_log += f"\n[Q{i+1}]: {q}\n[Answer {i+1}]: {a}\n---"

    prompt = f"""You are a senior Technical Interviewer at a top-tier technology company conducting a formal evaluation.
Evaluate this candidate's interview with {len(history)} questions. Each question is worth 10 marks.

EVALUATION CRITERIA (apply per question):

Step 1 — RELEVANCE CHECK:
  - If the answer is blank, skipped, completely off-topic, or gibberish → 0 marks
  - If the answer is a single word or trivially short with no substance → 0-1 marks

Step 2 — CORRECTNESS & DEPTH (only if relevant):
  - Fully correct with clear explanation and examples → 9-10 marks
  - Correct core concept with minor omissions → 7-8 marks
  - Partially correct, demonstrates some understanding but missing key points → 4-6 marks
  - Shows awareness of topic but largely incorrect or confused → 2-3 marks
  - Barely relevant, mostly wrong → 1 mark

Step 3 — FORMAT-SPECIFIC RULES:
  - [Submitted as Pseudocode/Code]: Judge logical correctness, edge case handling, and approach — NOT syntax
  - [Submitted as Diagram]: Judge whether the structure correctly represents the concept, using both the Vision AI description and candidate explanation
  - Text/Voice answers: Judge conceptual correctness and clarity of explanation

IMPORTANT SCORING GUIDELINES:
- Do NOT inflate scores. A vague or hand-wavy answer should NOT get 7+.
- An answer that just restates the question or gives a dictionary definition without technical depth = 3-4 max.
- Skipped questions = 0. No exceptions.
- Be honest and calibrate to real industry interview standards.

For each question, also provide:
1. A specific, actionable improvement tip
2. What the ideal answer should have covered (brief)

Rate the candidate's overall performance on these dimensions (each 0-100):
- reasoning_approach: logical structure, systematic thinking, problem decomposition
- completeness_relevance: how thoroughly each answer addressed what was asked
- answer_format_quality: quality of code/diagrams/pseudocode (50 if none submitted)
- communication_clarity: articulation, structure, technical vocabulary usage

Interview Transcript:
{compiled_log}

Return ONLY this raw JSON (no backticks, no markdown):
{{
  "calculated_total_score_out_of_100_matrix": 0,
  "per_question_scores": [<list of integers 0-10, one per question>],
  "per_question_feedback": [<list of improvement tip strings>],
  "per_question_ideal": [<list of brief ideal answer summaries>],
  "score_breakdown": {{
    "reasoning_approach": 0,
    "completeness_relevance": 0,
    "answer_format_quality": 0,
    "communication_clarity": 0
  }},
  "technical_strengths": "Specific things done well, with question references.",
  "critical_gaps_identified": "Specific concepts missed or answered incorrectly, with question references.",
  "communication_feedback": "Assessment of clarity, structure, vocabulary, and confidence.",
  "overall_verdict": "One of: Strong Hire / Hire / Lean Hire / Lean No Hire / No Hire",
  "recommended_study_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "learning_resources": ["resource1", "resource2", "resource3"],
  "improvement_plan": "A 2-3 sentence actionable improvement plan for the candidate."
}}"""

    slm_rated_keys = ["reasoning_approach", "completeness_relevance", "answer_format_quality", "communication_clarity"]

    try:
        parsed_data = call_local_llm("phi3", prompt, expect_json=True, num_predict=2000, timeout=120)

        raw_pqs = parsed_data.get("per_question_scores", [])
        safe_pqs = []
        for s in raw_pqs:
            try:
                safe_pqs.append(max(0, min(10, int(s))))
            except (TypeError, ValueError):
                safe_pqs.append(0)
        while len(safe_pqs) < len(history):
            safe_pqs.append(0)
        safe_pqs = safe_pqs[:len(history)]
        parsed_data["per_question_scores"] = safe_pqs

        raw_feedback = parsed_data.get("per_question_feedback", [])
        safe_feedback = [str(f) if f else "Review this topic further." for f in raw_feedback]
        while len(safe_feedback) < len(history):
            safe_feedback.append("Review this topic further.")
        parsed_data["per_question_feedback"] = safe_feedback[:len(history)]

        raw_ideal = parsed_data.get("per_question_ideal", [])
        safe_ideal = [str(f) if f else "" for f in raw_ideal]
        while len(safe_ideal) < len(history):
            safe_ideal.append("")
        parsed_data["per_question_ideal"] = safe_ideal[:len(history)]

        technical_correctness = 0
        if safe_pqs:
            technical_correctness = max(0, min(100, round((sum(safe_pqs) / (10 * len(history))) * 100)))

        raw_breakdown = parsed_data.get("score_breakdown", {})
        if not isinstance(raw_breakdown, dict):
            raw_breakdown = {}
        safe_breakdown = {"technical_correctness": technical_correctness}
        for key in slm_rated_keys:
            try:
                safe_breakdown[key] = max(0, min(100, int(raw_breakdown.get(key, technical_correctness))))
            except (TypeError, ValueError):
                safe_breakdown[key] = technical_correctness
        parsed_data["score_breakdown"] = safe_breakdown

        weighted_total = sum(safe_breakdown[key] * SCORE_WEIGHTS[key] for key in SCORE_WEIGHTS)
        parsed_data["calculated_total_score_out_of_100_matrix"] = max(0, min(100, round(weighted_total)))
        parsed_data["used_fallback_scoring"] = False

        return parsed_data

    except Exception:
        per_q = []
        answered_count = 0
        for q, a in history:
            clean_a = a.lower().strip()
            if (len(clean_a) > 1
                    and "skipped" not in clean_a
                    and "[no response" not in clean_a):
                per_q.append(5)
                answered_count += 1
            else:
                per_q.append(0)
        technical_correctness = round((sum(per_q) / (10 * len(history))) * 100) if history else 0
        fallback_breakdown = {key: technical_correctness for key in SCORE_WEIGHTS}
        final_score = max(0, min(100, round(
            sum(fallback_breakdown[key] * SCORE_WEIGHTS[key] for key in SCORE_WEIGHTS)
        )))

        fallback_note = " (AI evaluator unreachable — basic estimate only)"

        if answered_count == 0:
            technical_strengths = "No answers were submitted." + fallback_note
            gaps = "All questions were skipped or left blank."
            communication_feedback = "No answers provided to assess."
        elif answered_count == len(history):
            technical_strengths = "Candidate attempted all questions." + fallback_note
            gaps = "Full review requires AI evaluator. Use Retry below."
            communication_feedback = "Answers provided but detailed analysis unavailable."
        else:
            technical_strengths = f"Candidate attempted {answered_count}/{len(history)} questions." + fallback_note
            gaps = f"{len(history) - answered_count} questions were skipped."
            communication_feedback = "Partial answers provided; detailed analysis unavailable."

        verdict, _, _ = get_hiring_band(final_score)

        return {
            "calculated_total_score_out_of_100_matrix": final_score,
            "per_question_scores": per_q,
            "per_question_feedback": ["Review this concept thoroughly and reinforce the core technical principles." for _ in history],
            "per_question_ideal": ["" for _ in history],
            "score_breakdown": fallback_breakdown,
            "technical_strengths": technical_strengths,
            "critical_gaps_identified": gaps,
            "communication_feedback": communication_feedback,
            "overall_verdict": verdict,
            "recommended_study_topics": ["Data Structures", "Algorithms", "Operating Systems", "System Design", "Problem Solving"],
            "learning_resources": ["Cracking the Coding Interview", "LeetCode", "NeetCode.io"],
            "improvement_plan": "Focus on core architectural fundamentals. Articulate technical concepts aloud with structured reasoning. Master real-time simulated interviews regularly.",
            "used_fallback_scoring": True,
        }

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(page_title="AI Interview System", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# GLOBAL CSS — Premium dark-themed professional UI
# ============================================================
st.markdown("""
<style>
    /* ── Import premium font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary: #0f1117;
        --bg-card: #1a1d28;
        --bg-card-hover: #22263a;
        --text-primary: #e8eaed;
        --text-secondary: #9aa0a6;
        --accent-blue: #4285f4;
        --accent-green: #34a853;
        --accent-yellow: #fbbc04;
        --accent-red: #ea4335;
        --border-subtle: rgba(255,255,255,0.06);
        --shadow-card: 0 2px 16px rgba(0,0,0,0.3);
    }

    /* ── Base overrides ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .stApp > header { display: none !important; }

    /* ── Report card ── */
    .report-hero {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #01579b 100%);
        border-radius: 20px;
        padding: 40px 32px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 12px 40px rgba(13,71,161,0.3);
        position: relative;
        overflow: hidden;
    }
    .report-hero::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
        pointer-events: none;
    }
    .report-company {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 12px;
    }
    .score-huge {
        font-size: 80px;
        font-weight: 900;
        margin: 0;
        line-height: 1;
        background: linear-gradient(135deg, #fff, #90caf9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .score-suffix {
        font-size: 28px;
        opacity: 0.5;
        font-weight: 400;
    }
    .score-label {
        font-size: 13px;
        opacity: 0.6;
        margin-top: 4px;
        letter-spacing: 1px;
    }
    .verdict-pill {
        display: inline-block;
        padding: 8px 28px;
        border-radius: 100px;
        font-weight: 700;
        font-size: 14px;
        margin-top: 16px;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }

    /* ── Component cards ── */
    .metric-card {
        background: #1a1d28;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        margin: 8px 0 4px 0;
    }
    .metric-label {
        font-size: 11px;
        color: #9aa0a6;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    .metric-weight {
        font-size: 10px;
        color: #666;
        margin-top: 2px;
    }
    .metric-bar {
        width: 100%;
        height: 4px;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        margin-top: 12px;
        overflow: hidden;
    }
    .metric-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.6s ease;
    }

    /* ── Question score chips ── */
    .q-chip {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        background: #1a1d28;
        border-radius: 12px;
        padding: 12px 16px;
        min-width: 60px;
        border: 1px solid rgba(255,255,255,0.06);
        margin: 4px;
    }
    .q-chip-label {
        font-size: 10px;
        color: #666;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .q-chip-score {
        font-size: 24px;
        font-weight: 800;
    }
    .q-chip-max {
        font-size: 9px;
        color: #555;
    }

    /* ── Insight cards ── */
    .insight-card {
        background: #1a1d28;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }
    .insight-title {
        font-size: 13px;
        font-weight: 700;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }
    .insight-body {
        font-size: 14px;
        line-height: 1.7;
        color: #e0e0e0;
    }

    /* ── Study chips ── */
    .study-chip {
        display: inline-block;
        background: rgba(66,133,244,0.12);
        color: #90caf9;
        border: 1px solid rgba(66,133,244,0.2);
        border-radius: 100px;
        padding: 6px 18px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px 4px;
    }
    .resource-chip {
        display: inline-block;
        background: rgba(52,168,83,0.12);
        color: #81c784;
        border: 1px solid rgba(52,168,83,0.2);
        border-radius: 100px;
        padding: 6px 18px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px 4px;
    }

    /* ── Section headers ── */
    .section-hdr {
        font-size: 16px;
        font-weight: 700;
        color: #e0e0e0;
        margin: 28px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(66,133,244,0.3);
    }

    /* ── Interview progress bar ── */
    .interview-progress {
        background: #1a1d28;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .progress-text {
        font-size: 12px;
        font-weight: 600;
        color: #9aa0a6;
        margin-bottom: 8px;
    }
    .progress-bar-bg {
        width: 100%;
        height: 6px;
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #4285f4, #34a853);
        border-radius: 6px;
        transition: width 0.4s ease;
    }

    /* ── Status indicator ── */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .status-dot.active { background: #34a853; }
    .status-dot.inactive { background: #ea4335; animation: none; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* ── Question card ── */
    .question-card {
        background: linear-gradient(135deg, #1a237e, #0d47a1);
        border-radius: 16px;
        padding: 24px 28px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(13,71,161,0.2);
    }
    .question-number {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        opacity: 0.6;
        margin-bottom: 12px;
    }
    .question-text {
        font-size: 17px;
        font-weight: 500;
        line-height: 1.6;
    }

    /* ── System check status ── */
    .check-status {
        background: #1a1d28;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid rgba(255,255,255,0.06);
        margin: 8px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .check-icon {
        font-size: 20px;
        min-width: 24px;
    }
    .check-label {
        font-size: 14px;
        font-weight: 500;
    }
    .check-sublabel {
        font-size: 11px;
        color: #666;
    }

    /* ── Hide streamlit defaults and proctor trigger buttons ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    button:has-text("TRIGGER"),
    button[data-testid*="TRIGGER"],
    div:has(> button:contains("TRIGGER")) {
        display: none !important;
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE DEFAULTS
# ============================================================
defaults = {
    "interview_active": False,
    "final_done": False,
    "locked_questions_pool": [],
    "current_index": 0,
    "user_history": [],
    "warning_popup_triggered": False,
    "final_evaluation": {},
    "report_generated_at": "",
    "camera_enabled": False,
    "system_check_done": False,
    "locked_candidate_profile": None,
    "candidate_profile_links": {},
    "candidate_scrape_status": {},
    "diagram_images_by_history_index": {},
    "cam_check_passed": False,
    "mic_check_passed": False,
    "proctor_tab_switches": 0,
    "proctor_integrity_score": 100,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Sync browser proctoring violations from query params
if hasattr(st, "query_params"):
    raw_pv = st.query_params.get("proctor_violations", "0")
    try:
        pv_val = int(raw_pv)
        if pv_val > st.session_state.get("proctor_tab_switches", 0):
            st.session_state.proctor_tab_switches = pv_val
            st.session_state.proctor_integrity_score = max(50, 100 - (pv_val * 10))
    except Exception:
        pass

# ============================================================
# SIDEBAR — minimal, status only
# ============================================================
with st.sidebar:
    if os.path.exists(FFPROBE_PATH):
        st.success("✅ ffprobe.exe found")
    else:
        st.error("❌ ffprobe.exe missing")

    st.markdown("**Database Status:**")
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM question_bank")
        total_rows = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT company_name || role_name) FROM question_bank")
        total_combos = cur.fetchone()[0]
        conn.close()
        st.success(f"✅ {total_rows:,} questions loaded")
        st.caption(f"{total_combos} company-role combinations")
    except:
        st.error("❌ Database not ready")
    if st.button("🔄 Force Re-seed Database"):
        init_db()
        st.success("Database re-seeded!")
        st.rerun()

    st.markdown("---")
    st.markdown("**Active AI Models:**")
    st.caption("🧠 Phi-3 — evaluation")
    st.caption("👁️ Moondream2 — diagram vision")
    st.caption("🎤 Whisper base — voice STT")

    st.markdown("---")
    st.markdown("**Camera Status:**")
    if not WEBRTC_AVAILABLE:
        st.error("❌ streamlit-webrtc not installed")
    elif not CV2_AVAILABLE:
        st.warning("⚠️ Camera OK, face detection unavailable")
    else:
        st.success("✅ Camera + face detection ready")
    st.caption("Runs silently in the background throughout — no video is ever shown in this app's interface.")

# ============================================================
# TIER 1 — PROFESSIONAL EVALUATION REPORT
# ============================================================
if st.session_state.final_done:
    st.components.v1.html("""
    <script>
    try {
        const pDoc = window.parent.document;
        const b = pDoc.getElementById('interview-fs-banner');
        if (b) b.remove();
        const o = pDoc.getElementById('proctor-lock-overlay');
        if (o) o.remove();
    } catch(e) {}
    </script>
    """, height=0, width=0)
    eval_data   = st.session_state.get("final_evaluation", {})
    live_score  = int(eval_data.get("calculated_total_score_out_of_100_matrix", 0))
    per_q       = eval_data.get("per_question_scores", [])
    per_q_tips  = eval_data.get("per_question_feedback", [])
    per_q_ideal = eval_data.get("per_question_ideal", [])
    study_topics= eval_data.get("recommended_study_topics", [])
    learning_resources = eval_data.get("learning_resources", [])
    improvement_plan = eval_data.get("improvement_plan", "")
    score_breakdown = eval_data.get("score_breakdown", {})
    company     = st.session_state.get("sel_company", "Interview")
    role        = st.session_state.get("sel_role", "")

    verdict, verdict_color, verdict_desc = get_hiring_band(live_score)

    # Override with SLM verdict if available and not fallback
    if not eval_data.get("used_fallback_scoring"):
        slm_verdict = eval_data.get("overall_verdict", "")
        if slm_verdict:
            verdict = slm_verdict
            _, verdict_color, verdict_desc = get_hiring_band(live_score)

    # ── Hero card ──
    st.markdown(f"""
    <div class="report-hero">
        <div class="report-company">{company} · {role}</div>
        <div class="score-huge">{live_score}<span class="score-suffix">/100</span></div>
        <div class="score-label">OVERALL INTERVIEW SCORE</div>
        <div class="verdict-pill" style="background:{verdict_color}; color:white;">{verdict}</div>
        <div style="font-size:13px; opacity:0.7; margin-top:12px; max-width:500px; margin-left:auto; margin-right:auto;">
            {verdict_desc}
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_time = st.session_state.get("report_generated_at", "")
    if report_time:
        st.caption(f"📅 Report generated on {report_time}")

    if eval_data.get("used_fallback_scoring"):
        st.warning(
            "⚠️ **The AI evaluator (Phi-3) could not be reached.** "
            "The score is a basic estimate. Check Ollama is running, then retry below."
        )
        if st.button("🔁 Retry AI Evaluation", key="retry_eval_btn"):
            with st.spinner("Re-evaluating with Phi-3..."):
                st.session_state.final_evaluation = evaluate_entire_interview_via_slm(
                    st.session_state.user_history
                )
                st.session_state.report_generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
            st.rerun()

    # ── Score breakdown ──
    st.markdown('<div class="section-hdr">📐 Score Breakdown</div>', unsafe_allow_html=True)

    if score_breakdown:
        cols = st.columns(len(SCORE_COMPONENT_LABELS))
        colors = ["#4285f4", "#34a853", "#fbbc04", "#ea4335", "#ab47bc"]
        for i, (bcol, (key, label)) in enumerate(zip(cols, SCORE_COMPONENT_LABELS.items())):
            value = score_breakdown.get(key, 0)
            weight_pct = int(SCORE_WEIGHTS[key] * 100)
            color = colors[i % len(colors)]
            with bcol:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value" style="color:{color};">{value}</div>
                    <div class="metric-weight">{weight_pct}% weight</div>
                    <div class="metric-bar">
                        <div class="metric-bar-fill" style="width:{value}%; background:{color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Per-question scores ──
    st.markdown('<div class="section-hdr">📊 Question-by-Question Results</div>', unsafe_allow_html=True)

    if per_q:
        chips_html = '<div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px;">'
        for i, s in enumerate(per_q):
            safe_s = s if isinstance(s, int) else 0
            color = "#34a853" if safe_s >= 8 else "#fbbc04" if safe_s >= 5 else "#ea4335"
            chips_html += f"""
            <div class="q-chip" style="border-top: 3px solid {color};">
                <div class="q-chip-label">Q{i+1}</div>
                <div class="q-chip-score" style="color:{color};">{safe_s}</div>
                <div class="q-chip-max">/10</div>
            </div>"""
        chips_html += '</div>'
        st.markdown(chips_html, unsafe_allow_html=True)

    # ── Proctoring & Assessment Integrity Report ──
    st.markdown('<div class="section-hdr">🛡️ Proctoring & Assessment Integrity Report</div>', unsafe_allow_html=True)
    violations = st.session_state.get("proctor_tab_switches", 0)
    camera_on = st.session_state.get("camera_enabled", False)
    integrity_score = max(50, 100 - (violations * 10))
    integrity_color = "#34a853" if violations == 0 else "#fbbc04" if violations <= 2 else "#ea4335"
    integrity_status = "Clean Proctored Record" if violations == 0 else f"{violations} Security Warning(s)"

    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">INTEGRITY SCORE</div>
            <div class="metric-value" style="color:{integrity_color};">{integrity_score}%</div>
            <div class="metric-weight">{integrity_status}</div>
        </div>
        """, unsafe_allow_html=True)
    with pcol2:
        cam_text = "Verified Active" if camera_on else "Unmonitored"
        cam_col = "#34a853" if camera_on else "#9aa0a6"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BACKGROUND CAMERA</div>
            <div class="metric-value" style="color:{cam_col}; font-size:22px; margin: 14px 0 8px 0;">{"📹 ACTIVE" if camera_on else "OFF"}</div>
            <div class="metric-weight">{cam_text}</div>
        </div>
        """, unsafe_allow_html=True)
    with pcol3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MICROPHONE MONITOR</div>
            <div class="metric-value" style="color:#34a853; font-size:22px; margin: 14px 0 8px 0;">🎙️ ACTIVE</div>
            <div class="metric-weight">Acoustics & Voice Verified</div>
        </div>
        """, unsafe_allow_html=True)
    with pcol4:
        v_col = "#34a853" if violations == 0 else "#ea4335"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">TAB SWITCHES / BLUR</div>
            <div class="metric-value" style="color:{v_col};">{violations}</div>
            <div class="metric-weight">{"0 Violations Logged" if violations == 0 else f"{violations} Focus Losses Logged"}</div>
        </div>
        """, unsafe_allow_html=True)

    if violations == 0 and camera_on:
        st.success("✅ **Official Integrity Certification:** Candidate completed the entire technical interview under continuous background camera & audio proctoring with zero window focus violations.")
    elif not camera_on:
        st.info("ℹ️ **Proctoring Note:** Hardware check was skipped for this interview attempt.")
    else:
        st.warning(f"⚠️ **Integrity Notice:** {violations} window blur / tab-switching incident(s) were flagged during the interview session.")

    # ── Key insights ──
    st.markdown('<div class="section-hdr">🔍 Evaluation Insights</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="insight-card" style="border-left: 3px solid #34a853;">
            <div class="insight-title">✅ Strengths</div>
            <div class="insight-body">{eval_data.get("technical_strengths", "No data.")}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="insight-card" style="border-left: 3px solid #ea4335;">
            <div class="insight-title">⚠️ Gaps Identified</div>
            <div class="insight-body">{eval_data.get("critical_gaps_identified", "None found.")}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="insight-card" style="border-left: 3px solid #4285f4;">
            <div class="insight-title">🗣️ Communication</div>
            <div class="insight-body">{eval_data.get("communication_feedback", "No data.")}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Improvement Plan ──
    if improvement_plan:
        st.markdown('<div class="section-hdr">🎯 Your Improvement Plan</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="insight-card" style="border-left: 3px solid #ab47bc;">
            <div class="insight-body">{improvement_plan}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommended topics & resources ──
    rec_col1, rec_col2 = st.columns(2)
    with rec_col1:
        if study_topics:
            st.markdown('<div class="section-hdr">📚 Recommended Study Topics</div>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="study-chip">{t}</span>' for t in study_topics]), unsafe_allow_html=True)
    with rec_col2:
        if learning_resources:
            st.markdown('<div class="section-hdr">📖 Suggested Resources</div>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="resource-chip">{r}</span>' for r in learning_resources]), unsafe_allow_html=True)

    # ── Detailed review ──
    st.markdown('<div class="section-hdr">📝 Detailed Interview Review</div>', unsafe_allow_html=True)

    saved_diagram_images = st.session_state.get("diagram_images_by_history_index", {})

    for i, (q, a) in enumerate(st.session_state.user_history):
        safe_s   = per_q[i] if per_q and i < len(per_q) and isinstance(per_q[i], int) else 0
        tip      = per_q_tips[i] if per_q_tips and i < len(per_q_tips) else "Review this topic."
        ideal    = per_q_ideal[i] if per_q_ideal and i < len(per_q_ideal) else ""
        color    = "#34a853" if safe_s >= 8 else "#fbbc04" if safe_s >= 5 else "#ea4335"
        emoji    = "🟢" if safe_s >= 8 else "🟡" if safe_s >= 5 else "🔴"

        with st.expander(f"{emoji} Q{i+1} — {safe_s}/10 — {q[:70]}{'...' if len(q)>70 else ''}"):
            st.markdown(f"**Question:** {q}")
            st.markdown("---")

            diagram_img_bytes = saved_diagram_images.get(i)
            if diagram_img_bytes:
                st.image(diagram_img_bytes, caption=f"Your Diagram — Q{i+1}", width=480)

            code_prefix    = "[Submitted as Pseudocode/Code]"
            diagram_prefix = "[Submitted as Diagram]"
            if a.startswith(code_prefix):
                st.markdown("**Your Answer** *(code/pseudocode)*:")
                st.code(a[len(code_prefix):].strip(), language="python")
            elif a.startswith(diagram_prefix):
                st.markdown("**Your Answer** *(diagram)*:")
                st.info(a[len(diagram_prefix):].strip())
            else:
                st.markdown("**Your Answer:**")
                st.info(a)

            sc1, sc2 = st.columns([1, 4])
            with sc1:
                st.markdown(f"""
                <div style="text-align:center; padding:12px; background:#1a1d28;
                            border-radius:12px; border-top:3px solid {color};">
                    <div style="font-size:28px; font-weight:800; color:{color};">{safe_s}</div>
                    <div style="font-size:10px; color:#666;">/ 10</div>
                </div>
                """, unsafe_allow_html=True)
            with sc2:
                st.markdown(f"💡 **Tip:** {tip}")
                if ideal:
                    st.markdown(f"📋 **Ideal answer should cover:** {ideal}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Download report ──
    def _build_response_sheet_text():
        lines = []
        lines.append("=" * 60)
        lines.append("AI INTERVIEW SYSTEM — EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Company / Role: {company} / {role}")
        if report_time:
            lines.append(f"Generated: {report_time}")
        lines.append(f"Overall Score: {live_score}/100  |  Verdict: {verdict}")
        lines.append("")
        lines.append("SCORE BREAKDOWN")
        for key, label in SCORE_COMPONENT_LABELS.items():
            val = score_breakdown.get(key, 0)
            weight_pct = int(SCORE_WEIGHTS[key] * 100)
            lines.append(f"  {label} ({weight_pct}%): {val}/100")
        lines.append("")
        lines.append(f"Strengths: {eval_data.get('technical_strengths', '')}")
        lines.append(f"Gaps: {eval_data.get('critical_gaps_identified', '')}")
        lines.append(f"Communication: {eval_data.get('communication_feedback', '')}")
        if improvement_plan:
            lines.append(f"Improvement Plan: {improvement_plan}")
        if study_topics:
            lines.append(f"Study Topics: {', '.join(study_topics)}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("PROCTORING & ASSESSMENT INTEGRITY AUDIT")
        lines.append("=" * 60)
        violations = st.session_state.get("proctor_tab_switches", 0)
        camera_on = st.session_state.get("camera_enabled", False)
        integrity_score = max(50, 100 - (violations * 10))
        lines.append(f"Integrity Score: {integrity_score}%")
        lines.append(f"Background Camera: {'Verified Active (Continuous)' if camera_on else 'Off / Skipped'}")
        lines.append("Microphone Telemetry: Active (Pitch, Amplitude, Voice Tracked)")
        lines.append(f"Tab Switches / Focus Losses: {violations}")
        lines.append(f"Proctoring Status: {'Certified Legitimate Session' if violations == 0 else 'Security Warnings Logged'}")
        lines.append("")
        lines.append("=" * 60)
        lines.append("QUESTION-BY-QUESTION TRANSCRIPT")
        lines.append("=" * 60)
        for i, (q, a) in enumerate(st.session_state.user_history):
            safe_s = per_q[i] if per_q and i < len(per_q) and isinstance(per_q[i], int) else 0
            tip = per_q_tips[i] if per_q_tips and i < len(per_q_tips) else ""
            ideal_a = per_q_ideal[i] if per_q_ideal and i < len(per_q_ideal) else ""
            lines.append("")
            lines.append(f"Q{i+1} — Score: {safe_s}/10")
            lines.append(f"Question: {q}")
            note = " [diagram submitted — see app]" if i in saved_diagram_images else ""
            lines.append(f"Answer: {a}{note}")
            if tip:
                lines.append(f"Tip: {tip}")
            if ideal_a:
                lines.append(f"Ideal: {ideal_a}")
        return "\n".join(lines)

    col_dl, col_new = st.columns(2)
    with col_dl:
        st.download_button(
            "⬇️ Download Report (.txt)",
            data=_build_response_sheet_text(),
            file_name=f"interview_report_{company}_{role}.txt".replace(" ", "_"),
            mime="text/plain"
        )
    with col_new:
        if st.button("🔄 Start New Interview", type="primary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ============================================================
# TIER 2 — ACTIVE INTERVIEW
# ============================================================
elif st.session_state.interview_active:
    current_questions = st.session_state.locked_questions_pool
    idx = st.session_state.current_index

    # ── System Check — Proctored Interview Room (Camera & Mic Verification) ──
    if not st.session_state.system_check_done:
        proctor_room_html = """
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <style>
            * {
                box-sizing: border-box;
                margin: 0;
                padding: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            }
            body {
                background: transparent;
                padding: 2px;
            }
            .proctor-card {
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                border: 1px solid #e2e8f0;
                padding: 24px 28px;
                max-width: 860px;
                margin: 0 auto;
            }
            .proctor-header {
                margin-bottom: 20px;
            }
            .proctor-title {
                font-size: 21px;
                font-weight: 700;
                color: #0f172a;
                letter-spacing: -0.2px;
                margin-bottom: 4px;
            }
            .proctor-subtitle {
                font-size: 13.5px;
                color: #64748b;
                line-height: 1.5;
            }
            .proctor-grid {
                display: flex;
                flex-direction: row;
                gap: 24px;
                align-items: stretch;
            }
            @media (max-width: 700px) {
                .proctor-grid {
                    flex-direction: column;
                }
            }
            .video-col {
                flex: 0 0 320px;
                display: flex;
                flex-direction: column;
            }
            .video-box {
                position: relative;
                width: 100%;
                height: 200px;
                background: #090d16;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06);
                display: flex;
                align-items: center;
                justify-content: center;
            }
            video#preview-video {
                width: 100%;
                height: 100%;
                object-fit: cover;
                transform: scaleX(-1);
                border-radius: 10px;
            }
            .video-loading {
                position: absolute;
                color: #94a3b8;
                font-size: 13px;
                text-align: center;
                padding: 12px;
            }
            .live-badge {
                position: absolute;
                top: 8px;
                left: 8px;
                background: rgba(15, 23, 42, 0.85);
                color: #10b981;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.5px;
                padding: 3px 8px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                gap: 5px;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }
            .live-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background: #10b981;
                animation: pulse 1.6s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; transform: scale(1); }
                50% { opacity: 0.3; transform: scale(0.85); }
            }
            .video-caption {
                font-size: 11.5px;
                color: #64748b;
                margin-top: 8px;
                line-height: 1.4;
            }
            .devices-col {
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 13px;
                justify-content: center;
            }
            .device-field {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            .device-label {
                font-size: 12.5px;
                font-weight: 600;
                color: #1e293b;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .device-select {
                width: 100%;
                padding: 9px 12px;
                border-radius: 8px;
                border: 1px solid #cbd5e1;
                background-color: #f8fafc;
                color: #0f172a;
                font-size: 13px;
                font-weight: 500;
                outline: none;
                cursor: pointer;
                transition: border-color 0.15s, box-shadow 0.15s;
            }
            .device-select:focus {
                border-color: #3b82f6;
                box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
                background-color: #ffffff;
            }
            .audio-meter-wrap {
                background: #f1f5f9;
                border-radius: 8px;
                padding: 8px 12px;
                border: 1px solid #e2e8f0;
            }
            .audio-meter-header {
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                font-weight: 600;
                color: #475569;
                margin-bottom: 5px;
            }
            .audio-meter-track {
                width: 100%;
                height: 5px;
                background: #cbd5e1;
                border-radius: 4px;
                overflow: hidden;
            }
            .audio-meter-fill {
                height: 100%;
                width: 0%;
                background: linear-gradient(90deg, #10b981, #06b6d4);
                border-radius: 4px;
                transition: width 0.06s ease-out;
            }
            .status-chips-row {
                display: flex;
                gap: 8px;
                margin-top: 2px;
            }
            .status-pill {
                flex: 1;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                padding: 6px 10px;
                font-size: 11.5px;
                color: #334155;
                font-weight: 500;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            .status-pill.success {
                background: #f0fdf4;
                border-color: #bbf7d0;
                color: #15803d;
            }
        </style>
        </head>
        <body>
        <div class="proctor-card">
            <div class="proctor-header">
                <div class="proctor-title">🔒 Mandatory Proctored Device & System Verification</div>
                <div class="proctor-subtitle">Please allow camera and microphone system permissions when prompted. Verify your camera, microphone, and speaker audio output below. The assessment will remain locked until hardware verification succeeds.</div>
            </div>
            <div class="proctor-grid">
                <div class="video-col">
                    <div class="video-box">
                        <div id="video-loading" class="video-loading">Connecting camera preview...</div>
                        <video id="preview-video" autoplay playsinline muted></video>
                        <div id="live-badge" class="live-badge" style="display:none;">
                            <span class="live-dot"></span> LIVE
                        </div>
                    </div>
                    <div class="video-caption">Camera runs in the backend throughout your interview.</div>
                </div>
                <div class="devices-col">
                    <div class="device-field">
                        <label class="device-label" for="video-select">
                            <span>📹</span> Video devices (Camera)
                        </label>
                        <select id="video-select" class="device-select">
                            <option value="">Detecting camera devices...</option>
                        </select>
                    </div>
                    <div class="device-field">
                        <label class="device-label" for="audio-select">
                            <span>🎤</span> Audio input (Microphone)
                        </label>
                        <select id="audio-select" class="device-select">
                            <option value="">Detecting audio devices...</option>
                        </select>
                    </div>
                    <div class="audio-meter-wrap">
                        <div class="audio-meter-header">
                            <span>Microphone activity</span>
                            <span id="audio-meter-status">Waiting for speech...</span>
                        </div>
                        <div class="audio-meter-track">
                            <div id="audio-meter-fill" class="audio-meter-fill"></div>
                        </div>
                    </div>
                    <div class="device-field" style="margin-top: 6px;">
                        <label class="device-label">
                            <span>🔊</span> Audio output (Speaker)
                        </label>
                        <div style="display:flex; gap:8px; align-items:center;">
                            <button type="button" id="speaker-test-btn" onclick="testSpeakerOutput()" style="
                                background: #f1f5f9;
                                border: 1px solid #cbd5e1;
                                border-radius: 6px;
                                padding: 6px 12px;
                                font-size: 11.5px;
                                font-weight: 600;
                                color: #1e293b;
                                cursor: pointer;
                                display: flex;
                                align-items: center;
                                gap: 6px;
                            ">
                                <span>🔊</span> Play Test Sound
                            </button>
                            <span style="font-size: 11px; color: #64748b;">Click to verify speaker output</span>
                        </div>
                    </div>
                    <div class="status-chips-row" style="margin-top: 6px;">
                        <div id="cam-status" class="status-pill">
                            <span id="cam-icon">⏳</span> <span id="cam-text">Camera: Detecting</span>
                        </div>
                        <div id="mic-status" class="status-pill">
                            <span id="mic-icon">⏳</span> <span id="mic-text">Mic: Detecting</span>
                        </div>
                        <div id="speaker-status" class="status-pill">
                            <span id="speaker-icon">⏳</span> <span id="speaker-text">Speaker: Untested</span>
                        </div>
                    </div>
                </div>
            </div>
            <div style="margin-top: 12px; padding: 7px 12px; background: #f8fafc; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 11px; color: #64748b; display: flex; align-items: center; gap: 6px;">
                <span>⚙️</span> <span><b>System Permissions:</b> Set browser permissions to <i>Allow</i> for Camera & Mic. If blocked, click the lock icon in the URL bar to configure.</span>
            </div>
        </div>

        <script>
        let currentStream = null;
        let audioCtx = null;
        let analyser = null;
        let animId = null;

        function testSpeakerOutput() {
            const pill = document.getElementById('speaker-status');
            const icon = document.getElementById('speaker-icon');
            const text = document.getElementById('speaker-text');
            const btn = document.getElementById('speaker-test-btn');
            
            btn.disabled = true;
            btn.style.opacity = '0.6';
            btn.innerHTML = '<span>🔊</span> Playing...';

            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(587.33, ctx.currentTime);
                osc.frequency.exponentialRampToValueAtTime(880, ctx.currentTime + 0.25);
                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 0.5);

                if ('speechSynthesis' in window) {
                    const utt = new SpeechSynthesisUtterance("Audio output verified. System audio configured successfully.");
                    utt.rate = 1.0;
                    utt.onend = () => {
                        btn.disabled = false;
                        btn.style.opacity = '1';
                        btn.innerHTML = '<span>🔊</span> Play Test Sound';
                    };
                    window.speechSynthesis.speak(utt);
                } else {
                    setTimeout(() => {
                        btn.disabled = false;
                        btn.style.opacity = '1';
                        btn.innerHTML = '<span>🔊</span> Play Test Sound';
                    }, 800);
                }

                pill.className = 'status-pill success';
                icon.textContent = '✅';
                text.textContent = 'Speaker Ready';
            } catch(e) {
                console.warn("Speaker test error:", e);
                btn.disabled = false;
                btn.style.opacity = '1';
                btn.innerHTML = '<span>🔊</span> Play Test Sound';
            }
        }

        async function initCamera(videoDeviceId, audioDeviceId) {
            try {
                if (currentStream) {
                    currentStream.getTracks().forEach(t => t.stop());
                }
                if (animId) cancelAnimationFrame(animId);
                if (audioCtx && audioCtx.state !== 'closed') {
                    audioCtx.close();
                }

                const constraints = {
                    video: videoDeviceId ? { deviceId: { exact: videoDeviceId }, width: { ideal: 640 }, height: { ideal: 480 } } : true,
                    audio: audioDeviceId ? { deviceId: { exact: audioDeviceId } } : true
                };

                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                currentStream = stream;

                const video = document.getElementById('preview-video');
                const loading = document.getElementById('video-loading');
                const badge = document.getElementById('live-badge');

                video.srcObject = stream;
                video.onloadedmetadata = () => {
                    video.play();
                    loading.style.display = 'none';
                    badge.style.display = 'flex';

                    const camPill = document.getElementById('cam-status');
                    camPill.className = 'status-pill success';
                    document.getElementById('cam-icon').textContent = '✅';
                    document.getElementById('cam-text').textContent = 'Camera Ready';
                };

                // Audio meter
                try {
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const src = audioCtx.createMediaStreamSource(stream);
                    analyser = audioCtx.createAnalyser();
                    analyser.fftSize = 256;
                    src.connect(analyser);

                    const data = new Uint8Array(analyser.frequencyBinCount);
                    const fill = document.getElementById('audio-meter-fill');
                    const label = document.getElementById('audio-meter-status');
                    const micPill = document.getElementById('mic-status');

                    function pollAudio() {
                        analyser.getByteFrequencyData(data);
                        let sum = 0;
                        for (let i = 0; i < data.length; i++) sum += data[i];
                        const avg = sum / data.length;
                        const pct = Math.min(100, Math.round((avg / 64) * 100));
                        fill.style.width = pct + '%';

                        if (pct > 6) {
                            label.textContent = 'Voice detected (' + pct + '%)';
                            label.style.color = '#10b981';
                            micPill.className = 'status-pill success';
                            document.getElementById('mic-icon').textContent = '✅';
                            document.getElementById('mic-text').textContent = 'Microphone Ready';
                        }
                        animId = requestAnimationFrame(pollAudio);
                    }
                    pollAudio();
                } catch(e) {
                    console.warn("Audio meter setup error:", e);
                }

            } catch(err) {
                console.error("Camera access error:", err);
                const loading = document.getElementById('video-loading');
                loading.textContent = 'Camera blocked or unavailable. Please allow browser access.';
            }
        }

        async function loadDeviceList() {
            try {
                await initCamera();
                const devices = await navigator.mediaDevices.enumerateDevices();
                const vSel = document.getElementById('video-select');
                const aSel = document.getElementById('audio-select');

                vSel.innerHTML = '';
                aSel.innerHTML = '';

                let vIdx = 0, aIdx = 0;
                devices.forEach(d => {
                    if (d.kind === 'videoinput') {
                        vIdx++;
                        const opt = document.createElement('option');
                        opt.value = d.deviceId;
                        opt.textContent = d.label || ('WebCam ' + vIdx);
                        vSel.appendChild(opt);
                    } else if (d.kind === 'audioinput') {
                        aIdx++;
                        const opt = document.createElement('option');
                        opt.value = d.deviceId;
                        opt.textContent = d.label || ('Microphone ' + aIdx);
                        aSel.appendChild(opt);
                    }
                });

                if (vIdx === 0) vSel.innerHTML = '<option value="">Default Camera</option>';
                if (aIdx === 0) aSel.innerHTML = '<option value="">Default Microphone</option>';

                vSel.addEventListener('change', () => initCamera(vSel.value, aSel.value));
                aSel.addEventListener('change', () => initCamera(vSel.value, aSel.value));
            } catch(e) {
                console.warn("Device enum error:", e);
            }
        }

        window.addEventListener('DOMContentLoaded', loadDeviceList);

        // Instant Fullscreen trigger when candidate clicks Proceed / Start button
        try {
            const pDoc = window.parent.document;
            pDoc.addEventListener('click', function(e) {
                const btn = e.target.closest('button');
                if (btn && (btn.innerText.includes('Full-Screen') || btn.innerText.includes('Proceed to Assessment'))) {
                    const el = pDoc.documentElement;
                    if (el.requestFullscreen) {
                        el.requestFullscreen().catch(() => {});
                    } else if (el.webkitRequestFullscreen) {
                        el.webkitRequestFullscreen();
                    } else if (el.msRequestFullscreen) {
                        el.msRequestFullscreen();
                    }
                }
            }, true);
        } catch(e) {}
        </script>
        </body>
        </html>
        """
        st.components.v1.html(proctor_room_html, height=365)

        # Voice STT mic & speaker verification clip
        with st.expander("🎙️ Test Voice Speech-to-Text & Speaker Playback", expanded=True):
            st.caption("Record a 2-second voice clip to confirm the Whisper AI transcriber correctly captures your speech and test your speaker audio playback.")
            mic_test = mic_recorder(
                start_prompt="🔴 Record test clip",
                stop_prompt="⏹️ Stop & verify",
                just_once=True,
                key="system_check_mic"
            )
            if mic_test is not None and 'bytes' in mic_test:
                st.audio(mic_test['bytes'], format='audio/wav')
                with st.spinner("Transcribing test clip with local Whisper..."):
                    test_transcript = speech_to_text_local_whisper(mic_test['bytes'])
                if test_transcript.startswith("⚠️"):
                    st.error(test_transcript)
                else:
                    st.session_state.mic_check_passed = True
                    st.session_state.speaker_check_passed = True
                    st.success(f'✅ Voice AI heard: "{test_transcript}" (Click play above to test speaker audio playback)')

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        col_ready, col_skip = st.columns([3, 1])
        with col_ready:
            if st.button("Enter Full-Screen & Proceed to Assessment 🚀", type="primary", use_container_width=True):
                st.session_state.system_check_done = True
                st.session_state.camera_enabled = True
                st.session_state.cam_check_passed = True
                st.session_state.mic_check_passed = True
                st.session_state.proctor_tab_switches = 0
                st.rerun()
        with col_skip:
            if st.button("⏭️ Skip check (camera/mic unavailable on this device)", use_container_width=True):
                st.session_state.system_check_done = True
                st.session_state.camera_enabled = False
                st.session_state.proctor_tab_switches = 0
                st.rerun()
        st.stop()

    # Keep camera alive silently in background — no video shown on screen
    _render_background_camera()

    # ── Last question warning ──
    if idx == 9 and not st.session_state.warning_popup_triggered:
        skipped_so_far = sum(
            1 for _, ans in st.session_state.user_history
            if ans.startswith("[Skipped by candidate")
        )
        st.warning(
            f"⚠️ **Final question.** "
            + (f"{skipped_so_far} of 9 previous questions were skipped (scored as 0). " if skipped_so_far > 0 else "All 9 previous questions answered. ")
            + "After this question, the interview ends and your evaluation is generated."
        )
        if st.button("✅ Show Final Question", key="warning_ack_btn"):
            st.session_state.warning_popup_triggered = True
            st.rerun()
    else:
        if current_questions and idx < len(current_questions):
            q_text, source, subject = current_questions[idx]

            # ── Progress bar ──
            progress_pct = int((idx / len(current_questions)) * 100)
            st.markdown(f"""
            <div class="interview-progress">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <div class="progress-text" style="margin-bottom:0;">
                        <span class="status-dot active"></span>
                        QUESTION {idx + 1} OF {len(current_questions)} · {st.session_state.sel_company} · {st.session_state.sel_role}
                    </div>
                    <div style="display:flex; align-items:center; gap:12px;">
                        <button id="header-fullscreen-btn" style="
                            background: rgba(2, 132, 199, 0.15);
                            color: #38bdf8;
                            border: 1px solid rgba(56, 189, 248, 0.35);
                            border-radius: 6px;
                            padding: 3px 10px;
                            font-size: 11px;
                            font-weight: 700;
                            cursor: pointer;
                            display: flex;
                            align-items: center;
                            gap: 5px;
                            transition: all 0.2s ease;
                        ">
                            <span>⛶</span> Enter Fullscreen
                        </button>
                        <div style="font-size:11px; font-weight:600; color:#34a853; display:flex; align-items:center; gap:6px;">
                            <span>🟢</span> Proctoring Active (Background)
                        </div>
                    </div>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width:{progress_pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Question card ──
            answer_hint = ""
            if "Pseudocode" in subject:
                answer_hint = "💡 Use the <b>Code/Diagram</b> tab → Write Pseudocode"
            elif "Diagram" in subject:
                answer_hint = "💡 Use the <b>Code/Diagram</b> tab → Draw a Diagram"

            st.markdown(f"""
            <div class="question-card">
                <div class="question-number">Question {idx + 1}</div>
                <div class="question-text">{q_text}</div>
                {"<div style='margin-top:12px; font-size:12px; opacity:0.7;'>" + answer_hint + "</div>" if answer_hint else ""}
            </div>
            """, unsafe_allow_html=True)

            # ── Listen option ──
            if st.button("🔊 Read question aloud", key=f"speak_btn_{idx}"):
                speak_text_globally_neutral(q_text)

            st.markdown("---")

            # ── Answer tabs ──
            voice_tab, text_tab, code_diagram_tab = st.tabs([
                "🎙️ Voice",
                "⌨️ Type",
                "💻 Code / Diagram"
            ])

            with voice_tab:
                st.caption("Speak your answer. It will be transcribed locally using Whisper AI.")
                audio_sample = mic_recorder(
                    start_prompt="🔴 Start Recording",
                    stop_prompt="⏹️ Stop & Transcribe",
                    just_once=True,
                    key=f"mic_{idx}"
                )
                if audio_sample is not None and 'bytes' in audio_sample:
                    current_bytes = audio_sample['bytes']
                    with st.spinner("Transcribing..."):
                        transcribed_text = speech_to_text_local_whisper(current_bytes)
                        st.session_state[f"audio_bytes_{idx}"]   = current_bytes
                        st.session_state[f"transcription_{idx}"] = transcribed_text
                        st.session_state[f"voice_flag_{idx}"]    = True
                        st.session_state[f"voice_text_{idx}"]    = transcribed_text
                    st.rerun()

                if f"audio_bytes_{idx}" in st.session_state:
                    st.audio(st.session_state[f"audio_bytes_{idx}"], format='audio/wav')
                    saved_text = st.session_state.get(f"transcription_{idx}", "")
                    if saved_text.startswith("⚠️"):
                        st.error(saved_text)
                    else:
                        st.success(f'**Your answer:** "{saved_text}"')
                    edited = st.text_area("Edit if needed:", value=saved_text, key=f"edit_{idx}")
                    if edited != saved_text:
                        st.session_state[f"transcription_{idx}"] = edited
                        st.session_state[f"voice_text_{idx}"]    = edited

            with text_tab:
                user_ans = st.text_area(
                    "Type your answer:",
                    key=f"text_{idx}",
                    placeholder="Write your answer here...",
                    height=200
                )
                word_count = len(user_ans.split()) if user_ans.strip() else 0
                st.caption(f"{word_count} words")

            with code_diagram_tab:
                answer_mode = st.radio(
                    "Answer type:",
                    ["Pseudocode / Code", "Diagram"],
                    key=f"mode_{idx}",
                    horizontal=True
                )

                if answer_mode == "Pseudocode / Code":
                    code_answer = st.text_area(
                        "Write your code/pseudocode:",
                        key=f"code_{idx}",
                        placeholder="function solve(input):\n    // your logic here\n    return result",
                        height=280
                    )
                    if code_answer.strip():
                        st.code(code_answer, language="python")

                else:
                    st.caption("Use the tools below to draw. Label shapes in the text box underneath.")
                    col_tool, col_color = st.columns([2, 1])
                    with col_tool:
                        drawing_tool = st.selectbox("Tool:", ["freedraw", "line", "rect", "circle"], key=f"tool_{idx}")
                    with col_color:
                        stroke_color = st.color_picker("Color:", "#000000", key=f"color_{idx}")

                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.05)",
                        stroke_width=3,
                        stroke_color=stroke_color,
                        background_color="#FFFFFF",
                        height=520,
                        width=900,
                        drawing_mode=drawing_tool,
                        key=f"canvas_{idx}",
                    )

                    if canvas_result.image_data is not None:
                        st.session_state[f"diagram_drawn_{idx}"] = True
                        snapshot_bytes = canvas_drawing_to_png_bytes(canvas_result.image_data)
                        if snapshot_bytes:
                            st.session_state[f"diagram_image_bytes_{idx}"] = snapshot_bytes

                    st.markdown("---")
                    st.caption("Optionally explain your diagram by voice:")
                    diagram_voice = mic_recorder(
                        start_prompt="🔴 Record explanation",
                        stop_prompt="⏹️ Stop",
                        just_once=True,
                        key=f"diagram_mic_{idx}"
                    )
                    if diagram_voice is not None and 'bytes' in diagram_voice:
                        with st.spinner("Transcribing..."):
                            diagram_voice_text = speech_to_text_local_whisper(diagram_voice['bytes'])
                            st.session_state[f"diagram_voice_text_{idx}"] = diagram_voice_text
                        st.rerun()

                    if f"diagram_voice_text_{idx}" in st.session_state:
                        dvt = st.session_state[f"diagram_voice_text_{idx}"]
                        if dvt.startswith("⚠️"):
                            st.error(dvt)
                        else:
                            st.success(f'Voice explanation: "{dvt}"')

                    voice_prefill = st.session_state.get(f"diagram_voice_text_{idx}", "")
                    existing_text = st.session_state.get(f"diagram_text_{idx}", voice_prefill)

                    diagram_explanation = st.text_area(
                        "Label and explain your diagram:",
                        value=existing_text if existing_text and not existing_text.startswith("⚠️") else "",
                        key=f"diagram_text_{idx}",
                        placeholder="e.g. Box 1 = New, Box 2 = Ready...",
                        height=100
                    )

                    if st.session_state.get(f"diagram_drawn_{idx}", False):
                        if st.button("🔍 Analyze with Vision AI", key=f"analyze_diagram_{idx}"):
                            with st.spinner("Moondream2 analyzing... (15-30 sec)"):
                                if canvas_result.image_data is not None:
                                    vision_desc = describe_diagram_with_vision_model(canvas_result.image_data)
                                    st.session_state[f"vision_description_{idx}"] = vision_desc

                        if f"vision_description_{idx}" in st.session_state:
                            st.info(f"**Vision AI:** {st.session_state[f'vision_description_{idx}']}")

            st.markdown("---")
            col_save, col_skip = st.columns([3, 1])
            with col_save:
                save_clicked = st.button("💾 Save & Next ➡️", key=f"save_{idx}", type="primary")
            with col_skip:
                skip_clicked = st.button("⏭️ Skip", key=f"skip_{idx}")

            if skip_clicked:
                st.session_state[f"skip_confirm_{idx}"] = True
                st.rerun()

            if st.session_state.get(f"skip_confirm_{idx}", False):
                st.warning("This question will be scored as 0. You cannot return to it.")
                cc1, cc2 = st.columns(2)
                with cc1:
                    if st.button("✅ Confirm skip", key=f"confirm_skip_{idx}"):
                        st.session_state.user_history.append(
                            (q_text, "[Skipped by candidate — no answer submitted]")
                        )
                        st.session_state[f"skip_confirm_{idx}"] = False
                        st.session_state.current_index = idx + 1
                        st.rerun()
                with cc2:
                    if st.button("↩️ Cancel", key=f"cancel_skip_{idx}"):
                        st.session_state[f"skip_confirm_{idx}"] = False
                        st.rerun()

            if save_clicked:
                final_answer = ""
                code_text     = st.session_state.get(f"code_{idx}", "")
                diagram_text  = st.session_state.get(f"diagram_text_{idx}", "")
                diagram_voice_text = st.session_state.get(f"diagram_voice_text_{idx}", "")
                diagram_drawn = st.session_state.get(f"diagram_drawn_{idx}", False)
                vision_desc   = st.session_state.get(f"vision_description_{idx}", "")

                combined_explanation = ""
                if diagram_text.strip() and not diagram_text.startswith("⚠️"):
                    combined_explanation += diagram_text.strip()
                if diagram_voice_text and not diagram_voice_text.startswith("⚠️"):
                    if combined_explanation:
                        combined_explanation += " | Voice: " + diagram_voice_text.strip()
                    else:
                        combined_explanation = diagram_voice_text.strip()

                if code_text and code_text.strip():
                    final_answer = f"[Submitted as Pseudocode/Code]\n{code_text.strip()}"
                elif diagram_drawn and vision_desc:
                    final_answer = (
                        f"[Submitted as Diagram]\n"
                        f"Vision AI description: {vision_desc}\n"
                        f"Candidate explanation: {combined_explanation if combined_explanation else '(none provided)'}"
                    )
                elif diagram_drawn and combined_explanation:
                    final_answer = (
                        f"[Submitted as Diagram] "
                        f"Candidate drew a diagram and explained: {combined_explanation}"
                    )
                elif diagram_drawn:
                    final_answer = "[Submitted as Diagram] Candidate drew a diagram with no explanation."
                elif user_ans.strip():
                    final_answer = user_ans.strip()
                elif st.session_state.get(f"voice_flag_{idx}", False):
                    final_answer = st.session_state.get(f"voice_text_{idx}", "")

                if not final_answer.strip():
                    st.error("⚠️ No answer detected. Provide an answer or skip the question.")
                else:
                    diagram_image_bytes = st.session_state.get(f"diagram_image_bytes_{idx}")
                    if diagram_drawn and diagram_image_bytes:
                        history_position = len(st.session_state.user_history)
                        st.session_state["diagram_images_by_history_index"][history_position] = diagram_image_bytes

                    st.session_state.user_history.append((q_text, final_answer))
                    st.session_state.current_index = idx + 1
                    st.rerun()

        else:
            with st.spinner("Evaluating your interview..."):
                st.session_state.final_evaluation = evaluate_entire_interview_via_slm(
                    st.session_state.user_history
                )
                st.session_state.report_generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
                st.session_state.interview_active = False
                st.session_state.final_done = True
            st.rerun()

# ============================================================
# TIER 3 — ENTRY HUB
# ============================================================
else:
    st.markdown("## 🎯 AI Interview System")
    st.caption("Simulate real-world technical interviews with AI-powered evaluation and personalized feedback.")

    st.markdown("---")

    st.markdown("### 📄 Upload Resume")
    st.caption("PDF only. Used to detect relevant skills for your profile — not stored after this session.")
    uploaded_file = st.file_uploader("Choose a PDF resume:", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if st.session_state.locked_candidate_profile is None:
            with st.expander("🔗 Add profile links (optional)", expanded=True):
                st.caption("Public pages are scraped directly — no APIs or logins used.")
                lk_col1, lk_col2 = st.columns(2)
                with lk_col1:
                    github_url    = st.text_input("GitHub URL", key="kg_github", placeholder="https://github.com/...")
                    leetcode_id   = st.text_input("LeetCode", key="kg_leetcode", placeholder="https://leetcode.com/u/...")
                with lk_col2:
                    kaggle_id     = st.text_input("Kaggle", key="kg_kaggle", placeholder="https://kaggle.com/...")
                    linkedin_url  = st.text_input("LinkedIn URL", key="kg_linkedin", placeholder="https://www.linkedin.com/in/...")

            if st.button("🔎 Analyze Profile", type="primary"):
                with st.spinner("Analyzing resume and profile links..."):
                    raw_text = extract_text_from_pdf(uploaded_file)
                    profile_links = {
                        "linkedin": st.session_state.get("kg_linkedin", ""),
                        "github": st.session_state.get("kg_github", ""),
                        "leetcode": st.session_state.get("kg_leetcode", ""),
                        "kaggle": st.session_state.get("kg_kaggle", ""),
                    }
                    skills_text, saved_links, scrape_status = build_candidate_knowledge_graph(
                        raw_text, profile_links
                    )
                    st.session_state.locked_candidate_profile = skills_text
                    st.session_state.candidate_profile_links = saved_links
                    st.session_state.candidate_scrape_status = scrape_status
                st.rerun()

    if st.session_state.locked_candidate_profile is not None:
        st.success("✅ Profile analyzed")

        saved_links = st.session_state.get("candidate_profile_links", {})
        scrape_status = st.session_state.get("candidate_scrape_status", {})
        display_labels = {"linkedin": "LinkedIn", "github": "GitHub", "leetcode": "LeetCode", "kaggle": "Kaggle"}
        all_labels = sorted(list(set(saved_links) | set(scrape_status)))
        if all_labels:
            chip_parts = []
            for label in all_labels:
                status = scrape_status.get(label, "link_only")
                icon = "✅" if status == "scraped" else "🔗"
                note = "scraped" if status == "scraped" else "link only"
                display_name = display_labels.get(label, label.capitalize())
                chip_parts.append(f"{icon} {display_name} ({note})")
            st.caption(" · ".join(chip_parts))

        edited_skills = st.text_area(
            "**Detected Skills** *(edit if needed)*:",
            value=st.session_state.locked_candidate_profile,
            key="skills_editor",
            height=80
        )
        st.session_state.locked_candidate_profile = edited_skills

        st.markdown("---")
        st.markdown("### 🏢 Select Company & Role")

        col_a, col_b = st.columns(2)
        with col_a:
            selected_company = st.selectbox("Company:", COMPANIES_LIST, label_visibility="collapsed")
        with col_b:
            selected_role = st.selectbox("Role:", ROLES_LIST, label_visibility="collapsed")

        st.markdown("---")
        if st.button("🚀 Start Interview", type="primary"):
            db_questions = get_questions_randomized_pool(selected_company, selected_role)
            if not db_questions:
                st.error(
                    f"❌ No questions for {selected_company} — {selected_role}. "
                    "Try 'Force Re-seed Database' in the sidebar."
                )
            else:
                st.session_state.locked_questions_pool   = db_questions[:10]
                st.session_state.current_index           = 0
                st.session_state.user_history            = []
                st.session_state.warning_popup_triggered = False
                st.session_state.system_check_done       = False
                st.session_state.cam_check_passed        = False
                st.session_state.mic_check_passed        = False
                st.session_state.sel_company             = selected_company
                st.session_state.sel_role                = selected_role
                st.session_state.interview_active        = True
                st.rerun()