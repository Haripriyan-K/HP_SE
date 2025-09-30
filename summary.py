# # summary.py
# import os
# import tempfile
# import subprocess
# from typing import Optional

# import whisper
# from transformers import pipeline
# from pydub import AudioSegment
# from moviepy.editor import VideoFileClip
# import streamlit as st


# # ---------------- Summarizer Setup ----------------
# # Load Hugging Face summarizer (only once for efficiency)
# @st.cache_resource
# def load_summarizer():
#     return pipeline("summarization", model="facebook/bart-large-cnn")


# # ---------------- Audio Summarization ----------------
# @st.cache_resource
# def load_whisper_model(model_size: str = "tiny"):
#     return whisper.load_model(model_size)


# def transcribe_audio_chunks(audio_file: str, model_size: str = "tiny") -> str:
#     """Transcribe long audio by splitting into 5-min chunks."""
#     model = load_whisper_model(model_size)

#     # Convert to wav with pydub
#     audio = AudioSegment.from_file(audio_file)
#     duration_ms = len(audio)

#     chunk_ms = 5 * 60 * 1000  # 5 minutes per chunk
#     transcripts = []

#     for i in range(0, duration_ms, chunk_ms):
#         chunk = audio[i:i + chunk_ms]

#         # Create a Windows-safe temp file
#         fd, tmp_path = tempfile.mkstemp(suffix=".wav")
#         os.close(fd)  # release handle immediately

#         try:
#             chunk.export(tmp_path, format="wav")
#             result = model.transcribe(tmp_path, fp16=False)
#             transcripts.append(result["text"])
#         finally:
#             if os.path.exists(tmp_path):
#                 os.remove(tmp_path)

#     return " ".join(transcripts)


# def summarize_audio(audio_file: str, model_size: str = "tiny") -> str:
#     """Summarize audio by transcription + text summarization."""
#     summarizer = load_summarizer()
#     transcript = transcribe_audio_chunks(audio_file, model_size=model_size)

#     # Hugging Face summarizer handles ~1024 tokens max → split if needed
#     if len(transcript) < 4000:
#         summary = summarizer(
#             transcript,
#             max_length=150,
#             min_length=40,
#             do_sample=False
#         )[0]['summary_text']
#     else:
#         # Split into smaller chunks
#         words = transcript.split()
#         chunks = [" ".join(words[i:i + 500]) for i in range(0, len(words), 500)]
#         summaries = [
#             summarizer(chunk, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
#             for chunk in chunks
#         ]
#         summary = " ".join(summaries)

#     return summary


# # ---------------- Text Summarization ----------------
# def summarize_text(text: str) -> str:
#     """Summarize plain text"""
#     summarizer = load_summarizer()
#     summary = summarizer(
#         text,
#         max_length=150,
#         min_length=40,
#         do_sample=False
#     )[0]['summary_text']
#     return summary


# # ---------------- Video Summarization ----------------
# def _ffmpeg_extract_audio(video_path: str, out_wav: str) -> None:
#     """Use ffmpeg CLI to extract audio to WAV (PCM 16-bit mono, 16k sample rate)."""
#     cmd = [
#         "ffmpeg", "-y", "-i", video_path,
#         "-vn",                      # no video
#         "-acodec", "pcm_s16le",     # WAV PCM 16-bit
#         "-ar", "16000",             # sample rate 16k
#         "-ac", "1",                 # mono
#         out_wav
#     ]
#     subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# def summarize_video(video_file: str, model_size: str = "tiny") -> str:
#     """Extract audio from video and summarize."""
#     if not os.path.exists(video_file):
#         return f"ERROR: video file not found: {video_file}"

#     tmp_wav: Optional[str] = None
#     clip: Optional[VideoFileClip] = None

#     fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
#     os.close(fd)

#     try:
#         try:
#             # Try moviepy extraction first
#             clip = VideoFileClip(video_file)
#             if clip.audio is None:
#                 return "No audio track found in the provided video."

#             clip.audio.write_audiofile(tmp_wav, logger=None, verbose=False)
#         except Exception:
#             # Fallback to ffmpeg
#             try:
#                 _ffmpeg_extract_audio(video_file, tmp_wav)
#             except subprocess.CalledProcessError:
#                 return "ERROR: failed to extract audio (ffmpeg failed). Ensure ffmpeg is installed."

#         if not tmp_wav or not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0:
#             return "ERROR: extracted audio file missing or empty."

#         summary = summarize_audio(tmp_wav, model_size=model_size)
#         if summary is None:
#             return "ERROR: summarize_audio returned None."
#         return summary

#     finally:
#         try:
#             if clip is not None:
#                 clip.close()
#         except Exception:
#             pass
#         try:
#             if tmp_wav and os.path.exists(tmp_wav):
#                 os.remove(tmp_wav)
#         except Exception:
#             pass


# # ---------------- Streamlit Frontend ----------------
# def main():
#     st.title("🎧📹 Text, Audio & Video Summarizer")
#     st.write("Upload text, audio, or video to generate concise summaries.")

#     option = st.radio("Choose input type:", ["Text", "Audio", "Video"])

#     if option == "Text":
#         text_input = st.text_area("Paste your text here:")
#         if st.button("Summarize Text"):
#             if text_input.strip():
#                 with st.spinner("Summarizing..."):
#                     try:
#                         summary = summarize_text(text_input)
#                         st.subheader("Summary")
#                         st.write(summary)
#                     except Exception as e:
#                         st.error(f"Error summarizing text: {str(e)}")
#             else:
#                 st.warning("Please enter some text.")

#     elif option == "Audio":
#         audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
#         if st.button("Summarize Audio") and audio_file is not None:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
#                 tmp.write(audio_file.read())
#                 tmp_path = tmp.name
            
#             try:
#                 with st.spinner("Transcribing & summarizing audio..."):
#                     summary = summarize_audio(tmp_path)
#                 st.subheader("Summary")
#                 st.write(summary)
#             except Exception as e:
#                 st.error(f"Error processing audio: {str(e)}")
#             finally:
#                 if os.path.exists(tmp_path):
#                     os.remove(tmp_path)

#     elif option == "Video":
#         video_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])
#         if st.button("Summarize Video") and video_file is not None:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
#                 tmp.write(video_file.read())
#                 tmp_path = tmp.name
            
#             try:
#                 with st.spinner("Extracting audio & summarizing video..."):
#                     summary = summarize_video(tmp_path)
#                 st.subheader("Summary")
#                 st.write(summary)
#             except Exception as e:
#                 st.error(f"Error processing video: {str(e)}")
#             finally:
#                 if os.path.exists(tmp_path):
#                     os.remove(tmp_path)


# if __name__ == "__main__":
#     main()


# #------------------------------------------------------------------------------------------------------------------
    


# summary.py
import os
import tempfile
import subprocess
from typing import Optional

import whisper
from transformers import pipeline
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
import streamlit as st

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="Smart Summarizer",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #f9f9f9 0%, #e3f2fd 100%); font-family: 'Segoe UI', sans-serif; }
    .main-title { text-align: center; font-size: 2.8rem; font-weight: 700; background: -webkit-linear-gradient(#2196f3, #21cbf3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem; }
    .subtitle { text-align: center; font-size: 1.1rem; color: #444; margin-bottom: 2rem; }
    .stRadio > div { display: flex; justify-content: center; gap: 1.5rem; }
    div.stButton > button { background: linear-gradient(90deg, #2196f3, #21cbf3); color: white; border: none; border-radius: 8px; padding: 0.6rem 1.4rem; font-size: 1rem; font-weight: 600; cursor: pointer; transition: transform 0.2s ease, background 0.3s ease; }
    div.stButton > button:hover { transform: scale(1.05); background: linear-gradient(90deg, #1976d2, #21a1f3); }
    .summary-box { background: black; padding: 1.5rem; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-top: 1rem; }
    .summary-box h3 { color: #2196f3; margin-bottom: 0.5rem; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- Summarizer Setup ----------------
@st.cache_resource
def load_summarizer():
    # Lighter and faster summarizer
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_whisper_model(model_size: str = "tiny"):
    return whisper.load_model(model_size)

# ---------------- Text Summarization ----------------
def summarize_text(text: str) -> str:
    """Summarize large text by splitting into chunks."""
    summarizer = load_summarizer()
    words = text.split()
    max_chunk_words = 500

    chunks = [" ".join(words[i:i + max_chunk_words]) for i in range(0, len(words), max_chunk_words)]
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"[Error summarizing chunk: {str(e)}]")
    return " ".join(summaries)

# ---------------- Audio Summarization ----------------
def transcribe_audio_chunks(audio_file: str, model_size: str = "tiny") -> str:
    model = load_whisper_model(model_size)
    audio = AudioSegment.from_file(audio_file)
    duration_ms = len(audio)
    chunk_ms = 5 * 60 * 1000  # 5 minutes
    transcripts = []

    for i in range(0, duration_ms, chunk_ms):
        chunk = audio[i:i + chunk_ms]
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            chunk.export(tmp_path, format="wav")
            result = model.transcribe(tmp_path, fp16=False)
            transcripts.append(result["text"])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return " ".join(transcripts)

def summarize_audio(audio_file: str, model_size: str = "tiny") -> str:
    """Summarize audio by transcription + text summarization."""
    transcript = transcribe_audio_chunks(audio_file, model_size=model_size)
    return summarize_text(transcript)

# ---------------- Video Summarization ----------------
def _ffmpeg_extract_audio(video_path: str, out_wav: str) -> None:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_wav
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def summarize_video(video_file: str, model_size: str = "tiny") -> str:
    if not os.path.exists(video_file):
        return f"ERROR: video file not found: {video_file}"

    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    clip: Optional[VideoFileClip] = None
    try:
        try:
            clip = VideoFileClip(video_file)
            if clip.audio is None:
                return "No audio track found in the provided video."
            clip.audio.write_audiofile(tmp_wav, logger=None, verbose=False)
        except Exception:
            _ffmpeg_extract_audio(video_file, tmp_wav)

        return summarize_audio(tmp_wav, model_size=model_size)
    finally:
        if clip: clip.close()
        if os.path.exists(tmp_wav): os.remove(tmp_wav)


# # ---------------- Streamlit Frontend ----------------
def main():
    st.markdown("<h1 class='main-title'>✨ Content Summarizer</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Summarize text, audio, or video into crisp insights.</p>", unsafe_allow_html=True)

    option = st.radio("Choose input type:", ["📝 Text", "🎧 Audio", "📹 Video"], horizontal=True)

    if option == "📝 Text":
        text_input = st.text_area("Paste your text here:")
        if st.button("🚀 Summarize Text"):
            if text_input.strip():
                with st.spinner("✨ Summarizing text..."):
                    try:
                        summary = summarize_text(text_input)
                        st.markdown("<div class='summary-box'><h3>Summary</h3>" + summary + "</div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter some text.")

    elif option == "🎧 Audio":
        audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
        if st.button("🚀 Summarize Audio") and audio_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            try:
                with st.spinner("🎤 Transcribing & summarizing audio..."):
                    summary = summarize_audio(tmp_path)
                st.markdown("<div class='summary-box'><h3>Summary</h3>" + summary + "</div>", unsafe_allow_html=True)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

    elif option == "📹 Video":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "mkv", "avi", "mov"])
        if st.button("🚀 Summarize Video") and video_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(video_file.read())
                tmp_path = tmp.name
            try:
                with st.spinner("🎬 Extracting audio & summarizing video..."):
                    summary = summarize_video(tmp_path)
                st.markdown("<div class='summary-box'><h3>Summary</h3>" + summary + "</div>", unsafe_allow_html=True)
            finally:
                if os.path.exists(tmp_path): os.remove(tmp_path)

if __name__ == "__main__":
    main()




#__________________________________________________________________________________\





