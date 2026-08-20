import os
import time
import glob
import uuid
import asyncio
import tempfile
import json
import re
from datetime import datetime
import httpx

import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter, Reverb, Chorus, Distortion, PitchShift, Delay

# Video Editing (MoviePy)
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip, TextClip, ColorClip, CompositeAudioClip

# FastAPI & Gradio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Depends
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import APIKeyHeader
import gradio as gr

# MCP (Model Context Protocol)
from mcp.server.fastmcp import FastMCP

# ==========================================
# 0. Core Configuration & Production Safety
# ==========================================
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = os.path.join(TEMP_DIR, "media_uploads")
OUTPUT_DIR = os.path.join(TEMP_DIR, "media_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
SPEAKERS_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models", "speakers")
os.makedirs(SPEAKERS_DIR, exist_ok=True)

API_KEY_SECRET=os.environ.get("API_KEY", "") 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY_SECRET and api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

gpu_lock = asyncio.Lock()
cpu_render_lock = asyncio.Lock() # Lock for Heavy Video Rendering

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

def get_available_speakers():
    speakers = ["Default Model"]
    if os.path.exists(SPEAKERS_DIR):
        for item in os.listdir(SPEAKERS_DIR):
            if os.path.isdir(os.path.join(SPEAKERS_DIR, item)):
                speakers.append(item)
    return speakers

def cleanup_old_files():
    now = time.time()
    for directory in [UPLOAD_DIR, OUTPUT_DIR]:
        for filepath in glob.glob(os.path.join(directory, "*")):
            try:
                if os.stat(filepath).st_mtime < now - 3600: os.remove(filepath)
            except Exception: pass

def append_log(current_logs, new_message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {new_message}"
    if not current_logs: return log_entry
    logs_list = current_logs.split("\n")
    logs_list.append(log_entry)
    if len(logs_list) > 20: logs_list = logs_list[-20:]
    return "\n".join(logs_list)

# ==========================================
# 1. Studio Presets Configuration
# ==========================================
STUDIO_PRESETS = {
    "🎙️ Podcast Studio": {"bass": 5.0, "treble": 3.5, "comp": 3.5, "reverb": 0.05, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "นุ่มลึก มีน้ำหนัก ฟังสบาย"},
    "📖 Audiobook Pro": {"bass": 2.0, "treble": 2.0, "comp": 2.5, "reverb": 0.15, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "ใสสะอาด มีมิติเสียงก้องนิดๆ"},
    "🗣️ Natural Human": {"bass": 1.0, "treble": 1.5, "comp": 1.5, "reverb": 0.08, "gate": False, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "ธรรมชาติ ไม่บีบอัดมาก"}
}

# ==========================================
# 2. AI Audio & Video Operations
# ==========================================

# --- AUDIO ---
async def _generate_audio_chunk(engine: str, mode: str, text_chunk: str, lang: str, ref_path: str, speed: float, speaker_model: str):
    await asyncio.sleep(1) # Simulate GPU time
    sample_rate = 44100
    duration = max(1.0, len(text_chunk) * 0.1) 
    audio_data = np.zeros((int(sample_rate * duration), 1))
    return audio_data, sample_rate

async def generate_tts_safely(engine: str, mode: str, full_text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0, speaker_model: str = "Default Model"):
    chunks = re.split(r'(?<=[.!?\n])\s+', full_text.strip())
    chunks = [c for c in chunks if c.strip()]
    if not chunks: chunks = [full_text]
    all_audio_arrays = []
    sample_rate = 44100
    async with gpu_lock:
        for chunk in chunks:
            if not chunk.strip(): continue
            audio_data, sr = await _generate_audio_chunk(engine, mode, chunk, lang, ref_path, speed, speaker_model)
            sample_rate = sr
            all_audio_arrays.append(audio_data)
            pause = np.zeros((int(sample_rate * 0.2), audio_data.shape[1] if len(audio_data.shape) > 1 else 1))
            all_audio_arrays.append(pause)
    if all_audio_arrays:
        final_audio = np.concatenate(all_audio_arrays, axis=0)
        sf.write(out_path, final_audio, sample_rate)
    else:
        sf.write(out_path, np.zeros((sample_rate, 1)), sample_rate)

def apply_studio_mastering(input_path: str, output_path: str, gate: bool=True, bass: float=4.5, treble: float=3.0, comp: float=3.0, reverb_amount: float=0.0, drive_amount: float=0.0, pitch_shift: int=0, delay_time: float=0.0, humanize: bool=False):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1: audio_data = audio_data.T 
    board = Pedalboard([
        PitchShift(semitones=pitch_shift) if pitch_shift != 0 else None,
        NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250) if gate else None,
        HighpassFilter(cutoff_frequency_hz=300 if bass <= -10 else 80),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass), 
        HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble), 
        Distortion(drive_db=drive_amount) if drive_amount > 0 else None,
        Compressor(threshold_db=-15, ratio=comp, attack_ms=2.0, release_ms=100),
        Delay(delay_seconds=delay_time, feedback=0.3, mix=0.4) if delay_time > 0 else None,
        Reverb(room_size=0.3 if delay_time > 0 else 0.1, dry_level=1.0, wet_level=reverb_amount) if reverb_amount > 0 else None,
        Chorus(rate_hz=0.5, depth=0.05, mix=0.1) if humanize else None,
        Limiter(threshold_db=-1.0)
    ])
    board = Pedalboard([effect for effect in board if effect is not None])
    effected_audio = board(audio_data, sample_rate)
    if len(effected_audio.shape) > 1: effected_audio = effected_audio.T
    sf.write(output_path, effected_audio, sample_rate)
    return output_path

# --- NEW: VIDEO EDITING ---
async def process_video_edit(video_path: str, audio_path: str, output_path: str, trim_start: float = 0.0, trim_end: float = None, volume_ratio: float = 1.0, add_watermark: str = ""):
    """
    Automated Video Editing Engine using MoviePy.
    Runs inside a lock to prevent CPU overload during rendering.
    """
    async with cpu_render_lock:
        # Load Video
        video = VideoFileClip(video_path)
        
        # 1. Trimming
        if trim_end is None or trim_end <= 0:
            trim_end = video.duration
        video = video.subclip(trim_start, trim_end)
        
        # 2. Audio Replacement/Merging
        if audio_path and os.path.exists(audio_path):
            new_audio = AudioFileClip(audio_path)
            
            # If the new audio is shorter than video, we loop it or just let it end.
            # Here, we just set it. It will stop when video ends or audio ends.
            video = video.set_audio(new_audio)
        
        # 3. Volume Adjustment
        if volume_ratio != 1.0:
            video = video.volumex(volume_ratio)
            
        # 4. Add Watermark / Text (Basic implementation)
        if add_watermark:
            # Create text clip. Note: requires ImageMagick installed on system for TextClip
            try:
                txt_clip = TextClip(add_watermark, fontsize=50, color='white')
                txt_clip = txt_clip.set_position(('right','bottom')).set_duration(video.duration)
                video = CompositeVideoClip([video, txt_clip])
            except Exception as e:
                print(f"Warning: Text watermark failed (ImageMagick might be missing): {e}")

        # Render Output
        video.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            temp_audiofile=os.path.join(TEMP_DIR, f"temp-audio-{uuid.uuid4().hex[:6]}.m4a"),
            remove_temp=True,
            logger=None # Disable stdout logs to keep console clean
        )
        
        video.close()
        if 'new_audio' in locals():
            new_audio.close()

# ==========================================
# 3. FastAPI Setup (Production Ready)
# ==========================================
app = FastAPI(title="AI Media Studio API (Audio & Video)")
mcp = FastMCP("Media_Studio_MCP")
app.mount("/sse", mcp.get_starlette_app())

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AI Media Studio App", "auth_required": bool(API_KEY_SECRET)}

# --- AUDIO ENDPOINT ---
@app.post("/api/tts/generate")
async def api_generate_tts(
    engine: str = Form(...),
    text: str = Form(...),
    language: str = Form("en"),
    mode: str = Form("standard"),
    speaker_model: str = Form("Default Model"), 
    speed: float = Form(1.0),
    apply_humanize: bool = Form(False),
    reference_audio: UploadFile = File(None),
    api_key: str = Depends(verify_api_key) 
):
    cleanup_old_files()
    if engine not in ["omnivoice", "cosyvoice"]: raise HTTPException(status_code=400, detail="Invalid Engine")

    job_id = str(uuid.uuid4())
    ref_audio_path = None
    if reference_audio:
        ref_ext = reference_audio.filename.split('.')[-1] if '.' in reference_audio.filename else 'wav'
        ref_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref.{ref_ext}")
        with open(ref_audio_path, "wb") as f: f.write(await reference_audio.read())

    raw_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_raw.wav")
    try:
        await generate_tts_safely(engine, mode, text, language, ref_audio_path, raw_output_path, speed, speaker_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(path=raw_output_path, media_type="audio/wav", filename=f"{engine}_raw.wav")

# --- NEW: VIDEO ENDPOINT (n8n Automation) ---
@app.post("/api/video/edit")
async def api_video_edit(
    video_file: UploadFile = File(...),
    audio_file: UploadFile = File(None), # Optional: Overwrite with TTS audio
    trim_start: float = Form(0.0),
    trim_end: float = Form(0.0), # 0 means till end
    volume_ratio: float = Form(1.0),
    watermark_text: str = Form(""),
    api_key: str = Depends(verify_api_key)
):
    cleanup_old_files()
    job_id = str(uuid.uuid4())
    
    # Save Video
    vid_ext = video_file.filename.split('.')[-1] if '.' in video_file.filename else 'mp4'
    vid_path = os.path.join(UPLOAD_DIR, f"{job_id}_vid.{vid_ext}")
    with open(vid_path, "wb") as f: f.write(await video_file.read())
        
    # Save Audio if provided
    aud_path = None
    if audio_file:
        aud_ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'wav'
        aud_path = os.path.join(UPLOAD_DIR, f"{job_id}_aud.{aud_ext}")
        with open(aud_path, "wb") as f: f.write(await audio_file.read())

    out_vid_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.mp4")

    try:
        await process_video_edit(vid_path, aud_path, out_vid_path, trim_start, trim_end, volume_ratio, watermark_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video rendering failed: {str(e)}")

    if not os.path.exists(out_vid_path):
        raise HTTPException(status_code=500, detail="Video was not generated.")

    return FileResponse(path=out_vid_path, media_type="video/mp4", filename="edited_video.mp4")


# ==========================================
# 4. Gradio UI Setup
# ==========================================
def gradio_video_edit(video_in, audio_in, trim_start, trim_end, watermark, current_logs):
    cleanup_old_files()
    if not video_in: return None, append_log(current_logs, "❌ ERROR: No video provided.")
    
    logs = append_log(current_logs, "🎬 START VIDEO EDIT: Rendering started (CPU locked)...")
    yield None, logs
    
    job_id = str(uuid.uuid4())
    out_vid_path = os.path.join(OUTPUT_DIR, f"{job_id}_ui_output.mp4")
    
    try:
        asyncio.run(process_video_edit(
            video_path=video_in, 
            audio_path=audio_in, 
            output_path=out_vid_path, 
            trim_start=trim_start, 
            trim_end=trim_end if trim_end > 0 else None,
            add_watermark=watermark
        ))
        logs = append_log(logs, "✅ SUCCESS: Video rendered successfully.")
        yield out_vid_path, logs
    except Exception as e:
        logs = append_log(logs, f"❌ VIDEO ERROR: {str(e)}")
        yield None, logs


def gradio_tts(tts_mode, engine, custom_speaker, language, speed, text, instruct_prompt, ref_audio, current_logs):
    cleanup_old_files()
    if not text.strip(): return None, None, "❌ เกิดข้อผิดพลาด", append_log(current_logs, "❌ ERROR: ข้อความว่างเปล่า")

    logs = append_log(current_logs, f"🚀 START: Model={custom_speaker}, Chunking Long Text if needed...")
    yield None, None, "⏳ กำลังคิวและประมวลผล (GPU Lock Active)...", logs
    
    output_audio = os.path.join(OUTPUT_DIR, f"gradio_{int(time.time())}.wav")
    try:
        asyncio.run(generate_tts_safely(engine, tts_mode, text, language, ref_audio, output_audio, speed, custom_speaker))
        logs = append_log(logs, "✅ SUCCESS: สร้างเสียงสำเร็จ")
        yield output_audio, output_audio, "✅ สำเร็จ", logs
    except Exception as e:
        logs = append_log(logs, f"❌ FATAL ERROR: {str(e)}")
        yield None, None, f"❌ ล้มเหลว: {str(e)}", logs

def update_sliders_from_preset(preset_name):
    if preset_name in STUDIO_PRESETS:
        p = STUDIO_PRESETS[preset_name]
        return (gr.update(value=p['desc']), gr.update(value=p['gate']), gr.update(value=p['bass']), 
                gr.update(value=p['treble']), gr.update(value=p['comp']), gr.update(value=p['reverb']),
                gr.update(value=p['drive']), gr.update(value=p['pitch']), gr.update(value=p['delay']))
    return [gr.update()] * 9

def gradio_studio(input_audio, preset, humanize, export_format, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, current_logs):
    cleanup_old_files()
    if not input_audio: return None, "❌ ไม่พบไฟล์", append_log(current_logs, "❌ ERROR: No input file")
    try:
        logs = append_log(current_logs, f"⚙️ START STUDIO: {preset}")
        ext = export_format.lower()
        output_file = os.path.join(OUTPUT_DIR, f"studio_{int(time.time())}.{ext}")
        apply_studio_mastering(input_audio, output_file, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, humanize)
        logs = append_log(logs, f"✅ SUCCESS: Exported as {ext.upper()}")
        return output_file, "✅ สำเร็จ", logs
    except Exception as e:
        return None, str(e), append_log(current_logs, f"❌ ERROR: {str(e)}")


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎬 AI Media Studio (Audio & Video Production)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            # --- TAB 1: AUDIO TTS ---
            with gr.Tab("🎙️ 1. TTS Generation"):
                with gr.Row():
                    with gr.Column():
                        engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="Engine")
                        tts_mode = gr.Radio(choices=["Standard", "Zero-Shot (Voice Cloning)", "Instruct (Emotion)"], value="Standard", label="Mode")
                        dynamic_speakers = get_available_speakers()
                        speaker_dropdown = gr.Dropdown(choices=dynamic_speakers, value=dynamic_speakers[0], label="Speaker Checkpoint")
                        lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="Language")
                        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                        text_input = gr.Textbox(label="Text Prompt", lines=4)
                        instruct_prompt = gr.Textbox(label="Instruction Prompt")
                        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
                        submit_btn = gr.Button("🚀 Generate Speech", variant="primary")
                    with gr.Column():
                        status_output = gr.Markdown("🟢 พร้อมใช้งาน")
                        output_audio = gr.Audio(label="Raw Audio", interactive=False)

            # --- TAB 2: AUDIO MASTERING ---
            with gr.Tab("🎛️ 2. Audio Mastering"):
                with gr.Row():
                    with gr.Column():
                        raw_audio_input = gr.Audio(label="Input Audio", type="filepath")
                        preset_dropdown = gr.Dropdown(choices=list(STUDIO_PRESETS.keys()), value=list(STUDIO_PRESETS.keys())[0], label="Studio Preset")
                        preset_desc = gr.Markdown(f"*{STUDIO_PRESETS[list(STUDIO_PRESETS.keys())[0]]['desc']}*")
                        humanize_checkbox = gr.Checkbox(value=False, label="🤖 ➔ 🧑 Humanize")
                        
                        with gr.Accordion("⚙️ ปรับแต่งแบบละเอียด", open=False):
                            bass_boost = gr.Slider(minimum=-15, maximum=15, value=5.0, label="Bass")
                            treble_boost = gr.Slider(minimum=-15, maximum=15, value=3.5, label="Treble")
                            comp_ratio = gr.Slider(minimum=1, maximum=10, value=3.5, label="Compression")
                            enable_gate = gr.Checkbox(value=True, label="Noise Gate")
                            reverb_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, label="Reverb")
                            delay_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, label="Delay")
                            drive_amount = gr.Slider(minimum=0.0, maximum=30.0, value=0.0, label="Distortion")
                            pitch_shift = gr.Slider(minimum=-12, maximum=12, value=0, label="Pitch")

                        export_format = gr.Radio(choices=["WAV", "FLAC"], value="WAV", label="Format")
                        process_btn = gr.Button("🎧 ประมวลผล", variant="primary")
                    with gr.Column():
                        studio_status = gr.Markdown("🟢 รอรับไฟล์")
                        studio_audio_output = gr.Audio(label="Mastered Audio", interactive=False)

            # --- NEW TAB 3: VIDEO EDITING ---
            with gr.Tab("🎞️ 3. Video Automation"):
                gr.Markdown("ตัดต่อวิดีโออัตโนมัติ (Automated Video Editing Engine)")
                with gr.Row():
                    with gr.Column():
                        video_input = gr.Video(label="🎥 อัปโหลด Footage วิดีโอ", sources=["upload"])
                        audio_input_video = gr.Audio(label="🎵 ไฟล์เสียงพากย์ (จะถูกนำไปสวมแทนเสียงเดิมในวิดีโอ)", type="filepath")
                        
                        gr.Markdown("### ✂️ เครื่องมือตัดต่อ (Editing Tools)")
                        with gr.Row():
                            trim_start = gr.Number(value=0, label="ตัดวินาทีเริ่มต้น (Start Trim)")
                            trim_end = gr.Number(value=0, label="ตัดวินาทีสิ้นสุด (End Trim, 0 = ไม่ตัด)")
                        
                        watermark_text = gr.Textbox(label="ลายน้ำข้อความ (Watermark / Text overlay)", placeholder="e.g., @SatangTheBank")
                        
                        video_process_btn = gr.Button("🎬 เริ่มประมวลผลวิดีโอ (Render Video)", variant="primary")
                        
                    with gr.Column():
                        video_output = gr.Video(label="✅ วิดีโอที่ตัดต่อเสร็จแล้ว", interactive=False)

        # --- LOGS SIDEBAR ---
        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=30, interactive=False, value="[System] Initialized.")
            clear_log_btn = gr.Button("🗑️ Clear")

    # --- Event Wiring ---
    submit_btn.click(
        fn=gradio_tts,
        inputs=[tts_mode, engine_dropdown, speaker_dropdown, lang_dropdown, speed_slider, text_input, instruct_prompt, ref_audio_input, system_logs_state],
        outputs=[output_audio, raw_audio_input, status_output, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    preset_dropdown.change(
        fn=update_sliders_from_preset,
        inputs=[preset_dropdown],
        outputs=[preset_desc, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount]
    )

    process_btn.click(
        fn=gradio_studio,
        inputs=[raw_audio_input, preset_dropdown, humanize_checkbox, export_format, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount, system_logs_state],
        outputs=[studio_audio_output, studio_status, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    
    # Video Process Event
    video_process_btn.click(
        fn=gradio_video_edit,
        inputs=[video_input, audio_input_video, trim_start, trim_end, watermark_text, system_logs_state],
        outputs=[video_output, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    clear_log_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[system_logs_state, logs_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)