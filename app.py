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

# CPU-based TTS
import edge_tts

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

API_KEY_SECRET=os.environ.get("TTS_API_KEY", "") 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY_SECRET and api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

gpu_lock = asyncio.Lock()
cpu_render_lock = asyncio.Lock()

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY=os.environ.get("LLM_API_KEY", "")
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

STUDIO_PRESETS = {
    "🎙️ Podcast Studio": {"bass": 5.0, "treble": 3.5, "comp": 3.5, "reverb": 0.05, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "นุ่มลึก มีน้ำหนัก ฟังสบาย"},
    "📖 Audiobook Pro": {"bass": 2.0, "treble": 2.0, "comp": 2.5, "reverb": 0.15, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "ใสสะอาด มีมิติเสียงก้องนิดๆ"},
    "🗣️ Natural Human": {"bass": 1.0, "treble": 1.5, "comp": 1.5, "reverb": 0.08, "gate": False, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "ธรรมชาติ ไม่บีบอัดมาก"}
}

# ==========================================
# 2. AI Audio & Video Operations
# ==========================================

async def _generate_audio_chunk(engine: str, mode: str, text_chunk: str, lang: str, ref_path: str, speed: float, speaker_model: str):
    """Generates audio for a chunk. Handles both GPU models (mocked) and CPU-fast models."""
    sample_rate = 44100
    
    if engine == "EdgeTTS (Fast CPU)":
        # 🌟 NEW CPU-BASED TTS (No GPU required, lightning fast)
        voice = "th-TH-PremwadeeNeural" if "th" in lang.lower() else "en-US-AriaNeural"
        
        # Edge-tts uses string formats like "+0%" or "+10%" for speed
        speed_percent = int((speed - 1.0) * 100)
        speed_str = f"+{speed_percent}%" if speed_percent >= 0 else f"{speed_percent}%"
        
        communicate = edge_tts.Communicate(text_chunk, voice, rate=speed_str)
        
        temp_mp3 = os.path.join(TEMP_DIR, f"edge_{uuid.uuid4().hex}.mp3")
        await communicate.save(temp_mp3)
        
        audio_data, sr = sf.read(temp_mp3)
        os.remove(temp_mp3)
        
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(-1, 1)
            
        return audio_data, sr
        
    else:
        # OmniVoice / CosyVoice (GPU Mock)
        await asyncio.sleep(1) 
        duration = max(1.0, len(text_chunk) * 0.1) 
        audio_data = np.zeros((int(sample_rate * duration), 1))
        return audio_data, sample_rate


async def generate_tts_safely(engine: str, mode: str, full_text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0, speaker_model: str = "Default Model"):
    chunks = re.split(r'(?<=[.!?\n])\s+', full_text.strip())
    chunks = [c for c in chunks if c.strip()]
    if not chunks: chunks = [full_text]
    all_audio_arrays = []
    sample_rate = 44100
    
    if engine == "EdgeTTS (Fast CPU)":
        for chunk in chunks:
            if not chunk.strip(): continue
            audio_data, sr = await _generate_audio_chunk(engine, mode, chunk, lang, ref_path, speed, speaker_model)
            sample_rate = sr
            all_audio_arrays.append(audio_data)
            pause = np.zeros((int(sample_rate * 0.2), audio_data.shape[1] if len(audio_data.shape) > 1 else 1))
            all_audio_arrays.append(pause)
    else:
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

async def process_video_edit(
    video_path: str, 
    audio_path: str, 
    output_path: str, 
    trim_start: float = 0.0, 
    trim_end: float = None, 
    mute_original_audio: bool = False,
    short_video_format: bool = True,  
    add_watermark: str = "",
    text_lines: str = "" 
):
    async with cpu_render_lock:
        video = VideoFileClip(video_path)
        
        if trim_end is None or trim_end <= 0:
            trim_end = video.duration
        video = video.subclip(trim_start, trim_end)

        if mute_original_audio:
            video = video.without_audio()

        if short_video_format:
            target_w, target_h = 1080, 1920
            video_aspect = video.w / video.h
            target_aspect = target_w / target_h
            
            if video_aspect > target_aspect:
                new_w = int(video.h * target_aspect)
                x_center = video.w / 2
                video = video.crop(x_center=x_center, width=new_w, height=video.h)
            else:
                new_h = int(video.w / target_aspect)
                y_center = video.h / 2
                video = video.crop(y_center=y_center, width=video.w, height=new_h)
            
            video = video.resize(newsize=(target_w, target_h))

        final_audio = None
        if audio_path and os.path.exists(audio_path):
            new_audio = AudioFileClip(audio_path)
            new_audio = new_audio.set_duration(min(new_audio.duration, video.duration))
            
            if mute_original_audio or video.audio is None:
                final_audio = new_audio
            else:
                final_audio = CompositeAudioClip([video.audio.volumex(0.3), new_audio])
                
        if final_audio:
            video = video.set_audio(final_audio)

        clips_to_composite = [video]
        
        if text_lines.strip():
            lines = text_lines.split("\n")
            lines = [l.strip() for l in lines if l.strip()]
            try:
                box_width = int(video.w * 0.85) 
                start_y = int(video.h * 0.25)   
                spacing = 30                    
                current_y = start_y
                for idx, line in enumerate(lines):
                    txt_clip = TextClip(line, fontsize=45, color='black', bg_color='white', method='caption', align='center', size=(box_width, None))
                    txt_clip = txt_clip.set_position(('center', current_y)).set_duration(video.duration)
                    clips_to_composite.append(txt_clip)
                    current_y += txt_clip.h + spacing
            except Exception as e:
                print(f"Warning: Text overlay failed: {e}")

        if add_watermark:
            try:
                wm_clip = TextClip(add_watermark, fontsize=40, color='white', bg_color='transparent')
                wm_clip = wm_clip.set_position(('right','bottom')).set_duration(video.duration).margin(bottom=50, right=50, opacity=0)
                clips_to_composite.append(wm_clip)
            except:
                pass

        if len(clips_to_composite) > 1:
            video = CompositeVideoClip(clips_to_composite)

        video.write_videofile(
            output_path, 
            codec="libx264", 
            audio_codec="aac",
            temp_audiofile=os.path.join(TEMP_DIR, f"temp-audio-{uuid.uuid4().hex[:6]}.m4a"),
            remove_temp=True,
            fps=30, 
            logger=None
        )
        
        video.close()
        if 'final_audio' in locals() and final_audio:
            final_audio.close()


# ==========================================
# 3. FastAPI Setup (Production Ready)
# ==========================================
app = FastAPI(title="AI Media Studio API (Audio & Video)")
mcp = FastMCP("Media_Studio_MCP")
app.mount("/sse", mcp.get_starlette_app())

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "AI Media Studio App", "auth_required": bool(API_KEY_SECRET)}

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
    if engine not in ["omnivoice", "cosyvoice", "EdgeTTS (Fast CPU)"]: 
        raise HTTPException(status_code=400, detail="Invalid Engine")

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

@app.post("/api/video/edit")
async def api_video_edit(
    video_file: UploadFile = File(None),
    video_local_path: str = Form(""),
    audio_file: UploadFile = File(None), 
    audio_local_path: str = Form(""),
    trim_start: float = Form(0.0),
    trim_end: float = Form(0.0), 
    mute_original_audio: bool = Form(True), 
    short_video_format: bool = Form(True),  
    text_lines: str = Form(""),             
    watermark_text: str = Form(""),
    return_local_path: bool = Form(False), 
    api_key: str = Depends(verify_api_key)
):
    cleanup_old_files()
    job_id = str(uuid.uuid4())
    
    vid_path = None
    if video_local_path and os.path.exists(video_local_path):
        vid_path = video_local_path
    elif video_file:
        vid_ext = video_file.filename.split('.')[-1] if '.' in video_file.filename else 'mp4'
        vid_path = os.path.join(UPLOAD_DIR, f"{job_id}_vid.{vid_ext}")
        with open(vid_path, "wb") as f: f.write(await video_file.read())
    else:
        raise HTTPException(status_code=400, detail="Must provide either video_file or video_local_path")
        
    aud_path = None
    if audio_local_path and os.path.exists(audio_local_path):
        aud_path = audio_local_path
    elif audio_file:
        aud_ext = audio_file.filename.split('.')[-1] if '.' in audio_file.filename else 'wav'
        aud_path = os.path.join(UPLOAD_DIR, f"{job_id}_aud.{aud_ext}")
        with open(aud_path, "wb") as f: f.write(await audio_file.read())

    out_vid_path = os.path.join(OUTPUT_DIR, f"{job_id}_output.mp4")

    try:
        await process_video_edit(
            video_path=vid_path, 
            audio_path=aud_path, 
            output_path=out_vid_path, 
            trim_start=trim_start, 
            trim_end=trim_end, 
            mute_original_audio=mute_original_audio,
            short_video_format=short_video_format,
            add_watermark=watermark_text,
            text_lines=text_lines
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video rendering failed: {str(e)}")

    if not os.path.exists(out_vid_path):
        raise HTTPException(status_code=500, detail="Video was not generated.")

    if return_local_path:
        return JSONResponse(content={"status": "success", "output_path": out_vid_path})
    
    return FileResponse(path=out_vid_path, media_type="video/mp4", filename="edited_video.mp4")


# ==========================================
# 4. Gradio UI Setup
# ==========================================
def gradio_video_edit(video_in, audio_in, trim_start, trim_end, mute_orig, force_916, text_lines, watermark, current_logs):
    cleanup_old_files()
    if not video_in: return None, append_log(current_logs, "❌ ERROR: No video provided.")
    logs = append_log(current_logs, "🎬 START VIDEO EDIT: Rendering started (CPU locked)...")
    if force_916: logs = append_log(logs, "🎞️ FORMAT: Cropping to 9:16")
    if mute_orig: logs = append_log(logs, "🔇 AUDIO: Original video muted")
    yield None, logs
    
    job_id = str(uuid.uuid4())
    out_vid_path = os.path.join(OUTPUT_DIR, f"{job_id}_ui_output.mp4")
    
    try:
        asyncio.run(process_video_edit(video_path=video_in, audio_path=audio_in, output_path=out_vid_path, trim_start=trim_start, trim_end=trim_end if trim_end > 0 else None, mute_original_audio=mute_orig, short_video_format=force_916, add_watermark=watermark, text_lines=text_lines))
        logs = append_log(logs, "✅ SUCCESS: Video rendered successfully.")
        yield out_vid_path, logs
    except Exception as e:
        logs = append_log(logs, f"❌ VIDEO ERROR: {str(e)}")
        yield None, logs

def gradio_tts(tts_mode, engine, custom_speaker, language, speed, text, instruct_prompt, ref_audio, current_logs):
    cleanup_old_files()
    if not text.strip(): return None, None, "❌", append_log(current_logs, "❌ ERROR: No text")
    logs = append_log(current_logs, f"🚀 START: Engine={engine}, Model={custom_speaker}")
    yield None, None, "⏳ กำลังประมวลผล...", logs
    
    out_path = os.path.join(OUTPUT_DIR, f"gradio_tts_{int(time.time())}.wav")
    
    try:
        asyncio.run(generate_tts_safely(engine, tts_mode, text, language, ref_audio, out_path, speed, custom_speaker))
        logs = append_log(logs, "✅ SUCCESS: สร้างเสียงสำเร็จ")
        yield out_path, out_path, "✅ สำเร็จ", logs
    except Exception as e:
        logs = append_log(logs, f"❌ ERROR: {str(e)}")
        yield None, None, f"❌ {str(e)}", logs

def update_sliders_from_preset(preset_name):
    if preset_name in STUDIO_PRESETS:
        p = STUDIO_PRESETS[preset_name]
        return (gr.update(value=p['desc']), gr.update(value=p['gate']), gr.update(value=p['bass']), 
                gr.update(value=p['treble']), gr.update(value=p['comp']), gr.update(value=p['reverb']),
                gr.update(value=p['drive']), gr.update(value=p['pitch']), gr.update(value=p['delay']))
    return [gr.update()] * 9

def gradio_studio(input_audio, preset, humanize, export_format, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, current_logs):
    cleanup_old_files()
    if not input_audio: return None, "❌", append_log(current_logs, "❌ No input")
    try:
        logs = append_log(current_logs, f"⚙️ START STUDIO: {preset}")
        ext = export_format.lower()
        output_file = os.path.join(OUTPUT_DIR, f"studio_{int(time.time())}.{ext}")
        apply_studio_mastering(input_audio, output_file, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, humanize)
        logs = append_log(logs, f"✅ SUCCESS")
        return output_file, "✅ สำเร็จ", logs
    except Exception as e:
        return None, str(e), append_log(current_logs, f"❌ ERROR: {str(e)}")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎬 AI Media Studio (Audio & Vertical Video Production)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. Audio Tools"):
                gr.Markdown("*(สร้างเสียงพากย์ด้วย AI ทั้งแบบ GPU และ CPU ที่รวดเร็ว)*")
                with gr.Row():
                    with gr.Column():
                        engine_dropdown = gr.Radio(choices=["CosyVoice 3.0", "OmniVoice", "EdgeTTS (Fast CPU)"], value="EdgeTTS (Fast CPU)", label="Engine")
                        tts_mode = gr.Radio(choices=["Standard", "Instruct (Emotion)"], value="Standard", label="Mode")
                        dynamic_speakers = get_available_speakers()
                        speaker_dropdown = gr.Dropdown(choices=dynamic_speakers, value=dynamic_speakers[0], label="Speaker Model")
                        lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="Language")
                        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
                        
                        text_input = gr.Textbox(label="Text Prompt", lines=4)
                        instruct_prompt = gr.Textbox(label="Instruction Prompt (Optional)")
                        
                        # Note: We keep ref_audio in case user chooses Zero-Shot on UI, even though standard UI hides it. We'll pass None or dummy.
                        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath", visible=False)
                        
                        submit_btn = gr.Button("🚀 Generate Speech", variant="primary")
                    with gr.Column():
                        status_output = gr.Markdown("🟢 พร้อมใช้งาน")
                        output_audio = gr.Audio(label="Output Audio", interactive=False)

            with gr.Tab("📱 2. Social Media Video Automator"):
                gr.Markdown("สร้างวิดีโอ 9:16 แนวตั้งอัตโนมัติ สำหรับ **Facebook Reels / TikTok / YouTube Shorts**")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="🎥 อัปโหลด Footage วิดีโอพื้นหลัง", sources=["upload"])
                        audio_input_video = gr.Audio(label="🎵 ไฟล์เสียงพากย์ / BGM", type="filepath")
                        
                        gr.Markdown("### ⚙️ Video Settings")
                        with gr.Row():
                            force_916 = gr.Checkbox(value=True, label="📱 บังคับอัตราส่วน 9:16 (แนวตั้ง)")
                            mute_orig = gr.Checkbox(value=True, label="🔇 ลบเสียงดั้งเดิมของวิดีโอ")
                        
                        with gr.Row():
                            trim_start = gr.Number(value=0, label="ตัดหัว (เริ่มวินาทีที่)")
                            trim_end = gr.Number(value=0, label="ตัดท้าย (สิ้นสุดวินาทีที่, 0 = ไม่ตัด)")
                        
                        text_lines = gr.Textbox(
                            label="รายการข้อความ (บรรทัดละ 1 กล่องข้อความ)", 
                            lines=8, 
                            placeholder="1. เริ่ม DCA S&P500\\n2. ซื้อประกันสุขภาพ\\n3. สร้าง Emergency Fund..."
                        )
                        watermark_text = gr.Textbox(label="ลายน้ำ (มุมขวาล่าง)", placeholder="@SatangTheBank")
                        video_process_btn = gr.Button("🎬 เริ่มประมวลผลวิดีโอ (Render Video)", variant="primary", size="lg")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 🌟 ผลลัพธ์วิดีโอ")
                        video_output = gr.Video(label="✅ วิดีโอที่ตัดต่อเสร็จแล้ว พร้อมโพสต์", interactive=False)

        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=30, interactive=False, value="[System] Initialized.")
            clear_log_btn = gr.Button("🗑️ Clear")

    submit_btn.click(
        fn=gradio_tts, 
        inputs=[tts_mode, engine_dropdown, speaker_dropdown, lang_dropdown, speed_slider, text_input, instruct_prompt, ref_audio_input, system_logs_state], 
        outputs=[output_audio, output_audio, status_output, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    
    video_process_btn.click(
        fn=gradio_video_edit,
        inputs=[video_input, audio_input_video, trim_start, trim_end, mute_orig, force_916, text_lines, watermark_text, system_logs_state],
        outputs=[video_output, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    clear_log_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[system_logs_state, logs_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)