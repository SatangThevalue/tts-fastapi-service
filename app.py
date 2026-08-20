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
UPLOAD_DIR = os.path.join(TEMP_DIR, "tts_uploads")
OUTPUT_DIR = os.path.join(TEMP_DIR, "tts_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
SPEAKERS_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models", "speakers")
os.makedirs(SPEAKERS_DIR, exist_ok=True)

# 🚨 PRODUCTION FEATURE 1: API Security 🚨
# Only allow requests with the correct API Key (if configured)
API_KEY_SECRET = os.environ.get("TTS_API_KEY", "") 
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY_SECRET and api_key != API_KEY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key

# 🚨 PRODUCTION FEATURE 2: GPU Concurrency Control 🚨
# Limit GPU access to 1 task at a time to prevent Out-Of-Memory (OOM) crashes
gpu_lock = asyncio.Lock()

# LLM Auto-Tagging Configuration
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "")
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
# 2. LLM Auto-Emotion Tagger 
# ==========================================
async def auto_tag_emotion_llm(text: str, engine: str, emotion: str) -> str:
    if not LLM_API_KEY: raise ValueError("Please provide an LLM API Key.")
    
    system_prompt = (
        "You are an AI scriptwriter. Modify the text to fit the emotion by inserting tags. "
        "OmniVoice Tags: [laughter], [sigh], [surprise-ah]. CosyVoice format: <|emotion|> Text."
    )
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Emotion: {emotion}\nText: {text}"}],
        "temperature": 0.3
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

def gradio_auto_tag(text, engine, emotion, current_logs, api_key, api_url, api_model):
    if not text.strip(): return text, append_log(current_logs, "❌ ERROR: No text provided.")
    global LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    LLM_API_KEY = api_key
    LLM_BASE_URL = api_url
    LLM_MODEL = api_model
    logs = append_log(current_logs, f"🤖 LLM Analyzing emotion '{emotion}'...")
    try:
        tagged_text = asyncio.run(auto_tag_emotion_llm(text, engine, emotion))
        return tagged_text, append_log(logs, f"✅ LLM Success.")
    except Exception as e:
        return text, append_log(logs, f"❌ LLM Error: {str(e)}")

def insert_tag_at_cursor(current_text, tag):
    return f"{current_text.rstrip()} {tag} "

# ==========================================
# 3. Core AI Generation (With Long-Text Handling)
# ==========================================
async def _generate_audio_chunk(engine: str, mode: str, text_chunk: str, lang: str, ref_path: str, speed: float, speaker_model: str):
    """Generates audio for a short chunk of text. Wrapped in GPU Lock later."""
    await asyncio.sleep(1) # Simulate GPU time
    sample_rate = 44100
    duration = max(1.0, len(text_chunk) * 0.1) 
    audio_data = np.zeros((int(sample_rate * duration), 1))
    return audio_data, sample_rate

# 🚨 PRODUCTION FEATURE 3: Long-Text Chunking & Concatenation 🚨
async def generate_tts_safely(engine: str, mode: str, full_text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0, speaker_model: str = "Default Model"):
    chunks = re.split(r'(?<=[.!?\\n])\s+', full_text.strip())
    chunks = [c for c in chunks if c.strip()]
    if not chunks:
        chunks = [full_text]

    all_audio_arrays = []
    sample_rate = 44100

    # Process each chunk safely using GPU Lock
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

def apply_studio_mastering(
    input_path: str, output_path: str, gate: bool=True, bass: float=4.5, treble: float=3.0, 
    comp: float=3.0, reverb_amount: float=0.0, drive_amount: float=0.0, pitch_shift: int=0, 
    delay_time: float=0.0, humanize: bool=False
):
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

# ==========================================
# 4. FastAPI Setup (Production Ready n8n endpoints)
# ==========================================
app = FastAPI(title="TTS Unified API (Production)")
mcp = FastMCP("TTS_Studio_MCP")
app.mount("/sse", mcp.get_starlette_app())

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "TTS Unified App", "auth_required": bool(API_KEY_SECRET)}

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
    # 🚨 API SECURED 🚨
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


# ==========================================
# 5. Gradio UI Setup
# ==========================================
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

# Build Gradio Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ Unified AI Voice Engine (Production Edition)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. TTS Generation & Emotion"):
                with gr.Row():
                    with gr.Column():
                        engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="Engine")
                        tts_mode = gr.Radio(choices=["Standard", "Zero-Shot (Voice Cloning)", "Instruct (Emotion)"], value="Instruct (Emotion)", label="Mode")
                        dynamic_speakers = get_available_speakers()
                        speaker_dropdown = gr.Dropdown(choices=dynamic_speakers, value=dynamic_speakers[0], label="เลือกไฟล์น้ำหนัก Fine-tune")
                        
                        with gr.Row():
                            lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="Language")
                            speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")

                        with gr.Accordion("🤖 LLM Auto-Emotion Agent", open=False):
                            with gr.Row():
                                llm_emotion_intent = gr.Dropdown(choices=["ตื่นเต้น (Excited)", "เศร้า (Sad)", "โกรธ (Angry)", "หัวเราะ (Laughing)", "ตกใจ (Surprised)", "ทางการ (Formal)"], value="ตื่นเต้น (Excited)", label="อารมณ์")
                                llm_auto_tag_btn = gr.Button("✨ ให้ AI ใส่ Tag อารมณ์ให้", variant="secondary")
                            with gr.Accordion("⚙️ LLM API Settings", open=False):
                                llm_api_url = gr.Textbox(label="API Base URL", value="https://api.openai.com/v1")
                                llm_api_key = gr.Textbox(label="API Key", type="password")
                                llm_api_model = gr.Textbox(label="Model", value="gpt-4o-mini")

                        text_input = gr.Textbox(label="Text Prompt (รองรับข้อความยาว ระบบจะหั่นประโยคให้อัตโนมัติ)", lines=6)
                        
                        gr.Markdown("*ปุ่มลัดแทรก Action Tags (สำหรับ OmniVoice)*")
                        with gr.Row():
                            tag_laugh_btn = gr.Button("😂 [laughter]", size="sm")
                            tag_sigh_btn = gr.Button("😮‍💨 [sigh]", size="sm")
                            tag_surprise_btn = gr.Button("😲 [surprise-ah]", size="sm")
                            tag_angry_btn = gr.Button("😤 [dissatisfaction]", size="sm")

                        instruct_prompt = gr.Textbox(label="Instruction Prompt (สำหรับโหมด Instruct)")
                        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
                        submit_btn = gr.Button("🚀 Generate Speech", variant="primary")
                    
                    with gr.Column():
                        status_output = gr.Markdown("🟢 พร้อมใช้งาน")
                        output_audio = gr.Audio(label="Raw Audio", interactive=False)

            with gr.Tab("🎛️ 2. Studio Presets & Effects"):
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

        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=30, interactive=False, value="[System] Initialized.")
            refresh_models_btn = gr.Button("🔄 Refresh Speaker Models", variant="secondary")
            clear_log_btn = gr.Button("🗑️ Clear")

    # --- Event Wiring ---
    
    def refresh_speakers():
        new_list = get_available_speakers()
        return gr.update(choices=new_list, value=new_list[0]), "🔄 Refreshed available models from disk."

    refresh_models_btn.click(fn=refresh_speakers, inputs=None, outputs=[speaker_dropdown, logs_display])

    tag_laugh_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[laughter]"), inputs=[text_input], outputs=[text_input])
    tag_sigh_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[sigh]"), inputs=[text_input], outputs=[text_input])
    tag_surprise_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[surprise-ah]"), inputs=[text_input], outputs=[text_input])
    tag_angry_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[dissatisfaction-hnn]"), inputs=[text_input], outputs=[text_input])

    llm_auto_tag_btn.click(
        fn=gradio_auto_tag,
        inputs=[text_input, engine_dropdown, llm_emotion_intent, system_logs_state, llm_api_key, llm_api_url, llm_api_model],
        outputs=[text_input, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    preset_dropdown.change(
        fn=update_sliders_from_preset,
        inputs=[preset_dropdown],
        outputs=[preset_desc, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount]
    )

    submit_btn.click(
        fn=gradio_tts,
        inputs=[tts_mode, engine_dropdown, speaker_dropdown, lang_dropdown, speed_slider, text_input, instruct_prompt, ref_audio_input, system_logs_state],
        outputs=[output_audio, raw_audio_input, status_output, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    process_btn.click(
        fn=gradio_studio,
        inputs=[raw_audio_input, preset_dropdown, humanize_checkbox, export_format, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount, system_logs_state],
        outputs=[studio_audio_output, studio_status, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    
    clear_log_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[system_logs_state, logs_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)