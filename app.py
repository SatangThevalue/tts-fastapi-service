import os
import time
import glob
import uuid
import asyncio
import tempfile
from datetime import datetime

import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter, Reverb, Chorus, Distortion, PitchShift, Delay

# FastAPI & Gradio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
import gradio as gr

# MCP (Model Context Protocol)
from mcp.server.fastmcp import FastMCP

# ==========================================
# 0. Core Configuration & Directory Setup
# ==========================================
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = os.path.join(TEMP_DIR, "tts_uploads")
OUTPUT_DIR = os.path.join(TEMP_DIR, "tts_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🌟 NEW: Speaker Models Directory (For Fine-tuned Weights)
SPEAKERS_DIR = os.path.join(os.path.dirname(__file__), "pretrained_models", "speakers")
os.makedirs(SPEAKERS_DIR, exist_ok=True)

def get_available_speakers():
    """Auto-discover subdirectories in the speakers folder to list custom voices."""
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
                if os.stat(filepath).st_mtime < now - 3600:
                    os.remove(filepath)
            except Exception:
                pass

def append_log(current_logs, new_message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {new_message}"
    if not current_logs: return log_entry
    logs_list = current_logs.split("\n")
    logs_list.append(log_entry)
    if len(logs_list) > 15: logs_list = logs_list[-15:]
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
# 2. Core AI Generation & Mastering
# ==========================================
async def _mock_tts_generation(engine: str, mode: str, text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0, custom_speaker: str = "Default Model"):
    """
    Mock function representing GPU inference.
    In reality, if custom_speaker != "Default Model", the code will load 
    the checkpoint from pretrained_models/speakers/{custom_speaker} 
    and swap the voice before generating.
    """
    await asyncio.sleep(2)
    sf.write(out_path, np.zeros((44100, 1)), 44100)

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
# 3. FastAPI Setup (n8n endpoints)
# ==========================================
app = FastAPI(title="TTS Unified API")
mcp = FastMCP("TTS_Studio_MCP")
app.mount("/sse", mcp.get_starlette_app())

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "TTS Unified App", "available_speakers": get_available_speakers()}

@app.post("/api/tts/generate")
async def api_generate_tts(
    engine: str = Form(...),
    text: str = Form(...),
    language: str = Form("en"),
    mode: str = Form("standard"),
    speaker_model: str = Form("Default Model"), # 🌟 NEW: Select fine-tuned model via API
    speed: float = Form(1.0),
    apply_humanize: bool = Form(False),
    reference_audio: UploadFile = File(None)
):
    cleanup_old_files()
    job_id = str(uuid.uuid4())
    ref_audio_path = None
    if reference_audio:
        ref_ext = reference_audio.filename.split('.')[-1] if '.' in reference_audio.filename else 'wav'
        ref_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref.{ref_ext}")
        with open(ref_audio_path, "wb") as f: f.write(await reference_audio.read())

    raw_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_raw.wav")
    
    try:
        await _mock_tts_generation(engine, mode, text, language, ref_audio_path, raw_output_path, speed, speaker_model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return FileResponse(path=raw_output_path, media_type="audio/wav", filename=f"{engine}_raw.wav")


# ==========================================
# 4. Gradio UI Setup
# ==========================================
def gradio_tts(tts_mode, engine, custom_speaker, language, speed, text, ref_audio, current_logs):
    cleanup_old_files()
    if not text.strip(): return None, None, "❌ เกิดข้อผิดพลาด", append_log(current_logs, "❌ ERROR: ข้อความว่างเปล่า")

    logs = append_log(current_logs, f"🚀 START: Model={custom_speaker}, Engine={engine}")
    if custom_speaker != "Default Model":
        logs = append_log(logs, f"🧠 INFO: Loading Fine-tuned weights for [{custom_speaker}]...")
        
    yield None, None, "⏳ กำลังประมวลผล...", logs
    
    time.sleep(2)
    output_audio = ref_audio if ref_audio else os.path.join(OUTPUT_DIR, f"dummy_{int(time.time())}.wav")
    if not ref_audio: sf.write(output_audio, np.zeros((44100, 1)), 44100) 

    logs = append_log(logs, "✅ SUCCESS: สร้างเสียงสำเร็จ")
    yield output_audio, output_audio, "✅ สำเร็จ", logs

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
    gr.Markdown("# 🎙️ Unified AI Voice Engine (Director's Edition)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. TTS Generation (Fine-Tune Mode)"):
                with gr.Row():
                    with gr.Column():
                        engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="Engine")
                        tts_mode = gr.Radio(choices=["Standard", "Zero-Shot (Voice Cloning)"], value="Standard", label="Mode")
                        
                        # 🌟 NEW: Dynamic Speaker Selection
                        gr.Markdown("### 🧠 เลือกเสียงโมเดล (Speaker Checkpoint)")
                        dynamic_speakers = get_available_speakers()
                        speaker_dropdown = gr.Dropdown(choices=dynamic_speakers, value=dynamic_speakers[0], label="เลือกไฟล์น้ำหนัก Fine-tune")
                        
                        lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)"], value="Thai (th)", label="Language")
                        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")

                        text_input = gr.Textbox(label="Text Prompt", lines=4)
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
            
            # 🌟 NEW: Refresh Models Button
            refresh_models_btn = gr.Button("🔄 Refresh Speaker Models", variant="secondary")
            
            clear_log_btn = gr.Button("🗑️ Clear")

    # --- Event Wiring ---
    
    def refresh_speakers():
        new_list = get_available_speakers()
        return gr.update(choices=new_list, value=new_list[0]), "🔄 Refreshed available models from disk."

    refresh_models_btn.click(
        fn=refresh_speakers,
        inputs=None,
        outputs=[speaker_dropdown, logs_display]
    )

    submit_btn.click(
        fn=gradio_tts,
        inputs=[tts_mode, engine_dropdown, speaker_dropdown, lang_dropdown, speed_slider, text_input, ref_audio_input, system_logs_state],
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