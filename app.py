import os
import time
import glob
import uuid
import asyncio
import tempfile
from datetime import datetime

import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter, Reverb

# FastAPI & Gradio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
import gradio as gr

# MCP (Model Context Protocol)
from mcp.server.fastmcp import FastMCP

# ==========================================
# 0. Core Configuration & Auto-Cleanup
# ==========================================
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = os.path.join(TEMP_DIR, "tts_uploads")
OUTPUT_DIR = os.path.join(TEMP_DIR, "tts_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cleanup_old_files():
    """Delete files older than 1 hour to prevent disk full (critical for VPS)"""
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
# 1. AI Logic & Studio Processing
# ==========================================
async def _mock_tts_generation(engine: str, mode: str, text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0):
    """
    Mock function representing GPU inference.
    Replace with actual OmniVoice or CosyVoice code.
    In reality, speed is passed to the TTS model during inference.
    """
    await asyncio.sleep(2) # Simulate GPU time
    # Create 1 second of silence as a dummy valid WAV file
    sf.write(out_path, np.zeros((44100, 1)), 44100)

def apply_studio_mastering(
    input_path: str, 
    output_path: str, 
    gate: bool=True, 
    bass: float=4.5, 
    treble: float=3.0, 
    comp: float=3.0,
    reverb_amount: float=0.0
):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.T 
        
    board = Pedalboard([
        NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250) if gate else None,
        HighpassFilter(cutoff_frequency_hz=80),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass), 
        HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble), 
        Compressor(threshold_db=-15, ratio=comp, attack_ms=2.0, release_ms=100),
        # Add subtle room reverb if requested (great for audiobook feel)
        Reverb(room_size=0.1, dry_level=1.0, wet_level=reverb_amount) if reverb_amount > 0 else None,
        Limiter(threshold_db=-1.0)
    ])
    
    board = Pedalboard([effect for effect in board if effect is not None])
    effected_audio = board(audio_data, sample_rate)
    
    if len(effected_audio.shape) > 1:
         effected_audio = effected_audio.T
         
    sf.write(output_path, effected_audio, sample_rate)
    return output_path

# ==========================================
# 2. FastAPI Setup (n8n endpoints)
# ==========================================
app = FastAPI(title="TTS Unified API", description="FastAPI + Gradio + MCP in a single application.")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "TTS Unified App"}

@app.post("/api/tts/generate")
async def api_generate_tts(
    engine: str = Form(...),
    text: str = Form(...),
    language: str = Form("en"),
    mode: str = Form("standard"),
    speed: float = Form(1.0), # Added missing speed parameter
    apply_studio_effect: bool = Form(False),
    reference_audio: UploadFile = File(None)
):
    """Endpoint specifically designed for n8n workflows"""
    cleanup_old_files()
    if engine not in ["omnivoice", "cosyvoice"]:
        raise HTTPException(status_code=400, detail="Engine must be 'omnivoice' or 'cosyvoice'")
    if mode == "zeroshot" and not reference_audio:
        raise HTTPException(status_code=400, detail="Zero-Shot mode requires 'reference_audio'")

    job_id = str(uuid.uuid4())
    ref_audio_path = None

    if reference_audio:
        ref_ext = reference_audio.filename.split('.')[-1] if '.' in reference_audio.filename else 'wav'
        ref_audio_path = os.path.join(UPLOAD_DIR, f"{job_id}_ref.{ref_ext}")
        with open(ref_audio_path, "wb") as f:
            f.write(await reference_audio.read())

    raw_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_raw.wav")
    final_output_path = os.path.join(OUTPUT_DIR, f"{job_id}_final.wav")

    try:
        await _mock_tts_generation(engine, mode, text, language, ref_audio_path, raw_output_path, speed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if apply_studio_effect:
        apply_studio_mastering(raw_output_path, final_output_path)
        return FileResponse(path=final_output_path, media_type="audio/wav", filename=f"{engine}_studio.wav")

    return FileResponse(path=raw_output_path, media_type="audio/wav", filename=f"{engine}_raw.wav")

# ==========================================
# 3. MCP Server Setup (For AI Agents)
# ==========================================
mcp = FastMCP("TTS_Studio_MCP")

@mcp.tool()
async def generate_podcast_tts(text: str, engine: str = "cosyvoice", language: str = "th", speed: float = 1.0) -> str:
    """
    Generate high-quality podcast-style TTS audio from text. 
    Use this when a user asks to speak or convert text to voice.
    Returns the absolute path to the generated WAV file.
    """
    cleanup_old_files()
    job_id = str(uuid.uuid4())
    raw_path = os.path.join(OUTPUT_DIR, f"{job_id}_mcp_raw.wav")
    final_path = os.path.join(OUTPUT_DIR, f"{job_id}_mcp_studio.wav")
    
    # 1. Generate Raw TTS
    await _mock_tts_generation(engine, "standard", text, language, None, raw_path, speed)
    # 2. Apply Studio Effects for Podcast vibe
    apply_studio_mastering(raw_path, final_path)
    
    return f"Audio generated successfully at: {final_path}"

mcp_app = mcp.get_starlette_app()
app.mount("/sse", mcp_app)


# ==========================================
# 4. Gradio UI Setup
# ==========================================
def gradio_tts(tts_mode, engine, language, speed, text, ref_audio, current_logs):
    cleanup_old_files()
    if not text.strip():
        logs = append_log(current_logs, "❌ ERROR: ข้อความว่างเปล่า")
        yield None, None, "❌ เกิดข้อผิดพลาด", logs
        return
    if tts_mode == "Zero-Shot (Voice Cloning)" and not ref_audio:
        logs = append_log(current_logs, "❌ ERROR: โหมดโคลนเสียงต้องมี Reference")
        yield None, None, "❌ เกิดข้อผิดพลาด", logs
        return

    logs = append_log(current_logs, f"🚀 START: Mode={tts_mode}, Engine={engine}, Lang={language}, Speed={speed}x")
    yield None, None, "⏳ กำลังประมวลผล...", logs
    
    # Generate Dummy audio synchronously for Gradio
    time.sleep(2)
    output_audio_path = ref_audio 
    if not output_audio_path:
        dummy_path = os.path.join(OUTPUT_DIR, f"dummy_{int(time.time())}.wav")
        sf.write(dummy_path, np.zeros((44100, 1)), 44100) 
        output_audio_path = dummy_path

    logs = append_log(logs, "✅ SUCCESS: สร้างเสียงสำเร็จ")
    yield output_audio_path, output_audio_path, "✅ สำเร็จ", logs

def gradio_studio(input_audio, export_format, enable_gate, bass, treble, comp, reverb, current_logs):
    cleanup_old_files()
    if not input_audio:
        return None, "❌ ไม่พบไฟล์", append_log(current_logs, "❌ ERROR: No input file")
    try:
        logs = append_log(current_logs, f"⚙️ START STUDIO Mastering")
        ext = export_format.lower()
        output_file = os.path.join(OUTPUT_DIR, f"studio_{int(time.time())}.{ext}")
        
        apply_studio_mastering(input_audio, output_file, enable_gate, bass, treble, comp, reverb)
        
        logs = append_log(logs, f"✅ SUCCESS: Exported as {ext.upper()}")
        return output_file, "✅ สำเร็จ", logs
    except Exception as e:
        return None, str(e), append_log(current_logs, f"❌ ERROR: {str(e)}")

# Build Gradio Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ Unified AI Voice Engine (Gradio + FastAPI + MCP)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. TTS Generation"):
                with gr.Row():
                    with gr.Column():
                        tts_mode = gr.Radio(choices=["Standard", "Zero-Shot (Voice Cloning)"], value="Zero-Shot (Voice Cloning)", label="Mode")
                        engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="Engine")
                        lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="Language")
                        
                        # Added missing feature: Speech Speed Control
                        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="ความเร็วในการพูด (Speech Speed)")
                        
                        text_input = gr.Textbox(label="Text Prompt", lines=4)
                        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
                        submit_btn = gr.Button("🚀 Generate Speech", variant="primary")
                    with gr.Column():
                        status_output = gr.Markdown("🟢 พร้อมใช้งาน")
                        output_audio = gr.Audio(label="Raw Audio", interactive=False)

            with gr.Tab("🎛️ 2. Studio Processing"):
                with gr.Row():
                    with gr.Column():
                        raw_audio_input = gr.Audio(label="Input Audio", type="filepath")
                        bass_boost = gr.Slider(minimum=0, maximum=12, value=4.5, label="Bass (Proximity)")
                        treble_boost = gr.Slider(minimum=0, maximum=12, value=3.0, label="Treble (Air)")
                        comp_ratio = gr.Slider(minimum=1, maximum=8, value=3.0, label="Compression")
                        
                        # Added missing feature: Subtle Room Reverb for audiobooks
                        reverb_amount = gr.Slider(minimum=0.0, maximum=0.5, value=0.0, step=0.05, label="Room Reverb (Wet Level)", info="เพิ่มมิติให้เสียงก้องเหมือนอยู่ในห้องเล็กๆ (เหมาะกับ Audiobook)")
                        
                        enable_gate = gr.Checkbox(value=True, label="Noise Gate")
                        export_format = gr.Radio(choices=["WAV", "FLAC"], value="WAV", label="Format")
                        process_btn = gr.Button("🎧 Process Studio Audio", variant="primary")
                    with gr.Column():
                        studio_status = gr.Markdown("🟢 รอรับไฟล์")
                        studio_audio_output = gr.Audio(label="Mastered Audio", interactive=False)

        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=20, interactive=False, value="[System] Initialized.")
            clear_log_btn = gr.Button("🗑️ Clear")

    submit_btn.click(
        fn=gradio_tts,
        inputs=[tts_mode, engine_dropdown, lang_dropdown, speed_slider, text_input, ref_audio_input, system_logs_state],
        outputs=[output_audio, raw_audio_input, status_output, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    process_btn.click(
        fn=gradio_studio,
        inputs=[raw_audio_input, export_format, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, system_logs_state],
        outputs=[studio_audio_output, studio_status, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    
    clear_log_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[system_logs_state, logs_display])


# ==========================================
# 5. Application Mount
# ==========================================
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)