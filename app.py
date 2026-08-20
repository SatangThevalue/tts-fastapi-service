import os
import time
import glob
import uuid
import asyncio
import tempfile
from datetime import datetime

import soundfile as sf
import numpy as np
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter, Reverb, Chorus

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
    "🎙️ Podcast Studio (เสียงแน่น นุ่มลึก)": {
        "bass": 5.0, "treble": 3.5, "comp": 3.5, "reverb": 0.05, "gate": True,
        "desc": "เพิ่มความทุ้มและบีบอัดเสียงให้ฟังสบาย เหมาะสำหรับรายการพอดแคสต์"
    },
    "📖 Audiobook (บรรยายชัดเจน มีมิติ)": {
        "bass": 2.0, "treble": 2.0, "comp": 2.5, "reverb": 0.15, "gate": True,
        "desc": "เสียงใสสะอาด มีมิติเสียงก้องนิดๆ ให้ความรู้สึกเหมือนกำลังเล่านิทานในห้อง"
    },
    "📻 FM Radio (เสียงดีเจ วิทยุยุค 90)": {
        "bass": 7.0, "treble": 5.0, "comp": 5.0, "reverb": 0.0, "gate": True,
        "desc": "อัดเบสหนักๆ บีบอัดเสียงแน่นสุดๆ สไตล์ดีเจจัดรายการวิทยุ"
    },
    "🗣️ Natural Human (เสียงคนพูดคุยปกติ)": {
        "bass": 1.0, "treble": 1.5, "comp": 1.5, "reverb": 0.08, "gate": False,
        "desc": "ปรับแต่งน้อยที่สุด แค่เพิ่มความใสเล็กน้อย ไม่ตัดเสียงลมหายใจทิ้ง เพื่อความเป็นธรรมชาติ"
    },
    "📞 Phone Call (จำลองเสียงโทรศัพท์)": {
        "bass": -10.0, "treble": -5.0, "comp": 4.0, "reverb": 0.0, "gate": True,
        "desc": "ตัดย่านเบสและแหลมทิ้ง จำลองเสียงอู้อี้ที่ผ่านสายโทรศัพท์"
    }
}


# ==========================================
# 2. AI Logic & Studio Processing
# ==========================================
async def _mock_tts_generation(engine: str, mode: str, text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0):
    await asyncio.sleep(2)
    sf.write(out_path, np.zeros((44100, 1)), 44100)

def apply_studio_mastering(
    input_path: str, 
    output_path: str, 
    gate: bool=True, 
    bass: float=4.5, 
    treble: float=3.0, 
    comp: float=3.0,
    reverb_amount: float=0.0,
    humanize: bool=False
):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.T 
        
    board = Pedalboard([
        NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250) if gate else None,
        
        # Phone call preset logic: if bass is extremely low, apply tight bandpass
        HighpassFilter(cutoff_frequency_hz=300 if bass <= -10 else 80),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass), 
        HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble), 
        
        Compressor(threshold_db=-15, ratio=comp, attack_ms=2.0, release_ms=100),
        Reverb(room_size=0.2, dry_level=1.0, wet_level=reverb_amount) if reverb_amount > 0 else None,
        
        # Humanize Effect: add extremely subtle chorus/modulation to break AI "robotic perfection"
        Chorus(rate_hz=0.5, depth=0.05, mix=0.1) if humanize else None,
        
        Limiter(threshold_db=-1.0)
    ])
    
    board = Pedalboard([effect for effect in board if effect is not None])
    effected_audio = board(audio_data, sample_rate)
    
    if len(effected_audio.shape) > 1:
         effected_audio = effected_audio.T
         
    sf.write(output_path, effected_audio, sample_rate)
    return output_path

# ==========================================
# 3. FastAPI Setup (n8n endpoints)
# ==========================================
app = FastAPI(title="TTS Unified API")

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "TTS Unified App"}

@app.post("/api/tts/generate")
async def api_generate_tts(
    engine: str = Form(...),
    text: str = Form(...),
    language: str = Form("en"),
    mode: str = Form("standard"),
    speed: float = Form(1.0),
    preset: str = Form(None), # Accepts preset name from STUDIO_PRESETS
    apply_humanize: bool = Form(False),
    reference_audio: UploadFile = File(None)
):
    cleanup_old_files()
    if engine not in ["omnivoice", "cosyvoice"]:
        raise HTTPException(status_code=400, detail="Invalid Engine")

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

    if preset and preset in STUDIO_PRESETS:
        p = STUDIO_PRESETS[preset]
        apply_studio_mastering(raw_output_path, final_output_path, p['gate'], p['bass'], p['treble'], p['comp'], p['reverb'], apply_humanize)
        return FileResponse(path=final_output_path, media_type="audio/wav", filename=f"{engine}_studio.wav")

    return FileResponse(path=raw_output_path, media_type="audio/wav", filename=f"{engine}_raw.wav")

# ==========================================
# 4. MCP Server Setup (For AI Agents)
# ==========================================
mcp = FastMCP("TTS_Studio_MCP")
mcp_app = mcp.get_starlette_app()
app.mount("/sse", mcp_app)

# ==========================================
# 5. Gradio UI Setup
# ==========================================
def gradio_tts(tts_mode, engine, language, speed, text, ref_audio, current_logs):
    cleanup_old_files()
    if not text.strip():
        return None, None, "❌ เกิดข้อผิดพลาด", append_log(current_logs, "❌ ERROR: ข้อความว่างเปล่า")

    logs = append_log(current_logs, f"🚀 START: Mode={tts_mode}, Engine={engine}, Lang={language}, Speed={speed}x")
    yield None, None, "⏳ กำลังประมวลผล...", logs
    
    time.sleep(2)
    output_audio = ref_audio if ref_audio else os.path.join(OUTPUT_DIR, f"dummy_{int(time.time())}.wav")
    if not ref_audio:
        sf.write(output_audio, np.zeros((44100, 1)), 44100) 

    logs = append_log(logs, "✅ SUCCESS: สร้างเสียงสำเร็จ")
    yield output_audio, output_audio, "✅ สำเร็จ", logs

def update_sliders_from_preset(preset_name):
    """Callback to update slider values when a user selects a Preset"""
    if preset_name in STUDIO_PRESETS:
        p = STUDIO_PRESETS[preset_name]
        return gr.update(value=p['desc']), gr.update(value=p['gate']), gr.update(value=p['bass']), gr.update(value=p['treble']), gr.update(value=p['comp']), gr.update(value=p['reverb'])
    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def gradio_studio(input_audio, preset, humanize, export_format, enable_gate, bass, treble, comp, reverb, current_logs):
    cleanup_old_files()
    if not input_audio:
        return None, "❌ ไม่พบไฟล์", append_log(current_logs, "❌ ERROR: No input file")
    try:
        logs = append_log(current_logs, f"⚙️ START STUDIO: {preset}")
        ext = export_format.lower()
        output_file = os.path.join(OUTPUT_DIR, f"studio_{int(time.time())}.{ext}")
        
        apply_studio_mastering(input_audio, output_file, enable_gate, bass, treble, comp, reverb, humanize)
        
        logs = append_log(logs, f"✅ SUCCESS: Exported as {ext.upper()}")
        return output_file, "✅ สำเร็จ", logs
    except Exception as e:
        return None, str(e), append_log(current_logs, f"❌ ERROR: {str(e)}")

# Build Gradio Layout
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ Unified AI Voice Engine (Pro Edition)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. TTS Generation"):
                with gr.Row():
                    with gr.Column():
                        tts_mode = gr.Radio(choices=["Standard", "Zero-Shot (Voice Cloning)"], value="Standard", label="Mode")
                        engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="Engine")
                        lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="Language")
                        speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="ความเร็ว (Speed)")
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
                        
                        # Added Feature: Presets Selection
                        gr.Markdown("### 🎛️ เลือกสไตล์เสียงด่วน (Presets)")
                        preset_dropdown = gr.Dropdown(
                            choices=list(STUDIO_PRESETS.keys()), 
                            value=list(STUDIO_PRESETS.keys())[0], 
                            label="Preset สตูดิโอ"
                        )
                        preset_desc = gr.Markdown(f"*{STUDIO_PRESETS[list(STUDIO_PRESETS.keys())[0]]['desc']}*")
                        
                        # Added Feature: Humanize AI Voice
                        humanize_checkbox = gr.Checkbox(
                            value=False, 
                            label="🤖 ➔ 🧑 Humanize (ลบความเพอร์เฟคของ AI)",
                            info="เพิ่มคลื่นเสียงแทรกซ้อนบางๆ (Micro-Modulation) ให้เนื้อเสียงไม่นิ่งเป๊ะจนดูเป็นหุ่นยนต์"
                        )
                        
                        with gr.Accordion("⚙️ ปรับแต่งแบบละเอียด (Manual EQ & Dynamics)", open=False):
                            bass_boost = gr.Slider(minimum=-12, maximum=12, value=5.0, label="Bass (Proximity)")
                            treble_boost = gr.Slider(minimum=-12, maximum=12, value=3.5, label="Treble (Air)")
                            comp_ratio = gr.Slider(minimum=1, maximum=8, value=3.5, label="Compression")
                            reverb_amount = gr.Slider(minimum=0.0, maximum=0.5, value=0.05, step=0.01, label="Room Reverb")
                            enable_gate = gr.Checkbox(value=True, label="Noise Gate")

                        export_format = gr.Radio(choices=["WAV", "FLAC"], value="WAV", label="Format")
                        process_btn = gr.Button("🎧 ประมวลผลและทดสอบฟัง (Process & Listen)", variant="primary")
                        
                    with gr.Column():
                        studio_status = gr.Markdown("🟢 รอรับไฟล์")
                        studio_audio_output = gr.Audio(label="Mastered Audio", interactive=False)

        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=20, interactive=False, value="[System] Initialized.")
            clear_log_btn = gr.Button("🗑️ Clear")

    # Connect Preset Dropdown to Update Sliders
    preset_dropdown.change(
        fn=update_sliders_from_preset,
        inputs=[preset_dropdown],
        outputs=[preset_desc, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount]
    )

    submit_btn.click(
        fn=gradio_tts,
        inputs=[tts_mode, engine_dropdown, lang_dropdown, speed_slider, text_input, ref_audio_input, system_logs_state],
        outputs=[output_audio, raw_audio_input, status_output, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    process_btn.click(
        fn=gradio_studio,
        inputs=[raw_audio_input, preset_dropdown, humanize_checkbox, export_format, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, system_logs_state],
        outputs=[studio_audio_output, studio_status, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])
    
    clear_log_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[system_logs_state, logs_display])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)