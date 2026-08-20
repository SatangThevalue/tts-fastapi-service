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
# 1. 10 Advanced Studio Presets Configuration
# ==========================================
# โครงสร้าง: bass, treble, comp, reverb, gate, drive, pitch, delay
STUDIO_PRESETS = {
    # --- กลุ่มงาน Professional ---
    "🎙️ Podcast Studio (เสียงแน่น นุ่มลึก)": {
        "bass": 5.0, "treble": 3.5, "comp": 3.5, "reverb": 0.05, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0,
        "desc": "เพิ่มความทุ้มและบีบอัดเสียงให้ฟังสบาย เหมาะสำหรับรายการพอดแคสต์"
    },
    "📖 Audiobook Pro (บรรยายชัดเจน มีมิติ)": {
        "bass": 2.0, "treble": 2.0, "comp": 2.5, "reverb": 0.15, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0,
        "desc": "เสียงใสสะอาด มีมิติเสียงก้องนิดๆ ให้ความรู้สึกเหมือนกำลังเล่านิทานในห้อง"
    },
    "🗣️ Natural Human (เสียงคนพูดคุยปกติ)": {
        "bass": 1.0, "treble": 1.5, "comp": 1.5, "reverb": 0.08, "gate": False, "drive": 0.0, "pitch": 0, "delay": 0.0,
        "desc": "ปรับแต่งน้อยที่สุด แค่เพิ่มความใสเล็กน้อย ไม่ตัดเสียงลมหายใจทิ้ง เพื่อความเป็นธรรมชาติ"
    },
    "📣 Public Address (ประกาศผ่านเสียงตามสาย/สนามบิน)": {
         "bass": -3.0, "treble": 5.0, "comp": 4.0, "reverb": 0.4, "gate": False, "drive": 5.0, "pitch": 0, "delay": 0.15,
         "desc": "จำลองเสียงประกาศก้องๆ ในพื้นที่กว้าง มีเสียงสะท้อน (Delay/Reverb) และความแตกพร่าเล็กน้อย"
    },
    
    # --- กลุ่มงาน Character & Creative ---
    "📻 Vintage Radio (วิทยุ FM ยุค 90)": {
        "bass": 7.0, "treble": 5.0, "comp": 6.0, "reverb": 0.0, "gate": True, "drive": 10.0, "pitch": 0, "delay": 0.0,
        "desc": "อัดเบสหนัก บีบอัดเสียงแน่นสุดๆ และใส่ความ Saturation (Drive) ให้เสียงดูวินเทจสไตล์ดีเจยุคเก่า"
    },
    "📞 Old Telephone (โทรศัพท์บ้านแบบเก่า)": {
        "bass": -15.0, "treble": -8.0, "comp": 5.0, "reverb": 0.0, "gate": True, "drive": 25.0, "pitch": 0, "delay": 0.0,
        "desc": "ตัดย่านเบสและแหลมทิ้งแบบสุดโต่ง เพิ่มความแตก (Distortion) จำลองเสียงอู้อี้ผ่านสายโทรศัพท์"
    },
    "👺 Anonymous (เสียงพรางตัว/ผู้ไม่ประสงค์ออกนาม)": {
        "bass": 2.0, "treble": -2.0, "comp": 3.0, "reverb": 0.1, "gate": True, "drive": 5.0, "pitch": -4, "delay": 0.0,
        "desc": "กดคีย์เสียงให้ต่ำลง (Pitch Shift -4) ฟังดูทุ้ม ลึกลับ เหมือนในข่าวอาชญากรรม"
    },
    "👾 Cyberpunk AI (เสียงหุ่นยนต์/AI โลกอนาคต)": {
        "bass": -2.0, "treble": 8.0, "comp": 4.0, "reverb": 0.2, "gate": True, "drive": 15.0, "pitch": 2, "delay": 0.05,
        "desc": "เสียงแหลมใส บีบอัดให้กระด้าง แอบมีความแตกพร่าดิจิทัล (Drive) และเสียงก้องสะท้อนสั้นๆ"
    },
    
    # --- กลุ่มงาน Special Environments ---
    "🛁 Bathroom Echo (เสียงร้องเพลงในห้องน้ำ)": {
        "bass": 3.0, "treble": 4.0, "comp": 2.0, "reverb": 0.8, "gate": False, "drive": 0.0, "pitch": 0, "delay": 0.0,
        "desc": "เพิ่ม Reverb แบบจัดเต็ม (Wet Level 80%) จำลองเสียงสะท้อนกังวานในห้องน้ำกระเบื้อง"
    },
    "🏟️ Stadium Concert (เสียงกึกก้องในสนามกีฬา)": {
        "bass": 4.0, "treble": 2.0, "comp": 3.0, "reverb": 0.7, "gate": False, "drive": 0.0, "pitch": 0, "delay": 0.3,
        "desc": "เพิ่มเสียงสะท้อนแบบหน่วงเวลา (Delay 300ms) และ Reverb กว้างๆ เหมาะกับฉากบรรยายสเกลใหญ่"
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
    drive_amount: float=0.0,   # ความแตกพร่า (Saturation/Distortion)
    pitch_shift: int=0,        # การเปลี่ยนคีย์เสียง (Semitones)
    delay_time: float=0.0,     # เสียงสะท้อน (Echo/Delay)
    humanize: bool=False
):
    audio_data, sample_rate = sf.read(input_path)
    if len(audio_data.shape) > 1:
        audio_data = audio_data.T 
        
    board = Pedalboard([
        # 1. Pitch Shifting (ทำเป็นอันดับแรก เพื่อไม่ให้กระทบ Reverb/Delay)
        PitchShift(semitones=pitch_shift) if pitch_shift != 0 else None,
        
        # 2. Noise Gate & EQ
        NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250) if gate else None,
        HighpassFilter(cutoff_frequency_hz=300 if bass <= -10 else 80),
        LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass), 
        HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble), 
        
        # 3. Saturation / Distortion (Tape warmth or Telephone crackle)
        Distortion(drive_db=drive_amount) if drive_amount > 0 else None,
        
        # 4. Compression (จับให้เสียงนิ่ง)
        Compressor(threshold_db=-15, ratio=comp, attack_ms=2.0, release_ms=100),
        
        # 5. Spatial Effects (Delay & Reverb)
        Delay(delay_seconds=delay_time, feedback=0.3, mix=0.4) if delay_time > 0 else None,
        Reverb(room_size=0.3 if delay_time > 0 else 0.1, dry_level=1.0, wet_level=reverb_amount) if reverb_amount > 0 else None,
        
        # 6. Humanize Effect (Micro-modulation)
        Chorus(rate_hz=0.5, depth=0.05, mix=0.1) if humanize else None,
        
        # 7. Safety Limiter
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
    preset: str = Form(None), 
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
        apply_studio_mastering(
            input_path=raw_output_path, 
            output_path=final_output_path, 
            gate=p['gate'], bass=p['bass'], treble=p['treble'], 
            comp=p['comp'], reverb_amount=p['reverb'], 
            drive_amount=p['drive'], pitch_shift=p['pitch'], delay_time=p['delay'],
            humanize=apply_humanize
        )
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
    if preset_name in STUDIO_PRESETS:
        p = STUDIO_PRESETS[preset_name]
        return (
            gr.update(value=p['desc']), 
            gr.update(value=p['gate']), 
            gr.update(value=p['bass']), 
            gr.update(value=p['treble']), 
            gr.update(value=p['comp']), 
            gr.update(value=p['reverb']),
            gr.update(value=p['drive']),
            gr.update(value=p['pitch']),
            gr.update(value=p['delay'])
        )
    return [gr.update()] * 9

def gradio_studio(input_audio, preset, humanize, export_format, enable_gate, bass, treble, comp, reverb, drive, pitch, delay, current_logs):
    cleanup_old_files()
    if not input_audio:
        return None, "❌ ไม่พบไฟล์", append_log(current_logs, "❌ ERROR: No input file")
    try:
        logs = append_log(current_logs, f"⚙️ START STUDIO: {preset}")
        ext = export_format.lower()
        output_file = os.path.join(OUTPUT_DIR, f"studio_{int(time.time())}.{ext}")
        
        apply_studio_mastering(
            input_audio=input_audio, 
            output_path=output_file, 
            gate=enable_gate, bass=bass, treble=treble, 
            comp=comp, reverb_amount=reverb, 
            drive_amount=drive, pitch_shift=pitch, delay_time=delay,
            humanize=humanize
        )
        
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
                        
                        gr.Markdown("### 🎛️ เลือกสไตล์เสียง (10 Pro Presets)")
                        preset_dropdown = gr.Dropdown(
                            choices=list(STUDIO_PRESETS.keys()), 
                            value=list(STUDIO_PRESETS.keys())[0], 
                            label="พรีเซ็ตสตูดิโอ (Studio Preset)"
                        )
                        preset_desc = gr.Markdown(f"*{STUDIO_PRESETS[list(STUDIO_PRESETS.keys())[0]]['desc']}*")
                        
                        humanize_checkbox = gr.Checkbox(
                            value=False, 
                            label="🤖 ➔ 🧑 Humanize (ลบความเพอร์เฟคของ AI)",
                            info="เพิ่มคลื่นเสียงแทรกซ้อน (Micro-Modulation) ให้เนื้อเสียงไม่นิ่งเป๊ะจนดูเป็นหุ่นยนต์"
                        )
                        
                        with gr.Accordion("⚙️ ปรับแต่งแบบละเอียด (Advanced Manual Controls)", open=False):
                            gr.Markdown("**Tone & Dynamics**")
                            bass_boost = gr.Slider(minimum=-15, maximum=15, value=5.0, label="Bass (Proximity)")
                            treble_boost = gr.Slider(minimum=-15, maximum=15, value=3.5, label="Treble (Air)")
                            comp_ratio = gr.Slider(minimum=1, maximum=10, value=3.5, label="Compression")
                            enable_gate = gr.Checkbox(value=True, label="Noise Gate")
                            
                            gr.Markdown("**Special FX (Space & Modulation)**")
                            reverb_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.01, label="Room Reverb (เสียงก้อง)")
                            delay_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Delay/Echo Time (เสียงสะท้อนหน่วงเวลา)")
                            drive_amount = gr.Slider(minimum=0.0, maximum=30.0, value=0.0, step=1.0, label="Distortion/Drive (ความแตกพร่าวินเทจ)")
                            pitch_shift = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Pitch Shift (เปลี่ยนคีย์เสียง / Semitones)")

                        export_format = gr.Radio(choices=["WAV", "FLAC"], value="WAV", label="Format")
                        process_btn = gr.Button("🎧 ประมวลผลและทดสอบฟัง (Process & Listen)", variant="primary")
                        
                    with gr.Column():
                        studio_status = gr.Markdown("🟢 รอรับไฟล์")
                        studio_audio_output = gr.Audio(label="Mastered Audio", interactive=False)

        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=20, interactive=False, value="[System] Initialized.")
            clear_log_btn = gr.Button("🗑️ Clear")

    preset_dropdown.change(
        fn=update_sliders_from_preset,
        inputs=[preset_dropdown],
        outputs=[preset_desc, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount]
    )

    submit_btn.click(
        fn=gradio_tts,
        inputs=[tts_mode, engine_dropdown, lang_dropdown, speed_slider, text_input, ref_audio_input, system_logs_state],
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