import os
import time
import glob
import uuid
import asyncio
import tempfile
import json
from datetime import datetime
import httpx  # For async LLM API calls

import soundfile as sf
import numpy as np
from pedalboard import (Pedalboard, Compressor, HighpassFilter, 
                        LowShelfFilter, HighShelfFilter, NoiseGate, 
                        Limiter, Reverb, Chorus, Distortion, PitchShift, Delay)

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
import gradio as gr
from mcp.server.fastmcp import FastMCP

# ==========================================
# 0. Configuration & Environment Variables
# ==========================================
TEMP_DIR = tempfile.gettempdir()
UPLOAD_DIR = os.path.join(TEMP_DIR, "tts_uploads")
OUTPUT_DIR = os.path.join(TEMP_DIR, "tts_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LLM Auto-Tagging Configuration
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# ==========================================
# 1. Cleanup & Logging
# ==========================================
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
# 2. Studio Presets Configuration
# ==========================================
STUDIO_PRESETS = {
    "🎙️ Podcast Studio": {"bass": 5.0, "treble": 3.5, "comp": 3.5, "reverb": 0.05, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "นุ่มลึก มีน้ำหนัก ฟังสบาย"},
    "📖 Audiobook Pro": {"bass": 2.0, "treble": 2.0, "comp": 2.5, "reverb": 0.15, "gate": True, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "ใสสะอาด มีมิติเสียงก้องนิดๆ"},
    "🗣️ Natural Human": {"bass": 1.0, "treble": 1.5, "comp": 1.5, "reverb": 0.08, "gate": False, "drive": 0.0, "pitch": 0, "delay": 0.0, "desc": "ธรรมชาติ ไม่บีบอัดมาก"},
    "📻 Vintage Radio": {"bass": 7.0, "treble": 5.0, "comp": 6.0, "reverb": 0.0, "gate": True, "drive": 10.0, "pitch": 0, "delay": 0.0, "desc": "เบสหนัก ความบีบอัดแน่น และ Drive เล็กน้อยสไตล์ยุค 90"},
    "📞 Old Telephone": {"bass": -15.0, "treble": -8.0, "comp": 5.0, "reverb": 0.0, "gate": True, "drive": 25.0, "pitch": 0, "delay": 0.0, "desc": "เสียงอู้อี้เหมือนคุยโทรศัพท์"},
    "👺 Anonymous (Pitch Down)": {"bass": 2.0, "treble": -2.0, "comp": 3.0, "reverb": 0.1, "gate": True, "drive": 5.0, "pitch": -4, "delay": 0.0, "desc": "เสียงคีย์ต่ำ พรางตัว ลึกลับ"}
}

# ==========================================
# 3. LLM Auto-Emotion Tagger (OpenAI Compatible)
# ==========================================
async def auto_tag_emotion_llm(text: str, engine: str, emotion: str) -> str:
    """Use an LLM to smartly insert emotion tags or formatting based on the chosen engine."""
    if not LLM_API_KEY:
        raise ValueError("Please provide an LLM API Key in settings before using Auto-Tagging.")
    
    system_prompt = ""
    if engine == "OmniVoice":
        system_prompt = (
            "You are an AI scriptwriter. Modify the user's text to fit the requested emotion by inserting OmniVoice non-verbal tags where appropriate. "
            "Supported tags ONLY: [laughter], [sigh], [surprise-ah], [surprise-oh], [surprise-wa], [dissatisfaction-hnn]. "
            "Insert them naturally in the text (e.g., at the beginning, pauses, or end). Do not change the original wording, only insert tags."
        )
    elif engine == "CosyVoice 3.0":
        system_prompt = (
            "You are an AI scriptwriter. Modify the user's text to express the requested emotion by adding a natural emotion tag at the beginning. "
            "Format exactly like this: <|emotion|> Original Text. "
            "Replace <|emotion|> with the requested emotion (e.g., <|happy|>, <|sad|>, <|angry|>, <|surprised|>, <|scared|>). "
            "Do not alter the user's original text, just prepend the emotion tag."
        )

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Emotion requested: {emotion}\nText: {text}"}
        ],
        "temperature": 0.3
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(f"{LLM_BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

def gradio_auto_tag(text, engine, emotion, current_logs, api_key, api_url, api_model):
    """Gradio handler for LLM Tagging"""
    if not text.strip():
        return text, append_log(current_logs, "❌ ERROR: No text provided for auto-tagging.")
    
    # Temporarily set environment variables from UI inputs
    global LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
    LLM_API_KEY = api_key
    LLM_BASE_URL = api_url
    LLM_MODEL = api_model

    logs = append_log(current_logs, f"🤖 LLM Request: Analyzing emotion '{emotion}' via {api_model}...")
    
    try:
        # Run async function synchronously for Gradio
        tagged_text = asyncio.run(auto_tag_emotion_llm(text, engine, emotion))
        logs = append_log(logs, f"✅ LLM Success: Auto-tagged text generated.")
        return tagged_text, logs
    except Exception as e:
        logs = append_log(logs, f"❌ LLM Error: {str(e)}")
        return text, logs

# ==========================================
# 4. Manual Tag Insertion Helpers
# ==========================================
def insert_tag_at_cursor(current_text, tag):
    """Simple helper to append a tag to the end of the text. 
    (In a real web app, Javascript is needed for exact cursor position, so we append for now)."""
    return f"{current_text.rstrip()} {tag} "

# ==========================================
# 5. Core AI Generation & Mastering
# ==========================================
async def _mock_tts_generation(engine: str, mode: str, text: str, lang: str, ref_path: str, out_path: str, speed: float = 1.0, instruct_prompt: str = ""):
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
# 6. Gradio Handlers
# ==========================================
def gradio_tts(tts_mode, engine, language, speed, text, instruct_prompt, ref_audio, current_logs):
    cleanup_old_files()
    if not text.strip(): return None, None, "❌ เกิดข้อผิดพลาด", append_log(current_logs, "❌ ERROR: ข้อความว่างเปล่า")

    logs = append_log(current_logs, f"🚀 START [TTS]: Engine={engine}, Instruct='{instruct_prompt[:15]}...'")
    yield None, None, "⏳ กำลังประมวลผล...", logs
    
    time.sleep(2)
    output_audio = ref_audio if ref_audio else os.path.join(OUTPUT_DIR, f"dummy_{int(time.time())}.wav")
    if not ref_audio: sf.write(output_audio, np.zeros((44100, 1)), 44100) 

    logs = append_log(logs, "✅ SUCCESS: สร้างเสียงสำเร็จ")
    yield output_audio, output_audio, "✅ สำเร็จ", logs

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

# ==========================================
# 7. FastAPI & MCP Mount Setup
# ==========================================
app = FastAPI(title="TTS Unified API")
mcp = FastMCP("TTS_Studio_MCP")
app.mount("/sse", mcp.get_starlette_app())


# ==========================================
# 8. Gradio UI Assembly
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ Unified AI Voice Engine (Director's Edition)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. TTS Generation & Emotion"):
                with gr.Row():
                    # --- Left Column: Settings ---
                    with gr.Column():
                        engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="Engine")
                        tts_mode = gr.Radio(choices=["Standard", "Zero-Shot (Voice Cloning)", "Instruct (Emotion)"], value="Instruct (Emotion)", label="Mode")
                        
                        with gr.Row():
                            lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="Language")
                            speed_slider = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")

                        # --- NEW: LLM Auto-Emotion Agent ---
                        with gr.Accordion("🤖 LLM Auto-Emotion Agent", open=True):
                            gr.Markdown("พิมพ์ข้อความ แล้วให้ LLM ช่วยวิเคราะห์และใส่ Emotion Tag ให้อัตโนมัติ")
                            with gr.Row():
                                llm_emotion_intent = gr.Dropdown(
                                    choices=["ตื่นเต้น (Excited)", "เศร้า (Sad)", "โกรธ (Angry)", "หัวเราะ (Laughing)", "ตกใจ (Surprised)", "ทางการ (Formal)"],
                                    value="ตื่นเต้น (Excited)", label="อารมณ์ที่ต้องการ"
                                )
                                llm_auto_tag_btn = gr.Button("✨ ให้ AI ใส่ Tag อารมณ์ให้", variant="secondary")
                            
                            with gr.Accordion("⚙️ LLM API Settings (OpenAI Compatible)", open=False):
                                llm_api_url = gr.Textbox(label="API Base URL", value="https://api.openai.com/v1")
                                llm_api_key = gr.Textbox(label="API Key", type="password")
                                llm_api_model = gr.Textbox(label="Model", value="gpt-4o-mini")

                        text_input = gr.Textbox(label="Text Prompt (ใส่ Tag ในนี้ได้เลย)", lines=4)
                        
                        # --- NEW: Manual Action Tags (OmniVoice specific) ---
                        gr.Markdown("*ปุ่มลัดแทรก Action Tags (สำหรับ OmniVoice)*")
                        with gr.Row():
                            tag_laugh_btn = gr.Button("😂 [laughter]", size="sm")
                            tag_sigh_btn = gr.Button("😮‍💨 [sigh]", size="sm")
                            tag_surprise_btn = gr.Button("😲 [surprise-ah]", size="sm")
                            tag_angry_btn = gr.Button("😤 [dissatisfaction]", size="sm")

                        instruct_prompt = gr.Textbox(label="Instruction Prompt (สำหรับโหมด Instruct)", placeholder="e.g. female, excited, fast pacing")
                        ref_audio_input = gr.Audio(label="Reference Audio", type="filepath")
                        
                        submit_btn = gr.Button("🚀 Generate Speech", variant="primary")
                    
                    # --- Right Column: Output ---
                    with gr.Column():
                        status_output = gr.Markdown("🟢 พร้อมใช้งาน")
                        output_audio = gr.Audio(label="Raw Audio", interactive=False)

            with gr.Tab("🎛️ 2. Studio Presets & Effects"):
                with gr.Row():
                    with gr.Column():
                        raw_audio_input = gr.Audio(label="Input Audio", type="filepath")
                        gr.Markdown("### 🎛️ เลือกสไตล์เสียง")
                        preset_dropdown = gr.Dropdown(choices=list(STUDIO_PRESETS.keys()), value=list(STUDIO_PRESETS.keys())[0], label="Studio Preset")
                        preset_desc = gr.Markdown(f"*{STUDIO_PRESETS[list(STUDIO_PRESETS.keys())[0]]['desc']}*")
                        humanize_checkbox = gr.Checkbox(value=False, label="🤖 ➔ 🧑 Humanize (ลดความเป็นหุ่นยนต์)")
                        
                        with gr.Accordion("⚙️ ปรับแต่งแบบละเอียด (Manual EQ & FX)", open=False):
                            bass_boost = gr.Slider(minimum=-15, maximum=15, value=5.0, label="Bass (Proximity)")
                            treble_boost = gr.Slider(minimum=-15, maximum=15, value=3.5, label="Treble (Air)")
                            comp_ratio = gr.Slider(minimum=1, maximum=10, value=3.5, label="Compression")
                            enable_gate = gr.Checkbox(value=True, label="Noise Gate")
                            reverb_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.05, step=0.01, label="Room Reverb")
                            delay_amount = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.05, label="Delay/Echo")
                            drive_amount = gr.Slider(minimum=0.0, maximum=30.0, value=0.0, step=1.0, label="Distortion/Drive")
                            pitch_shift = gr.Slider(minimum=-12, maximum=12, value=0, step=1, label="Pitch Shift")

                        export_format = gr.Radio(choices=["WAV", "FLAC"], value="WAV", label="Format")
                        process_btn = gr.Button("🎧 ประมวลผลและทดสอบฟัง", variant="primary")
                        
                    with gr.Column():
                        studio_status = gr.Markdown("🟢 รอรับไฟล์")
                        studio_audio_output = gr.Audio(label="Mastered Audio", interactive=False)

        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Logs")
            logs_display = gr.Textbox(label="Live Console", lines=30, interactive=False, value="[System] Initialized.")
            clear_log_btn = gr.Button("🗑️ Clear")

    # --- Event Wiring ---
    
    # 1. Manual Tags Buttons
    tag_laugh_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[laughter]"), inputs=[text_input], outputs=[text_input])
    tag_sigh_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[sigh]"), inputs=[text_input], outputs=[text_input])
    tag_surprise_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[surprise-ah]"), inputs=[text_input], outputs=[text_input])
    tag_angry_btn.click(fn=lambda t: insert_tag_at_cursor(t, "[dissatisfaction-hnn]"), inputs=[text_input], outputs=[text_input])

    # 2. LLM Auto Tagging
    llm_auto_tag_btn.click(
        fn=gradio_auto_tag,
        inputs=[text_input, engine_dropdown, llm_emotion_intent, system_logs_state, llm_api_key, llm_api_url, llm_api_model],
        outputs=[text_input, system_logs_state]
    ).then(fn=lambda log: log, inputs=[system_logs_state], outputs=[logs_display])

    # 3. Presets
    preset_dropdown.change(
        fn=update_sliders_from_preset,
        inputs=[preset_dropdown],
        outputs=[preset_desc, enable_gate, bass_boost, treble_boost, comp_ratio, reverb_amount, drive_amount, pitch_shift, delay_amount]
    )

    # 4. Generate & Process
    submit_btn.click(
        fn=gradio_tts,
        inputs=[tts_mode, engine_dropdown, lang_dropdown, speed_slider, text_input, instruct_prompt, ref_audio_input, system_logs_state],
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