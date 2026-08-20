import gradio as gr
import time
import os
import soundfile as sf
import tempfile
import glob
from datetime import datetime
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter

# ==========================================
# 0. ระบบจัดการไฟล์ (Auto-Cleanup)
# ==========================================
def cleanup_old_files():
    """ลบไฟล์ใน temp ที่เก่ากว่า 1 ชั่วโมง ป้องกัน Disk เต็ม"""
    temp_dir = tempfile.gettempdir()
    now = time.time()
    for f in glob.glob(os.path.join(temp_dir, "studio_processed_*.wav")) + glob.glob(os.path.join(temp_dir, "studio_processed_*.flac")):
        if os.stat(f).st_mtime < now - 3600:
            try:
                os.remove(f)
            except:
                pass

# ==========================================
# ระบบ Logging
# ==========================================
def append_log(current_logs, new_message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {new_message}"
    if not current_logs:
         return log_entry
    logs_list = current_logs.split("\n")
    logs_list.append(log_entry)
    if len(logs_list) > 15:
        logs_list = logs_list[-15:]
    return "\n".join(logs_list)

# ==========================================
# 1. ฟังก์ชันส่วน TTS Generation
# ==========================================
def generate_tts(tts_mode, engine, language, text, ref_audio, current_logs):
    cleanup_old_files() # รันทำความสะอาดไฟล์ขยะทุกครั้งที่มีการกด Generate
    
    if not text.strip():
        logs = append_log(current_logs, f"❌ ERROR [TTS]: ข้อความว่างเปล่า")
        yield None, None, "❌ **เกิดข้อผิดพลาด:** กรุณาป้อนข้อความ", logs
        return
        
    if tts_mode == "Zero-Shot (Voice Cloning)" and not ref_audio:
        logs = append_log(current_logs, f"❌ ERROR [TTS]: เลือกโหมดโคลนเสียง แต่ไม่พบไฟล์ Reference")
        yield None, None, "❌ **เกิดข้อผิดพลาด:** โหมด Zero-Shot ต้องอัปโหลดเสียงต้นแบบ", logs
        return

    logs = append_log(current_logs, f"🚀 START [TTS]: Mode={tts_mode}, Engine={engine}, Lang={language}")
    yield None, None, f"⏳ **กำลังประมวลผล:** เรียกใช้งาน {engine}...", logs
    
    time.sleep(1) 
    
    if tts_mode == "Zero-Shot (Voice Cloning)":
        logs = append_log(logs, f"⏳ INFO [TTS]: กำลังดึงลักษณะเสียงจาก Reference Audio...")
    else:
        logs = append_log(logs, f"⏳ INFO [TTS]: กำลังโหลดเสียง Default ของภาษา {language}...")
        
    yield None, None, f"⏳ **กำลังประมวลผล:** สร้างเนื้อเสียง...", logs
    time.sleep(1)
    
    # [จุดที่ต้องแก้ไข] - โค้ดของจริงให้เรียกโมเดลมา Infer ตรงนี้
    # สำหรับ Mock: ถ้ามี ref_audio คืนค่า ref_audio, ถ้าไม่มี (Standard Mode) คืนค่า Dummy
    output_audio_path = ref_audio 
    if not output_audio_path:
        # สร้างไฟล์ Dummy เมื่อไม่มี Ref
        dummy_path = os.path.join(tempfile.gettempdir(), f"dummy_{int(time.time())}.wav")
        import numpy as np
        sf.write(dummy_path, np.zeros((44100, 1)), 44100) # 1 sec silence
        output_audio_path = dummy_path

    logs = append_log(logs, f"✅ SUCCESS [TTS]: สร้างเสียงสำเร็จ")
    yield output_audio_path, output_audio_path, f"✅ **สำเร็จ:** พร้อมนำไปปรับแต่งในแท็บ Studio!", logs

# ==========================================
# 2. ฟังก์ชันส่วน Studio Post-Processing
# ==========================================
def process_studio_audio(input_audio, export_format, enable_gate, bass_boost, treble_boost, compression_ratio, current_logs):
    cleanup_old_files()
    
    if not input_audio:
        logs = append_log(current_logs, f"❌ ERROR [STUDIO]: ไม่พบไฟล์เสียง Input")
        return None, "❌ กรุณาอัปโหลดเสียง", logs

    try:
        logs = append_log(current_logs, f"⚙️ START [STUDIO]: Mastering (Bass={bass_boost}dB, Comp={compression_ratio}:1, Format={export_format})")
        
        audio_data, sample_rate = sf.read(input_audio)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.T 
            
        board = Pedalboard([
            NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250) if enable_gate else None,
            HighpassFilter(cutoff_frequency_hz=80),
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass_boost), 
            HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble_boost), 
            Compressor(threshold_db=-15, ratio=compression_ratio, attack_ms=2.0, release_ms=100),
            Limiter(threshold_db=-1.0)
        ])
        
        board = Pedalboard([effect for effect in board if effect is not None])
        effected_audio = board(audio_data, sample_rate)
        
        if len(effected_audio.shape) > 1:
             effected_audio = effected_audio.T

        ext = export_format.lower()
        output_file = os.path.join(tempfile.gettempdir(), f"studio_processed_{int(time.time())}.{ext}")
        sf.write(output_file, effected_audio, sample_rate)

        logs = append_log(logs, f"✅ SUCCESS [STUDIO]: บันทึกไฟล์เป็น {ext.upper()} สำเร็จ")
        return output_file, f"✅ **สำเร็จ:** ไฟล์พร้อมใช้งาน ({export_format})", logs

    except Exception as e:
         error_msg = f"❌ ERROR [STUDIO]: {str(e)}"
         logs = append_log(current_logs, error_msg)
         return None, error_msg, logs

# ==========================================
# 3. สร้าง UI หน้าเว็บด้วย Gradio Blocks
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🎙️ AI Voice & Studio Processor (Pro Edition)")
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): 
            with gr.Tab("🎙️ 1. สร้างเสียง (TTS Generation)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 1. ตั้งค่าเสียง")
                        # 🌟 เพิ่มระบบเลือกโหมด
                        tts_mode = gr.Radio(
                            choices=["Standard (Default Voice)", "Zero-Shot (Voice Cloning)"], 
                            value="Zero-Shot (Voice Cloning)", 
                            label="โหมดการทำงาน (TTS Mode)"
                        )
                        with gr.Row():
                            engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="TTS Engine")
                            lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="ภาษาเป้าหมาย")
                        
                        text_input = gr.Textbox(label="ข้อความ (Text Prompt)", lines=4)
                        ref_audio_input = gr.Audio(label="เสียงต้นแบบ (Reference Audio) - จำเป็นเฉพาะโหมด Zero-Shot", type="filepath", sources=["upload", "microphone"])
                        
                        submit_btn = gr.Button("🚀 สร้างเสียงพูด (Generate Speech)", variant="primary", size="lg")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎧 2. ผลลัพธ์ (Raw Output)")
                        status_output = gr.Markdown("🟢 **สถานะ:** พร้อมใช้งาน (รอรับคำสั่ง)")
                        output_audio = gr.Audio(label="เสียงที่ถูกสร้างขึ้น (Raw Generated Audio)", interactive=False)

            with gr.Tab("🎛️ 2. ปรับแต่งเสียงสตูดิโอ (Studio Post-Processing)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎚️ เครื่องมือปรับแต่ง (Studio Effects)")
                        raw_audio_input = gr.Audio(label="เสียงตั้งต้น (Input Audio)", type="filepath")
                        
                        bass_boost = gr.Slider(minimum=0, maximum=12, value=4.5, step=0.5, label="เพิ่มความทุ้ม (Bass/Proximity Effect dB)")
                        treble_boost = gr.Slider(minimum=0, maximum=12, value=3.0, step=0.5, label="เพิ่มความคมชัด (Treble/Air dB)")
                        compression_ratio = gr.Slider(minimum=1, maximum=8, value=3.0, step=0.5, label="ระดับความแน่น (Compression Ratio)")
                        enable_gate = gr.Checkbox(value=True, label="เปิดระบบตัดเสียงรบกวนรอบข้าง (Noise Gate)")
                        
                        # 🌟 เพิ่มการเลือก Format
                        export_format = gr.Radio(choices=["WAV", "FLAC"], value="WAV", label="รูปแบบไฟล์ขาออก (Export Format)")
                        process_btn = gr.Button("🎧 ประมวลผลเสียงระดับสตูดิโอ (Process)", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🌟 ผลลัพธ์สุดท้าย (Mastered Output)")
                        studio_status = gr.Markdown("🟢 **สถานะ:** รอการปรับแต่ง")
                        studio_audio_output = gr.Audio(label="เสียงที่ผ่านการปรับแต่งแล้ว (Studio Processed)", interactive=False)

        # ------------------- ด้านขวา: กรอบแสดง System Logs -------------------
        with gr.Column(scale=1): 
            gr.Markdown("### 💻 System Activity Logs")
            logs_display = gr.Textbox(label="Terminal Output", lines=20, interactive=False, value="[System] Initialized.")
            clear_log_btn = gr.Button("🗑️ Clear Logs", size="sm")

    # ------------------- Event Listeners -------------------
    submit_btn.click(
        fn=generate_tts,
        inputs=[tts_mode, engine_dropdown, lang_dropdown, text_input, ref_audio_input, system_logs_state],
        outputs=[output_audio, raw_audio_input, status_output, system_logs_state]
    ).then(
        fn=lambda log_text: log_text,
        inputs=[system_logs_state],
        outputs=[logs_display]
    )

    process_btn.click(
        fn=process_studio_audio,
        inputs=[raw_audio_input, export_format, enable_gate, bass_boost, treble_boost, compression_ratio, system_logs_state],
        outputs=[studio_audio_output, studio_status, system_logs_state]
    ).then(
        fn=lambda log_text: log_text,
        inputs=[system_logs_state],
        outputs=[logs_display]
    )
    
    clear_log_btn.click(
        fn=lambda: ("", ""), 
        inputs=None, 
        outputs=[system_logs_state, logs_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)