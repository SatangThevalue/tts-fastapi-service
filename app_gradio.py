import gradio as gr
import time
import os
import soundfile as sf
import tempfile
from datetime import datetime
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter

# ==========================================
# ระบบ Logging (เก็บสถานะเพื่อให้ Gradio แสดงผล)
# ==========================================
# เราจะสร้างฟังก์ชันสำหรับรับข้อความและแปลงเป็น format ที่มี Timestamp
def append_log(current_logs, new_message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {new_message}"
    
    # หากยังไม่มี log เลย
    if not current_logs:
         return log_entry
         
    # หากมี log แล้ว ให้ต่อท้าย (จำกัดแค่ 15 บรรทัดล่าสุดกันหน้าเว็บยาวเกินไป)
    logs_list = current_logs.split("\n")
    logs_list.append(log_entry)
    if len(logs_list) > 15:
        logs_list = logs_list[-15:]
        
    return "\n".join(logs_list)


# ==========================================
# 1. ฟังก์ชันส่วน TTS Generation
# ==========================================
def generate_tts(engine, language, text, ref_audio, current_logs):
    if not text.strip():
        logs = append_log(current_logs, f"❌ ERROR [TTS]: ข้อความว่างเปล่า ยกเลิกการประมวลผล")
        yield None, None, "❌ **เกิดข้อผิดพลาด:** กรุณาป้อนข้อความ", logs
        return
    if not ref_audio:
        logs = append_log(current_logs, f"❌ ERROR [TTS]: ไม่พบไฟล์เสียงต้นแบบ (Reference Audio)")
        yield None, None, "❌ **เกิดข้อผิดพลาด:** กรุณาอัปโหลดเสียงต้นแบบ", logs
        return

    logs = append_log(current_logs, f"🚀 START [TTS]: กำลังสั่งงาน {engine} (ภาษา: {language}) ข้อความ: '{text[:20]}...'")
    yield None, None, f"⏳ **กำลังประมวลผล:** ดึงเนื้อเสียงของคุณผ่าน {engine} (ภาษา: {language})", logs
    
    # จำลองระยะเวลาการรัน AI ของ GPU
    logs = append_log(logs, f"⏳ INFO [TTS]: โมเดลกำลัง Extract เสียงพูดต้นฉบับ...")
    yield None, None, f"⏳ **กำลังประมวลผล:** ดึงเนื้อเสียงของคุณผ่าน {engine} (ภาษา: {language})", logs
    time.sleep(1) 
    
    logs = append_log(logs, f"⏳ INFO [TTS]: กำลัง Generate เสียงใหม่...")
    yield None, None, f"⏳ **กำลังประมวลผล:** ดึงเนื้อเสียงของคุณผ่าน {engine} (ภาษา: {language})", logs
    time.sleep(1)
    
    # [จุดที่ต้องแก้ไข] - โค้ดของจริงให้เรียกโมเดลมา Infer ตรงนี้
    output_audio_path = ref_audio 

    logs = append_log(logs, f"✅ SUCCESS [TTS]: สร้างเสียงเสร็จสิ้น ส่งต่อข้อมูลไปยังแท็บ Studio")
    yield output_audio_path, output_audio_path, f"✅ **สำเร็จ:** ประมวลผลข้อความด้วย {engine} เสร็จสิ้น!", logs


# ==========================================
# 2. ฟังก์ชันส่วน Studio Post-Processing
# ==========================================
def process_studio_audio(input_audio, enable_gate, bass_boost, treble_boost, compression_ratio, current_logs):
    if not input_audio:
        logs = append_log(current_logs, f"❌ ERROR [STUDIO]: ไม่พบไฟล์เสียง Input")
        return None, "❌ กรุณาอัปโหลดเสียง หรือสร้างเสียงจากแท็บแรกก่อน", logs

    try:
        logs = append_log(current_logs, f"⚙️ START [STUDIO]: เริ่มประมวลผล EQ และ Compressor (Bass={bass_boost}dB, Treble={treble_boost}dB, Comp={compression_ratio}:1)")
        
        # อ่านไฟล์เสียง
        audio_data, sample_rate = sf.read(input_audio)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.T 
            
        logs = append_log(logs, f"⚙️ INFO [STUDIO]: โหลดไฟล์เสียงสำเร็จ (Sample Rate: {sample_rate}Hz)")
        
        # สร้าง Chain ของ Effect
        board = Pedalboard([
            NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250) if enable_gate else None,
            HighpassFilter(cutoff_frequency_hz=80),
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass_boost), 
            HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble_boost), 
            Compressor(threshold_db=-15, ratio=compression_ratio, attack_ms=2.0, release_ms=100),
            Limiter(threshold_db=-1.0)
        ])
        
        board = Pedalboard([effect for effect in board if effect is not None])

        logs = append_log(logs, f"⚙️ INFO [STUDIO]: กำลังเขียน Effect ทับลงใน Audio Pipeline...")
        effected_audio = board(audio_data, sample_rate)
        
        if len(effected_audio.shape) > 1:
             effected_audio = effected_audio.T

        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"studio_processed_{int(time.time())}.wav")
        sf.write(output_file, effected_audio, sample_rate)

        logs = append_log(logs, f"✅ SUCCESS [STUDIO]: Mastering เสียงเสร็จสมบูรณ์")
        return output_file, "✅ **สำเร็จ:** ประมวลผลเสียงแบบ Studio Podcast เรียบร้อยแล้ว!", logs

    except Exception as e:
         error_msg = f"❌ ERROR [STUDIO]: {str(e)}"
         logs = append_log(current_logs, error_msg)
         return None, error_msg, logs

# ==========================================
# 3. สร้าง UI หน้าเว็บด้วย Gradio Blocks
# ==========================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🎙️ AI Voice & Studio Processor
        ระบบสร้างเสียงพูด Zero-Shot แบบสมจริง และปรับแต่งเนื้อเสียงให้เป็น **Podcast Studio Quality**
        """
    )
    
    # ------------------- สร้างตัวแปรซ่อน (State) สำหรับเก็บประวัติ Logs -------------------
    # State คือตัวแปรที่ Gradio จะจำค่าไว้ให้แต่ละ User (session) ไม่ทับกัน
    system_logs_state = gr.State(value="")
    
    with gr.Row():
        with gr.Column(scale=3): # พื้นที่หลัก 75%
            
            # ------------------- TAB 1: TTS Generation -------------------
            with gr.Tab("🎙️ 1. สร้างเสียง (TTS Generation)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 1. ตั้งค่าเสียง")
                        with gr.Row():
                            engine_dropdown = gr.Radio(choices=["OmniVoice", "CosyVoice 3.0"], value="CosyVoice 3.0", label="TTS Engine")
                            lang_dropdown = gr.Dropdown(choices=["Thai (th)", "English (en)", "Chinese (zh)"], value="Thai (th)", label="ภาษาเป้าหมาย")
                        
                        text_input = gr.Textbox(label="ข้อความ (Text Prompt)", lines=4)
                        ref_audio_input = gr.Audio(label="เสียงต้นแบบ (Reference Audio)", type="filepath", sources=["upload", "microphone"])
                        
                        submit_btn = gr.Button("🚀 สร้างเสียงพูด (Generate Speech)", variant="primary", size="lg")
                        
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎧 2. ผลลัพธ์ (Raw Output)")
                        status_output = gr.Markdown("🟢 **สถานะ:** พร้อมใช้งาน (รอรับคำสั่ง)")
                        output_audio = gr.Audio(label="เสียงที่ถูกสร้างขึ้น (Raw Generated Audio)", interactive=False)

            # ------------------- TAB 2: Studio Post-Processing -------------------
            with gr.Tab("🎛️ 2. ปรับแต่งเสียงสตูดิโอ (Studio Post-Processing)"):
                gr.Markdown("เพิ่มความสมจริงระดับ Podcast (Proximity Effect & Compression) ให้เสียง AI ฟังดูแน่น มีน้ำหนัก")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎚️ เครื่องมือปรับแต่ง (Studio Effects)")
                        raw_audio_input = gr.Audio(label="เสียงตั้งต้น (Input Audio)", type="filepath")
                        
                        bass_boost = gr.Slider(minimum=0, maximum=12, value=4.5, step=0.5, label="เพิ่มความทุ้ม (Bass/Proximity Effect dB)")
                        treble_boost = gr.Slider(minimum=0, maximum=12, value=3.0, step=0.5, label="เพิ่มความคมชัด (Treble/Air dB)")
                        compression_ratio = gr.Slider(minimum=1, maximum=8, value=3.0, step=0.5, label="ระดับความแน่น (Compression Ratio)")
                        enable_gate = gr.Checkbox(value=True, label="เปิดระบบตัดเสียงรบกวนรอบข้าง (Noise Gate)")
                        
                        process_btn = gr.Button("🎧 ประมวลผลเสียงระดับสตูดิโอ (Process)", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🌟 ผลลัพธ์สุดท้าย (Mastered Output)")
                        studio_status = gr.Markdown("🟢 **สถานะ:** รอการปรับแต่ง")
                        studio_audio_output = gr.Audio(label="เสียงที่ผ่านการปรับแต่งแล้ว (Studio Processed)", interactive=False)

        # ------------------- ด้านขวา: กรอบแสดง System Logs -------------------
        with gr.Column(scale=1): # พื้นที่ 25% สำหรับแสดง Logs ตลอดเวลา
            gr.Markdown("### 💻 System Activity Logs")
            gr.Markdown("แสดงสถานะการทำงานของเบื้องหลัง (Live)")
            
            # Textbox สำหรับแสดง Log (ไม่อนุญาตให้พิมพ์แก้)
            logs_display = gr.Textbox(
                label="Terminal Output", 
                lines=20, 
                interactive=False, 
                value="[System] Initialized. Ready for actions.",
                elem_id="terminal-log" # ตั้ง id ไว้ เผื่ออยากเอา CSS ไปตกแต่งเป็นสีดำตัวหนังสือเขียวทีหลังได้
            )
            
            # ปุ่มสำหรับล้าง Log
            clear_log_btn = gr.Button("🗑️ Clear Logs", size="sm")

    # ------------------- Event Listeners -------------------
    # TAB 1: อัปเดต Logs พร้อมกับผลลัพธ์
    submit_btn.click(
        fn=generate_tts,
        inputs=[engine_dropdown, lang_dropdown, text_input, ref_audio_input, system_logs_state],
        outputs=[output_audio, raw_audio_input, status_output, system_logs_state]
    ).then(
        # เมื่อทำงานเสร็จ ให้อัปเดต UI text block ด้วยค่า State ล่าสุด
        fn=lambda log_text: log_text,
        inputs=[system_logs_state],
        outputs=[logs_display]
    )

    # TAB 2: อัปเดต Logs ของ Studio Process
    process_btn.click(
        fn=process_studio_audio,
        inputs=[raw_audio_input, enable_gate, bass_boost, treble_boost, compression_ratio, system_logs_state],
        outputs=[studio_audio_output, studio_status, system_logs_state]
    ).then(
        fn=lambda log_text: log_text,
        inputs=[system_logs_state],
        outputs=[logs_display]
    )
    
    # ล้าง Log
    clear_log_btn.click(
        fn=lambda: ("", ""), 
        inputs=None, 
        outputs=[system_logs_state, logs_display]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)