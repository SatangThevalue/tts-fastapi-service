import gradio as gr
import time
import os
import soundfile as sf
import tempfile
from pedalboard import Pedalboard, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Limiter

# ==========================================
# 1. ฟังก์ชันส่วน TTS Generation
# ==========================================
def generate_tts(engine, language, text, ref_audio):
    if not text.strip():
        yield None, None, "❌ **เกิดข้อผิดพลาด:** กรุณาป้อนข้อความที่ต้องการ"
        return
    if not ref_audio:
        yield None, None, "❌ **เกิดข้อผิดพลาด:** กรุณาอัปโหลดหรือบันทึกเสียงต้นแบบ (Reference Audio)"
        return

    yield None, None, f"⏳ **กำลังประมวลผล:** ดึงเนื้อเสียงของคุณผ่าน {engine} (ภาษา: {language}) กรุณารอสักครู่..."
    
    # จำลองระยะเวลาการรัน AI ของ GPU
    time.sleep(2) 
    
    # [จุดที่ต้องแก้ไข] - โค้ดของจริงให้เรียกโมเดลมา Infer ตรงนี้
    output_audio_path = ref_audio 

    # รีเทิร์น 3 ค่า: 
    # 1. output_audio (แสดงใน Tab 1)
    # 2. raw_audio_input (ส่งต่อไปยัง Tab 2 เป็น Input อัตโนมัติ)
    # 3. status_output (อัปเดตสถานะ Tab 1)
    yield output_audio_path, output_audio_path, f"✅ **สำเร็จ:** ประมวลผลข้อความด้วย {engine} เสร็จสิ้น! ➔ นำไปปรับแต่งต่อในแท็บ '🎛️ Studio Post-Processing' ได้เลย"

# ==========================================
# 2. ฟังก์ชันส่วน Studio Post-Processing (ด้วย Pedalboard)
# ==========================================
def process_studio_audio(input_audio, enable_gate, bass_boost, treble_boost, compression_ratio):
    if not input_audio:
        return None, "❌ กรุณาอัปโหลดเสียง หรือกลับไปสร้างเสียงในแท็บแรกก่อน"

    try:
        # อ่านไฟล์เสียง
        audio_data, sample_rate = sf.read(input_audio)
        
        # ถ้าระบบส่งกลับมาเป็น 2 dimensions (Stereo) ให้ยุบเป็น Mono/Stereo ที่ Pedalboard รองรับได้ง่าย
        if len(audio_data.shape) > 1:
            audio_data = audio_data.T # Transpose for pedalboard format (channels, samples)
        
        # สร้าง Chain ของ Effect เกรดสตูดิโอ (Podcast Style)
        board = Pedalboard([
            # 1. Noise Gate: ตัดเสียงซ่า/เสียงรบกวนเบาๆ ตอนที่ไม่มีคนพูด
            NoiseGate(threshold_db=-40.0, ratio=1.5, release_ms=250) if enable_gate else None,
            
            # 2. Highpass Filter: ตัดเสียงหึ่ง/ลมกระแทกไมค์ที่ย่านความถี่ต่ำมาก (Below 80Hz)
            HighpassFilter(cutoff_frequency_hz=80),
            
            # 3. EQ: Podcast/Radio Voice (ทุ้มลึก และ ปลายแหลมชัดใส)
            LowShelfFilter(cutoff_frequency_hz=120, gain_db=bass_boost), # เพิ่มความทุ้ม (Proximity Effect)
            HighShelfFilter(cutoff_frequency_hz=6000, gain_db=treble_boost), # เพิ่มความคมชัด (Air/Clarity)
            
            # 4. Compressor: บีบอัดเสียงให้สม่ำเสมอ ฟังสบาย ไม่สะดุ้งเมื่อเสียงดัง
            Compressor(threshold_db=-15, ratio=compression_ratio, attack_ms=2.0, release_ms=100),
            
            # 5. Limiter: ป้องกันเสียงแตก (Clipping) ขาออก
            Limiter(threshold_db=-1.0)
        ])
        
        # ลบ None ออกจาก list ถ้าผู้ใช้ไม่ได้เปิดใช้งาน Noise Gate
        board = Pedalboard([effect for effect in board if effect is not None])

        # ปรับแต่งเสียง (Run the audio through the pedalboard)
        effected_audio = board(audio_data, sample_rate)
        
        # หากก่อนหน้านี้ transpose ไป ต้องทำกลับมาให้อยู่ในรูป (samples, channels) สำหรับ soundfile
        if len(effected_audio.shape) > 1:
             effected_audio = effected_audio.T

        # บันทึกไฟล์ชั่วคราว
        temp_dir = tempfile.gettempdir()
        output_file = os.path.join(temp_dir, f"studio_processed_{int(time.time())}.wav")
        sf.write(output_file, effected_audio, sample_rate)

        return output_file, "✅ **สำเร็จ:** ประมวลผลเสียงแบบ Studio Podcast เรียบร้อยแล้ว!"

    except Exception as e:
         return None, f"❌ **Error:** {str(e)}"

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
        gr.Markdown("เพิ่มความสมจริงระดับ Podcast (Proximity Effect & Compression) ให้เสียง AI ฟังดูแน่น มีน้ำหนัก และไม่แข็งกระด้าง")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎚️ เครื่องมือปรับแต่ง (Studio Effects)")
                
                # กล่องรับเสียงดิบ (จะถูกลิงก์อัตโนมัติจาก Tab 1 ถ้าเจนเสร็จ)
                raw_audio_input = gr.Audio(label="เสียงตั้งต้น (Input Audio)", type="filepath")
                
                # Sliders สำหรับปรับ EQ และ Compressor
                bass_boost = gr.Slider(minimum=0, maximum=12, value=4.5, step=0.5, label="เพิ่มความทุ้ม (Bass/Proximity Effect dB)", info="เพิ่มน้ำหนักเสียงให้ฟุ้มลึกเหมือนอยู่ใกล้ไมค์")
                treble_boost = gr.Slider(minimum=0, maximum=12, value=3.0, step=0.5, label="เพิ่มความคมชัด (Treble/Air dB)", info="เพิ่มความใส ปลายเสียงหายใจ (Crispness)")
                compression_ratio = gr.Slider(minimum=1, maximum=8, value=3.0, step=0.5, label="ระดับความแน่น (Compression Ratio)", info="ยิ่งมาก เสียงที่เบากับดังจะยิ่งถูกบีบให้เท่ากัน (ฟังสบาย)")
                enable_gate = gr.Checkbox(value=True, label="เปิดระบบตัดเสียงรบกวนรอบข้าง (Noise Gate)")
                
                process_btn = gr.Button("🎧 ประมวลผลเสียงระดับสตูดิโอ (Process Studio Audio)", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🌟 ผลลัพธ์สุดท้าย (Mastered Output)")
                studio_status = gr.Markdown("🟢 **สถานะ:** รอการปรับแต่ง")
                studio_audio_output = gr.Audio(label="เสียงที่ผ่านการปรับแต่งแล้ว (Studio Processed Audio)", interactive=False)

    # ------------------- Event Listeners -------------------
    # เมื่อ Tab 1 เสร็จ -> ส่งเสียงไปที่ output_audio(Tab 1) และ โยนไปใส่ช่อง raw_audio_input(Tab 2) อัตโนมัติ
    submit_btn.click(
        fn=generate_tts,
        inputs=[engine_dropdown, lang_dropdown, text_input, ref_audio_input],
        outputs=[output_audio, raw_audio_input, status_output]
    )

    # เมื่อกดปุ่มใน Tab 2 -> รันระบบ Pedalboard Masterting
    process_btn.click(
        fn=process_studio_audio,
        inputs=[raw_audio_input, enable_gate, bass_boost, treble_boost, compression_ratio],
        outputs=[studio_audio_output, studio_status]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
