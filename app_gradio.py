import gradio as gr
import time
import os

# ฟังก์ชันจำลองการทำงานของ AI (แทนที่ด้วยโค้ด OmniVoice/CosyVoice ของจริงได้เลย)
def generate_tts(engine, language, text, ref_audio):
    # 1. ตรวจสอบความถูกต้องของข้อมูล (Validation)
    if not text.strip():
        yield None, "❌ **เกิดข้อผิดพลาด:** กรุณาป้อนข้อความที่ต้องการ"
        return
    if not ref_audio:
        yield None, "❌ **เกิดข้อผิดพลาด:** กรุณาอัปโหลดหรือบันทึกเสียงต้นแบบ (Reference Audio)"
        return

    # 2. อัปเดตสถานะให้ผู้ใช้ทราบว่ากำลังประมวลผล (Responsive UI)
    yield None, f"⏳ **กำลังประมวลผล:** ดึงเนื้อเสียงของคุณผ่าน {engine} (ภาษา: {language}) กรุณารอสักครู่..."
    
    # 3. จำลองระยะเวลาการรัน AI ของ GPU (2 วินาที)
    time.sleep(2) 
    
    # [จุดที่ต้องแก้ไข] - โค้ดของจริงให้เรียกโมเดลมา Infer ตรงนี้
    # เช่น output_audio_path = my_tts_model.infer(text, ref_audio)
    
    # สำหรับการเทสต์ เราจะคืนค่าเสียงต้นแบบกลับไปเพื่อให้รู้ว่าระบบทำงานผ่าน
    output_audio_path = ref_audio 

    # 4. ส่งมอบผลลัพธ์กลับไปยังหน้าเว็บ
    yield output_audio_path, f"✅ **สำเร็จ:** ประมวลผลข้อความด้วย {engine} เสร็จสิ้น!"

# สร้างหน้าเว็บด้วย Gradio Blocks เพื่อการจัดวางที่ยืดหยุ่น (Responsive Layout)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    # --- ส่วนหัว (Header) ---
    gr.Markdown(
        """
        # 🎙️ Zero-Shot TTS Studio (OmniVoice & CosyVoice)
        ระบบโคลนเสียงแบบ Zero-Shot เพียงอัปโหลดเสียงต้นแบบสั้นๆ (3-10 วินาที) ระบบจะนำน้ำเสียงนั้นไปพูดตามข้อความที่คุณต้องการ
        """
    )
    
    with gr.Row():
        # --- คอลัมน์ซ้าย: ส่วนรับข้อมูล (Inputs) ---
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 1. ตั้งค่าเสียง (Settings)")
            with gr.Row():
                engine_dropdown = gr.Radio(
                    choices=["OmniVoice", "CosyVoice 3.0"], 
                    value="CosyVoice 3.0", 
                    label="TTS Engine (AI Model)"
                )
                lang_dropdown = gr.Dropdown(
                    choices=["Thai (th)", "English (en)", "Chinese (zh)"], 
                    value="Thai (th)", 
                    label="ภาษาเป้าหมาย (Target Language)"
                )
            
            gr.Markdown("### ✍️ 2. ข้อความ และ เสียงต้นแบบ")
            text_input = gr.Textbox(
                label="ข้อความที่ต้องการให้พูด (Text Prompt)", 
                lines=4, 
                placeholder="พิมพ์ข้อความที่ต้องการให้ AI พูดที่นี่..."
            )
            
            # รองรับทั้งการอัปโหลดไฟล์ และเปิดไมค์อัดเสียงสดบนหน้าเว็บ
            ref_audio_input = gr.Audio(
                label="เสียงต้นแบบ (Reference Audio)", 
                type="filepath", 
                sources=["upload", "microphone"]
            )
            
            submit_btn = gr.Button("🚀 สร้างเสียงพูด (Generate Speech)", variant="primary", size="lg")
            
        # --- คอลัมน์ขวา: ส่วนแสดงผลลัพธ์ (Outputs) ---
        with gr.Column(scale=1):
            gr.Markdown("### 🎧 3. ผลลัพธ์ (Output)")
            
            # กล่องข้อความแสดงสถานะ (เช่น กำลังโหลด, สำเร็จ, เกิดข้อผิดพลาด)
            status_output = gr.Markdown("🟢 **สถานะ:** พร้อมใช้งาน (รอรับคำสั่ง)")
            
            output_audio = gr.Audio(
                label="เสียงที่ถูกสร้างขึ้น (Generated Audio)", 
                interactive=False
            )

    # --- ส่วนเชื่อมโยงการโต้ตอบ (Event Listeners) ---
    # เมื่อกดปุ่ม submit ให้ทำงานที่ฟังก์ชัน generate_tts
    submit_btn.click(
        fn=generate_tts,
        inputs=[engine_dropdown, lang_dropdown, text_input, ref_audio_input],
        outputs=[output_audio, status_output]
    )

if __name__ == "__main__":
    # เปิดใช้งานแบบ Public link เพื่อให้แชร์ได้ (share=True)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
