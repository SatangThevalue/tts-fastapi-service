# 📐 API Endpoint Design for Video Templates

เพื่อรองรับ Use Case รูปแบบอัตโนมัติ 2 แบบ (Quote 16:9 และ Pro Vlog) เราจะสร้าง **API Endpoints ย่อยแยกกันชัดเจน** เพื่อให้ n8n เรียกใช้งานได้ง่าย ลดความสับสนของพารามิเตอร์ และง่ายต่อการปรับแต่งในอนาคต

---

## 🟢 Endpoint 1: The Quote & Tip (คำคม / 16:9 Transformation)
**Endpoint:** `POST /api/video/templates/quote169`
**หน้าที่:** นำวิดีโอ (ไม่ว่าสัดส่วนไหน) มาบังคับเป็นแนวนอน 16:9 โดยใช้เทคนิคฉากหลังเบลอ (Smart Background Blur) ลบเสียงเดิม แปะข้อความตรงกลาง และสวมเสียง TTS/BGM ใหม่

**Parameters (Form-Data):**
* `video_path` (Required): พาธไฟล์วิดีโอต้นฉบับ
* `text_lines` (Required): ข้อความคำคมหรือเทคนิค
* `audio_path` (Optional): เสียงพากย์ (TTS) หรือ BGM ที่จะนำมาทับเสียงเดิม
* `font_size` (Optional): ขนาดตัวหนังสือ (Default: 60)
* `font_color` (Optional): สีตัวหนังสือ (Default: white)

**FFmpeg Logic เบื้องหลัง:**
ใช้ `filter_complex` แย่งสายสัญญาณวิดีโอเป็น 2 สาย สายแรกขยายและใส่ `gblur` สายสองคงสัดส่วนเดิมและ `overlay` ทับตรงกลาง จากนั้นใส่ `drawtext` และดึงเสียงจาก `audio_path` มาใส่แทน (`-map 0:v -map 1:a`)

---

## 🟢 Endpoint 2: The Pro Vlog (ซับไตเติลเด้ง + เสียงสตูดิโอ)
**Endpoint:** `POST /api/video/templates/pro_vlog`
**หน้าที่:** นำวิดีโอและเสียงพูด (หรือวิดีโอที่มีเสียงอยู่แล้ว) มาทำ Audio Mastering ให้เป็นเสียงสตูดิโอ ถอดความด้วย Faster-Whisper และเบิร์นซับไตเติลแบบ Karaoke Style ทับลงไป

**Parameters (Form-Data):**
* `video_path` (Required): พาธไฟล์วิดีโอหลัก
* `audio_path` (Optional): พาธไฟล์เสียง (หากต้องการทับเสียงเดิม ถ้าไม่ส่งมาจะใช้เสียงจากวิดีโอ)
* `subtitle_style` (Optional): สไตล์ซับไตเติล เช่น `tiktok_yellow`, `classic_white` (Default: tiktok_yellow)
* `enhance_audio` (Optional): `true/false` (เปิดโหมด Studio Mastering ผ่าน Pedalboard หรือไม่ - Default: true)

**Logic เบื้องหลัง:**
1. สกัดไฟล์เสียงออกมา (หากไม่มี `audio_path`)
2. ส่งเสียงผ่าน `Pedalboard` (Compressor, EQ, NoiseGate) 
3. ส่งเสียงเข้า `subtitle_engine.py` (Faster-Whisper) เพื่อสกัดและสร้างไฟล์ `.ass`
4. ใช้ FFmpeg คำสั่ง `-vf "ass=subtitle.ass"` เบิร์นซับไตเติลลงวิดีโอพร้อมสวมเสียงที่ Mastered แล้ว

---

## 🟢 (มีอยู่แล้ว) Endpoint 3: Raw / Dynamic Editor
**Endpoint:** `POST /api/video/edit`
**หน้าที่:** สำหรับการตัดต่อยิบย่อยแบบปรับค่าพารามิเตอร์อิสระ (Mute, Crop 9:16, Mix BGM) ซึ่งเราทำเสร็จไปแล้ว

---

## 🛠️ แผนการพัฒนา (Action Plan - Phase 2)
1. **ติดตั้ง Dependencies:** `faster-whisper`
2. **สร้าง Core Engines:** 
   - สร้างไฟล์ `subtitle_engine.py` สำหรับสร้างไฟล์ `.ass`
   - สร้างฟังก์ชัน `apply_smart_blur()` ใน `ffmpeg_processor.py`
3. **ปรับแต่ง `app.py`:** เพิ่ม 2 Endpoints ใหม่ ตามโครงสร้างด้านบน
4. **ทดสอบระบบ:** จำลองสร้างคลิป Quote 16:9 และ VLOG 1 คลิป เพื่อดูผลลัพธ์
