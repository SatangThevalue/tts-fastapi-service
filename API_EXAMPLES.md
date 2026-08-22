# 📚 คู่มือการใช้งาน API (n8n Integration Guide)

ระบบ **AI Media Studio Pro** รองรับการทำงานแบบ Automation ขั้นสูงผ่าน HTTP Requests ซึ่งออกแบบมาเพื่อใช้กับ `n8n`, `Make`, หรือ Custom Scripts โดยเฉพาะ

---

## 🟢 Endpoint 1: อัปเดตและดาวน์โหลดโมเดลพื้นฐานอัตโนมัติ
หากคุณนำระบบไป Deploy ใหม่บนเซิร์ฟเวอร์ที่ว่างเปล่า สามารถสั่งให้ระบบดาวน์โหลดโมเดล Piper TTS พื้นฐานได้ทันที
* **URL:** `http://localhost:7860/api/models/download_defaults`
* **Method:** `POST`
* **ประโยชน์:** ป้องกันปัญหา (No Piper models found)

---

## 🟢 Endpoint 2: ประมวลผลวิดีโอแบบ Ultimate Performance (FFmpeg)
ระบบนี้ถอด MoviePy ทิ้งและใช้ **FFmpeg Complex Filter** แบบม้วนเดียวจบ (Single-pass Processing) เพื่อรีดประสิทธิภาพสูงสุด ลด I/O และประหยัด RAM

* **URL:** `http://localhost:7860/api/video/edit`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data` (ใช้ Form-Data ใน n8n)

### 💡 5 ตัวอย่างการใช้งานจริงใน n8n

#### Example 1: ลบเสียงวิดีโอต้นฉบับทิ้งอย่างเดียว (Mute Original Video)
เหมาะสำหรับเตรียมคลิปดิบ (B-roll) ก่อนนำไปใช้ต่อ
* **Parameters (Form-Data):**
  - `video_path`: `/absolute/path/to/raw_video.mp4`
  - `mute_original_audio`: `true`

#### Example 2: สวมเสียงพากย์ AI และ Mix เสียงเพลงพื้นหลังเบาๆ
เหมาะสำหรับคลิปพอดแคสต์ หรือคลิปนำเสนอ
* **Parameters:**
  - `video_path`: `/path/to/video.mp4`
  - `bgm_path`: `/path/to/ai_voiceover.wav`
  - `bgm_volume`: `1.0` (เร่งเสียงพากย์ให้ดัง)
  - `mute_original_audio`: `true` (ปิดเสียงสภาพแวดล้อมเดิม)

#### Example 3: สร้างคลิปแนวตั้ง (Shorts/Reels) พร้อมแคปชัน
เหมาะสำหรับนำวิดีโอแนวนอนมาแปลงลง TikTok
* **Parameters:**
  - `video_path`: `/path/to/landscape_video.mp4`
  - `short_video_format`: `true` (ระบบจะครอปภาพกึ่งกลางเป็น 9:16)
  - `text_lines`: "รีวิวสินค้าล่าสุด!"
  - `font_size`: `60`
  - `font_color`: `yellow`
  - `font_name`: `Kanit-Bold` (ตัวเลือกฟอนต์)

#### Example 4: สร้างคลิปแบบม้วนเดียวจบ (All-in-One: ครอป + ใส่ข้อความ + ปิดเสียงเดิม + ใส่ BGM)
รีดประสิทธิภาพ FFmpeg สูงสุด ทำทุกอย่างในคำสั่งเดียว
* **Parameters:**
  - `video_path`: `/path/to/input.mp4`
  - `short_video_format`: `true`
  - `text_lines`: "โปรโมชันด่วน! วันนี้เท่านั้น"
  - `mute_original_audio`: `true`
  - `bgm_path`: `/path/to/trending_music.mp3`
  - `bgm_volume`: `0.3`

#### Example 5: ใส่แค่ลายน้ำตัวหนังสือ (Watermark Only)
* **Parameters:**
  - `video_path`: `/path/to/corporate_video.mp4`
  - `text_lines`: "© Satang AI Corporation"
  - `font_size`: `30`
  - `font_color`: `white`
  - `font_name`: `ChakraPetch-Bold`
  - *(ไม่ระบุค่าอื่น ระบบจะคงขนาดวิดีโอเดิมและเสียงเดิมไว้)*

---
*หมายเหตุ: ผลลัพธ์ทุก Request จะตอบกลับเป็น JSON พร้อมบอกพาทไฟล์ผลลัพธ์ (`output_path`) และคำสั่ง FFmpeg ที่ถูกรันจริงๆ (`command_used`)*
