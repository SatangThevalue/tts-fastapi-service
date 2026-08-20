# 🎬 AI Media Studio (Pro Edition)

ระบบสถาปัตยกรรมระดับองค์กร (Production-Grade) สำหรับการสร้างและปรับแต่งสื่อมัลติมีเดียอัตโนมัติ (Audio & Vertical Video) รองรับการใช้งานผ่าน Web UI (Gradio), การทำงานอัตโนมัติผ่าน n8n (FastAPI), และการใช้งานร่วมกับ AI Agents ผ่าน MCP Protocol

โปรเจ็กต์นี้ออกแบบมาให้ทำงานได้ทั้งในสภาวะที่มี GPU (เน้นอารมณ์) และไม่มี GPU (Fast CPU Offline) โดยมี **Piper TTS** เป็นพระเอกตัวหลักในการทำงานแบบ Offline

---

## 📑 สารบัญ
1. [เทคโนโลยีและเครื่องมือเบื้องหลัง (Under the Hood)](#1-เทคโนโลยีและเครื่องมือเบื้องหลัง-under-the-hood)
2. [เทคนิคการปรับแต่งเสียงขั้นสูง (Audio Engineering Techniques)](#2-เทคนิคการปรับแต่งเสียงขั้นสูง-audio-engineering-techniques)
3. [การติดตั้งและรันระบบด้วย `uv`](#3-การติดตั้งและรันระบบด้วย-uv)
4. [โครงสร้างสถาปัตยกรรม (Project Structure)](#4-โครงสร้างสถาปัตยกรรม-project-structure)
5. [คู่มือการใช้งาน API (สำหรับ n8n)](#5-คู่มือการใช้งาน-api-สำหรับ-n8n)
6. [คู่มือหน้าเว็บ Gradio UI](#6-คู่มือหน้าเว็บ-gradio-ui)

---

## 1. เทคโนโลยีและเครื่องมือเบื้องหลัง (Under the Hood)

ระบบนี้ผสมผสานไลบรารีระดับโลกเข้าด้วยกันเพื่อให้ได้ผลลัพธ์ที่สมบูรณ์แบบที่สุด:

*   **FastAPI & Uvicorn**: โครงสร้างกระดูกสันหลังของระบบ จัดการ HTTP Requests แบบ Asynchronous สามารถรองรับโหลดจากระบบ Automation อย่าง n8n ได้อย่างเสถียร
*   **Gradio (v4+)**: เฟรมเวิร์กสำหรับสร้างหน้า Web UI ที่ User-friendly ทรงพลังด้วยความสามารถในการ Mount ตัวเองเข้ากับ FastAPI (ทำงานพอร์ต 7860 เดียวกัน)
*   **Piper TTS (`piper-tts`)**: โมเดล Text-to-Speech ขุมพลัง CPU ที่ทำงานแบบ Offline 100% เร็วกว่าเวลาจริงหลายสิบเท่า และรองรับภาษาไทย (`.onnx` models)
*   **Edge TTS (`edge-tts`)**: โมเดลเสียงจาก Microsoft สำหรับงานที่ต้องการความเร็วระดับเสี้ยววินาทีและเสียงที่เป็นธรรมชาติสูง (รันบน CPU แต่ต้องการอินเทอร์เน็ต)
*   **Pedalboard (by Spotify)**: ไลบรารี C++ สำหรับประมวลผลเสียง (DSP) ระดับสตูดิโอ ทำงานไวกว่าไลบรารีเสียงทั่วไปหลายเท่า 
*   **PyDub & SoundFile**: จัดการโครงสร้างคลื่นเสียง (Waveforms) พื้นฐาน เช่น การตัดต่อ, การหาความเงียบ (Silence Detection), และการเขียนไฟล์ `.wav` / `.flac`
*   **MoviePy**: เอนจิ้นตัดต่อวิดีโอ (Video Automation) อาศัย FFmpeg เบื้องหลัง ทำหน้าที่ครอปภาพแนวตั้ง 9:16, สวมเสียงพากย์, และใส่ข้อความทับลงไป
*   **FastMCP**: โพรโทคอลมาตรฐาน (Model Context Protocol) ที่ทำให้แอปพลิเคชัน AI ภายนอก (เช่น Claude Desktop หรือ Cursor) สามารถสื่อสารและสั่งให้เซิร์ฟเวอร์นี้ผลิตเสียงได้โดยตรง

---

## 2. เทคนิคการปรับแต่งเสียงขั้นสูง (Audio Engineering Techniques)

ความลับที่ทำให้เสียง AI ของระบบนี้ฟังดู "เหมือนคนจริงๆ อัดเสียงในสตูดิโอ" คือเทคนิค Post-Processing เหล่านี้:

1.  **Foley & Breath Insertion (Generative Silence Detection)**
    *   *ปัญหา:* AI มักจะหยุดพูดแล้วเงียบสนิทแบบไร้เสียง (Digital Silence) ทำให้ดูปลอม
    *   *เทคนิค:* ใช้ `pydub.silence` ตรวจจับช่องว่างระหว่างประโยค (เกิน 400ms) จากนั้นนำไฟล์เสียง "สูดลมหายใจของมนุษย์" มายัดแทรกเข้าไป หากไม่มีไฟล์จริง ระบบจะใช้คณิตศาสตร์สร้างคลื่น White-noise แบบ Fade-in/out สั้นๆ มาหลอกจิตใต้สำนึกคนฟัง
2.  **Dynamic EQ (De-essing)**
    *   ใช้ `HighShelfFilter` จาก Pedalboard ดักกดความถี่ช่วง `6500 Hz` ลง `-2.0 dB` (จุดที่เสียง ส. มักจะบาดหู) ก่อนที่จะไปเร่งความใส (Air) ที่ย่าน `10000 Hz` ขึ้นไป
3.  **Harmonic Tape Saturation**
    *   ใส่ความอุ่น (Warmth) คล้ายเทปอนาล็อก โดยใช้ `Distortion(drive_db=5.0)` ทับเข้าไปเป็น Layer อ่อนๆ (ประมาณ 5-10%) เพื่อสร้าง Harmonic Overtones ลบความแข็งกระด้างแบบดิจิทัล
4.  **Convolution Reverb**
    *   เปลี่ยนจากการใช้ Algorithm จำลองเสียงก้อง เป็นการสวม "ลายนิ้วมือของสถานที่จริง (Impulse Response)" ลงไปในเสียง ทำให้เนื้อเสียงมีมิติและฟังสบายเหมือนนั่งฟังพอดแคสต์ในห้องปิด
5.  **Micro-Modulation (Humanize Mode)**
    *   ใส่เอฟเฟกต์ `Chorus` ที่ความลึก 5% เพื่อสร้างการสั่นแกว่งของเส้นเสียงจำลอง ลบล้าง "ความคงที่เป๊ะๆ" ของเสียงคอมพิวเตอร์

---

## 3. การติดตั้งและรันระบบด้วย `uv`

ขอแนะนำให้รันด้วย **`uv`** (Rust-based package manager) เพื่อความเร็วสูงสุดในการ Deploy

**1. ติดตั้ง `uv` (หากยังไม่มี):**
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**2. รันแบบ One-Liner (เหมาะสำหรับทดสอบ):**
```bash
uv run --with-requirements requirements.txt app.py
```

**3. รันแบบ Production (บน VPS):**
```bash
# สร้าง Environment และติดตั้ง dependencies ภายใน 1 วินาที
uv venv
uv pip install -r requirements.txt

# สตาร์ทเซิร์ฟเวอร์ด้วย Uvicorn (เปิดพอร์ต 7860 รับแขก)
./.venv/bin/uvicorn app:app --host 0.0.0.0 --port 7860 --workers 4
```

---

## 4. โครงสร้างสถาปัตยกรรม (Project Structure)

ระบบมีกลไกจัดระเบียบโฟลเดอร์อัตโนมัติ ดังนี้:

```text
tts-fastapi-service/
├── app.py                   # หัวใจหลักของระบบ (All-in-One: FastAPI + Gradio + MCP)
├── requirements.txt         # รายชื่อไลบรารีที่จำเป็น
├── pretrained_models/       
│   ├── piper_voices/        # 📥 วางไฟล์โมเดล Piper (.onnx, .json) ที่นี่
│   └── speakers/            # 📥 วางโมเดล GPU Fine-tuned (CosyVoice/OmniVoice) ที่นี่
├── assets/
│   ├── foley/               # 📥 วางไฟล์เสียงลมหายใจ (.wav) เพื่อใช้ทำ Foley
│   └── impulse_responses/   # 📥 วางไฟล์ IR (.wav) สำหรับทำ Convolution Reverb
└── temp/                    # โฟลเดอร์ชั่วคราว 
    ├── media_uploads/       # เก็บไฟล์ขาเข้า (ระบบจะ Auto-cleanup ทุก 1 ชม.)
    └── media_outputs/       # เก็บไฟล์ขาออก (ระบบจะ Auto-cleanup ทุก 1 ชม.)
```

---

## 5. คู่มือการใช้งาน API (สำหรับ n8n)

ระบบนี้ถูกออกแบบมาเพื่อ Automation อย่างแท้จริง ทุก Endpoints มีการป้องกัน **GPU/CPU OOM (Out of Memory)** ด้วยระบบ `asyncio.Lock()` เรียบร้อยแล้ว (เข้าคิวประมวลผลทีละ 1 งาน)

### 🔐 การตั้งค่าความปลอดภัย (Authentication)
หากต้องการป้องกันไม่ให้คนอื่นมาแอบใช้ API ให้ตั้งค่า Environment Variable บนเซิร์ฟเวอร์:
`export TTS_API_KEY="your_secret_key"`
เมื่อตั้งค่าแล้ว n8n ต้องแนบ Header: `X-API-Key: your_secret_key` มาในทุก Request

---

### 🎙️ API 1: สร้างเสียงพากย์ (TTS Endpoint)
เปลี่ยนข้อความให้กลายเป็นไฟล์เสียง `.wav`

*   **Endpoint:** `POST /api/tts/generate`
*   **Body Content Type:** `Multipart-Form Data`
*   **Parameters:**
    *   `text` *(String)* **[บังคับ]**: ข้อความที่ต้องการให้พูด (หากยาวมาก ระบบจะหั่นทีละประโยคให้อัตโนมัติ)
    *   `engine` *(String)*: เลือกขุมพลัง (Default: `PiperTTS (Fast CPU / Offline)`)
        *   ค่าที่รองรับ: `PiperTTS (Fast CPU / Offline)`, `EdgeTTS (Fast CPU / Online)`, `CosyVoice 3.0`, `OmniVoice`
    *   `piper_model` *(String)*: ระบุชื่อไฟล์โมเดล (เช่น `th_TH-ntsc-medium.onnx`) **หากไม่ระบุ ระบบจะดึงไฟล์ .onnx แรกที่เจอในเครื่องมาใช้ให้อัตโนมัติ**
    *   `speed` *(Float)*: ความเร็ว (Default: `1.0`)
    *   `apply_breaths` *(Boolean)*: แทรกเสียงหายใจ (Default: `false`)
    *   `apply_deessing` *(Boolean)*: ลดเสียงเสียดหู (Default: `false`)
    *   `apply_tape_saturation` *(Boolean)*: เพิ่มความอุ่นอนาล็อก (Default: `false`)

**📥 ผลลัพธ์:** ส่งกลับไฟล์เสียง Binary `audio/wav` 

---

### 🎬 API 2: ตัดต่อวิดีโอ (Video Editing Endpoint)
แปลง Footage วิดีโอธรรมดา ให้กลายเป็นวิดีโอแนวตั้ง (9:16) สวมเสียงพากย์ และขึ้น Subtitle แบบ List

*   **Endpoint:** `POST /api/video/edit`
*   **Body Content Type:** `Multipart-Form Data`
*   **Parameters:**
    *   `video_file` *(File)* หรือ `video_local_path` *(String)*: ไฟล์ Footage ต้นฉบับ
        *   *(Tip: การใช้ `video_local_path` จะเร็วกว่ามหาศาล หาก n8n รันอยู่บนเซิร์ฟเวอร์เดียวกัน เพราะข้ามขั้นตอนการอัปโหลดไปเลย)*
    *   `audio_file` *(File)* หรือ `audio_local_path` *(String)*: ไฟล์เสียงพากย์ที่จะนำมาสวมทับ
    *   `short_video_format` *(Boolean)*: บังคับ Crop หน้าจอเป็น 1080x1920 (Default: `true`)
    *   `mute_original_audio` *(Boolean)*: ปิดเสียงวิดีโอเดิม (Default: `true`)
    *   `text_lines` *(String)*: ข้อความที่จะวางเรียงกึ่งกลางจอ (ใส่ `\n` เพื่อขึ้นบรรทัดใหม่ / สร้างกล่องข้อความใหม่)
    *   `watermark_text` *(String)*: ลายน้ำที่มุมขวาล่าง

**📥 ผลลัพธ์:** ส่งกลับไฟล์วิดีโอ Binary `video/mp4` พร้อมโพสต์ลง Social Media

---

## 6. คู่มือหน้าเว็บ Gradio UI

หากคุณไม่ได้ใช้ n8n ก็สามารถเข้าใช้งานด้วยมือผ่าน Web UI ได้ที่ `http://<your-ip>:7860` 

*   **Tab 1 (Audio Tools):** หน้าต่างผลิตเสียงหลัก เลือก Engine, ปรับความเร็ว, และตั้งค่าอารมณ์
*   **Tab 2 (Model Manager):** หน้านี้สำคัญมาก! ใช้สำหรับ **อัปโหลด (Upload)** โมเดลเสียงใหม่ๆ (เช่น `.onnx` ของ Piper) หรือไฟล์น้ำหนัก Fine-tuned เข้าไปในเซิร์ฟเวอร์ และสามารถ **ลบ (Delete)** โมเดลที่ไม่ได้ใช้ทิ้งได้จากหน้าเว็บเลยโดยไม่ต้องต่อ SSH เข้าเซิร์ฟเวอร์ เมื่ออัปโหลดเสร็จกดปุ่ม "Refresh" เมนูทุกจุดจะอัปเดตโมเดลใหม่ให้อัตโนมัติ
*   **Tab 3 (Advanced Audio Mastering):** โซนสำหรับคนทำ Podcast ปรับแต่ง EQ, Reverb, De-essing ได้อย่างละเอียด พร้อมฟีเจอร์ "10 Studio Presets" ที่เซ็ตค่าให้พร้อมใช้
*   **Tab 4 (Video Automator):** หน้าอัปโหลดคลิป Footage และพิมพ์รายการข้อความ เพื่อเรนเดอร์คลิปวิดีโอ 9:16 แนวตั้งสำหรับโพสต์ลง Social
*   **System Logs (ด้านขวา):** แผง Console ที่โชว์การทำงานหลังบ้านทุกขั้นตอน ช่วยให้รู้ว่าตอนนี้ระบบกำลังทำอะไรอยู่ (เช่น กำลังคิวรัน GPU, ตัดต่อวิดีโอ, หรือหั่นข้อความ)