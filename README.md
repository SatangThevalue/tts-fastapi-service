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
*   **FFmpeg Complex Filter (Ultimate Performance Architecture)**: เปลี่ยนจากการใช้ MoviePy มาเป็นการเขียนสคริปต์ประกอบคำสั่ง FFmpeg ตรงแบบม้วนเดียวจบ (Single-pass Processing) ลดการอ่านเขียนไฟล์ I/O และประหยัด RAM เซิร์ฟเวอร์ได้มหาศาล ทำหน้าที่ครอปภาพแนวตั้ง 9:16, สวมเสียงพากย์, และใส่ข้อความทับลงไปได้อย่างรวดเร็ว
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

เราได้เตรียมคู่มือ API โดยละเอียดพร้อม **5 ตัวอย่างการใช้งานจริง** (เช่น การลบเสียง, แปลงคลิปแนวตั้ง, Mix เสียง) สามารถอ่านได้ที่:
👉 [**API_EXAMPLES.md**](./API_EXAMPLES.md)

### จุดเด่นของ API ปัจจุบัน:
1. **Ultimate Performance FFmpeg**: ทุก Endpoint วิดีโอทำงานด้วยสคริปต์ FFmpeg Complex Filter ชั้นสูง ประหยัด RAM ไม่ต้องใช้ MoviePy
2. **Auto-Provisioning**: มีระบบดาวน์โหลด Base Models (Piper TTS) อัตโนมัติเมื่อติดตั้งระบบใหม่


### 🔤 ฟอนต์ภาษาไทยฟรี (Commercial Use)
ระบบได้ทำการติดตั้งฟอนต์ 7 รูปแบบที่พร้อมให้ใช้งานผ่าน n8n (`font_name`) ดังนี้:
1. `Sarabun-Bold` (ค่าเริ่มต้น - ทางการ สุภาพ)
2. `Prompt-Bold` (โมเดิร์น ไร้หัว อ่านง่าย)
3. `Kanit-Bold` (โดดเด่น วัยรุ่น ยอดฮิต)
4. `Mitr-Bold` (กว้าง ชัดเจน เป็นมิตร)
5. `ChakraPetch-Bold` (เหลี่ยม ทันสมัย เทคโนโลยี)
6. `Pridi-Bold` (มีหัว คลาสสิก)
7. `Pattaya-Regular` (มีหัว ลายมือ ตวัด)
