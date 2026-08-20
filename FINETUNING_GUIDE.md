# คู่มือการ Fine-tune และเพิ่มเสียงผู้พูด (Speaker Checkpoints)

เอกสารนี้จะอธิบายขั้นตอนตั้งแต่การเตรียมข้อมูล, การรัน Fine-tune (SFT) เบื้องต้น, และวิธีนำไฟล์น้ำหนัก (Weights) ที่ได้ มาติดตั้งเพื่อให้ระบบ `app.py` ใช้งานได้อัตโนมัติ

---

## 1. การเตรียม Dataset (ข้อมูลสอน AI)

AI ต้องการข้อมูล 2 ส่วนคือ **ไฟล์เสียงสั้นๆ (3-10 วินาที)** และ **ข้อความที่ตรงกับเสียงนั้น (Transcript)**
เราได้เตรียมสคริปต์อัตโนมัติเพื่อช่วยหั่นไฟล์เสียงยาวๆ ให้เป็นท่อนสั้นตามจังหวะเงียบ/หายใจ

### วิธีใช้งานสคริปต์
1. นำไฟล์เสียงของคุณ (เช่น `my_voice.wav` ความยาว 10 นาที) มาวางในเครื่อง
2. ติดตั้งแพ็กเกจ: `pip install pydub` (และต้องติดตั้ง `ffmpeg` ในระบบ)
3. รันคำสั่ง:
   ```bash
   python tools/dataset_prep.py --audio my_voice.wav --out my_custom_dataset
   ```
4. สคริปต์จะสร้างโฟลเดอร์ `my_custom_dataset/wavs/` (เก็บไฟล์เสียงที่ถูกหั่น) และไฟล์ `metadata.jsonl`
5. **งานของคุณ:** เปิดไฟล์ `metadata.jsonl` แล้วพิมพ์ข้อความให้ตรงกับเสียงแต่ละไฟล์ (แนะนำให้ใช้ AI อย่าง OpenAI Whisper มารันเพื่อดึง Text ใส่เข้าไปอัตโนมัติ จะประหยัดเวลามาก)

---

## 2. การรัน Fine-tune (Supervised Fine-Tuning)

เมื่อได้ Dataset แล้ว การเทรนจะต้องรันโค้ดของ Official Repository 
*(แนะนำให้รันบน Google Colab GPU)*

### สำหรับ CosyVoice 3.0:
อ้างอิงจาก Repository ของ Alibaba:
```bash
python tools/train.py --config conf/cosyvoice.yaml --train_data my_custom_dataset/metadata.jsonl --checkpoint_dir /path/to/save/checkpoints
```
หลังเทรนเสร็จ คุณจะได้ไฟล์โมเดล เช่น `epoch_10.pt` หรือโฟลเดอร์โมเดลที่สมบูรณ์

---

## 3. การเปลี่ยนเสียงคนพูดคนที่ 1, คนที่ N (โหลดเข้าสู่ระบบ)

เมื่อคุณได้ไฟล์น้ำหนัก (Weights / Checkpoints) จากการ Fine-tune มาแล้ว ไม่ว่าจะกี่คนก็ตาม ระบบของเราถูกออกแบบมาให้ทำ **Auto-Discovery** (ค้นหาอัตโนมัติ) 

### วิธีติดตั้งโมเดลเสียง
1. เข้าไปที่โฟลเดอร์ `pretrained_models/speakers/`
2. สร้างโฟลเดอร์ใหม่ โดยตั้งชื่อเป็นชื่อผู้พูด เช่น:
   - `pretrained_models/speakers/Kru_Satang_Voice/`
   - `pretrained_models/speakers/Female_Assistant/`
3. นำไฟล์ Checkpoint (เช่น `.pt` หรือโฟลเดอร์โมเดล) ไปวางไว้ด้านในโฟลเดอร์ที่สร้างขึ้น

### การใช้งาน
- **บนหน้าเว็บ Gradio:** เมื่อคุณรีเฟรชหน้าเว็บ ระบบจะค้นหาโฟลเดอร์ใน `speakers/` และนำชื่อมาสร้างเป็นเมนู Dropdown ให้เลือกใช้งานได้ทันที (ในโหมด Custom Speaker)
- **ผ่าน API (n8n):** คุณสามารถส่งพารามิเตอร์ `speaker_name="Kru_Satang_Voice"` ผ่าน API ได้เลย ระบบจะสลับไปโหลดไฟล์น้ำหนักของคนนั้นมาสร้างเสียงให้โดยอัตโนมัติ!
