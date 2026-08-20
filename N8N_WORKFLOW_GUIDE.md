# 🤖 n8n Workflow Guide: AI Media Studio 🎬

เอกสารนี้รวบรวม **"สุดยอดเทคนิคและวิธีการวาง Workflow ใน n8n"** เพื่อเชื่อมต่อกับ `AI Media Studio API` ของเราให้เกิดประสิทธิภาพสูงสุด คุณสามารถนำ Use Cases เหล่านี้ไปประยุกต์ใช้เพื่อสร้างสถานีผลิตสื่ออัตโนมัติ (Automated Media Factory) ได้ทันที

---

## 🔑 ข้อมูลพื้นฐานสำหรับ n8n (Global Settings)
ไม่ว่าคุณจะเรียก API ตัวไหน ให้จำการตั้งค่า HTTP Request Node ของ n8n ไว้ดังนี้:
1. **Authentication:** ถ้าเซิร์ฟเวอร์ตั้งค่า `TTS_API_KEY` ไว้ ให้ไปที่แท็บ Headers เพิ่มคีย์ `X-API-Key` และใส่รหัสของคุณ
2. **Body Content Type:** เลือกเป็น `Multipart-Form Data` เสมอ
3. **Response Format:** เลือก `File` (สำหรับการโหลดวิดีโอ/เสียงกลับมา) หรือ `JSON` (ถ้าคุณใช้ฟีเจอร์ `return_local_path=true`)

---

## 🛠️ Use Case 1: ระบบแปลงบทความยาวเป็น Audiobook (Podcast)
**ปัญหาที่แก้:** มีบทความยาวๆ อยากให้อ่านออกเสียงแบบลื่นไหล มีการหยุดพักหายใจ และเสียงมีมิติเหมือนอัดในสตูดิโอ

### การวาง Node ใน n8n:
1. **[Webhook / Schedule]** ➔ เริ่มต้นรับข้อความบทความ หรือดึงจาก RSS Feed / Notion
2. **[LLM Node / OpenAI]** ➔ สั่งให้ AI ตรวจสอบและเกลาข้อความให้เป็นภาษาพูด
3. **[HTTP Request Node] ➔ ยิงไปที่ `/api/tts/generate`**
   - **Method:** `POST`
   - **Parameters (Send Body):**
     - `text`: *(ลากข้อความจาก Node 2 มาใส่)*
     - `engine`: `PiperTTS (Fast CPU / Offline)` *(เพื่อความรวดเร็วและฟรี)*
     - `speed`: `1.0`
     - `apply_breaths`: `true` *(สำคัญมาก! ระบบจะแทรกเสียงสูดลมหายใจระหว่างประโยคให้อัตโนมัติ)*
     - `apply_humanize`: `true` *(ใส่ความสั่นเล็กน้อยให้เสียงไม่เป็นหุ่นยนต์)*
     - `apply_tape_saturation`: `true` *(เพิ่มความหนานุ่มของเสียง)*
     - *(ออปชันเสริม)*: สามารถอัปโหลดไฟล์ `studio_room.wav` ในช่อง `convolution_ir_file` เพื่อจำลองมิติห้องอัดได้
4. **[Google Drive / Telegram]** ➔ นำไฟล์เสียง Binary ที่ได้ไปเซฟเก็บไว้ หรือส่งเข้า Telegram 

---

## 🛠️ Use Case 2: ระบบโคลนเสียงพากย์พร้อมสร้าง Video Reels (100% Auto)
**ปัญหาที่แก้:** ทำคลิปให้ความรู้ลง TikTok/Reels แต่ขี้เกียจพากย์เสียง และขี้เกียจเอาไปนั่งตัดต่อใส่ซับไตเติ้ลทีละคลิป

### การวาง Node ใน n8n:
1. **[Webhook]** ➔ รับข้อมูล (อาจจะเป็นหัวข้อ 1 ประโยค)
2. **[OpenAI Node]** ➔ เขียนสคริปต์สั้นๆ พร้อมสรุปใจความมาให้ 3 ข้อ
3. **[HTTP Request Node - 🗣️ TTS] ➔ โคลนเสียง**
   - **URL:** `/api/tts/generate`
   - **Parameters:**
     - `text`: *(ข้อความสคริปต์จาก Node 2)*
     - `engine`: `CosyVoice 3.0`
     - `mode`: `Zero-Shot (Voice Cloning)`
     - `reference_audio`: *(อัปโหลดไฟล์เสียงต้นแบบของคุณสั้นๆ 5 วินาทีลงไป)*
4. **[HTTP Request Node - 🎬 Video] ➔ ตัดต่อวิดีโอ**
   - **URL:** `/api/video/edit`
   - **Parameters:**
     - `video_local_path`: `/app/temp/media_uploads/bg_gameplay.mp4` *(สมมติว่าคุณมีวิดีโอสต็อกอยู่ในเครื่องเซิร์ฟเวอร์อยู่แล้ว การใช้ Local Path จะทำให้ n8n ทำงานเร็วขึ้น 10 เท่าเพราะไม่ต้องอัปโหลด)*
     - `audio_file`: *(ลากเอาไฟล์ Binary เสียงจาก Node 3 มาใส่)*
     - `short_video_format`: `true` *(ระบบจะครอปวิดีโอตรงกลางให้เป็น 9:16 อัตโนมัติ)*
     - `mute_original_audio`: `true`
     - `text_lines`: *(ลากข้อความสรุป 3 ข้อ จาก Node 2 มาใส่ ระบบจะขึ้นซับไตเติ้ลกลางจอให้บรรทัดละกล่อง)*
     - `watermark_text`: `@SatangTheValue`
5. **[HTTP Request Node / TikTok API]** ➔ โพสต์คลิปที่ตัดต่อเสร็จแล้วลงโซเชียล

---

## 🛠️ Use Case 3: การประมวลผลคลิปสัมภาษณ์ด้วย Studio Mastering (Post-Processing Only)
**ปัญหาที่แก้:** มีคลิปเสียงสัมภาษณ์ที่อัดมาจากมือถือ (เสียงก้อง เสียงซ่า เสียงส.บาดหู) อยากเอามา Mastering ให้กลายเป็น Podcast คุณภาพสูง

*(ระบบของเราไม่ได้มีแค่สร้างเสียง แต่ทำหน้าที่เป็น Audio Engineer ได้ด้วย)*

### การวาง Node ใน n8n:
1. **[Google Drive Trigger]** ➔ ดักจับเมื่อมีคนโยนไฟล์คลิปเสียงเข้ามาในโฟลเดอร์
2. **[HTTP Request Node] ➔ ยิงไปที่ `/api/tts/generate`**
   - *ทริค:* เราจะไม่ส่ง text แต่เราจะใช้พลังของระบบ Mastering ท้ายท่อ
   - ส่ง Parameter ปกติ แต่เน้นไปที่:
     - `apply_deessing`: `true` *(ลดเสียงแหลมเสียดหู)*
     - `apply_tape_saturation`: `true` 
     - *(ถ้าคุณมี Endpoint แยกสำหรับรับเสียงเข้ามารัน Mastering ตรงๆ ให้ใช้ Endpoint นั้น)*
     *(ในอัปเดตอนาคต เราสามารถเพิ่ม Route `/api/audio/mastering` แยกออกมาเพื่อรับไฟล์เสียงมนุษย์เข้ามารันผ่าน Pedalboard ล้วนๆ ได้)*

---

## 🌟 เคล็ดลับระดับ Pro (Best Practices)

### 1. ลดภาระ (Timeout) ของ n8n ด้วย `return_local_path`
ถ้าคุณรัน n8n และ AI Media Studio **บน Docker ในเครื่องเดียวกัน** 
คุณควรตั้งค่าให้ n8n กับ API มองเห็นโฟลเดอร์เดียวกัน (Mount Volume) 
จากนั้นใน n8n เวลาสั่งแก้ไขวิดีโอ ให้ส่ง `return_local_path: true` ไปด้วย
*   **ผลลัพธ์:** API จะไม่ส่งไฟล์วิดีโอ 100MB กลับมาทางสาย LAN ให้เสียเวลา แต่จะตอบแค่ JSON สั้นๆ ว่า `{"status": "success", "output_path": "/shared/video.mp4"}` 
*   **n8n ทำอะไรต่อ:** ใช้ Node **"Read/Write File"** ไปดึงไฟล์จาก Path นั้นขึ้นมาแทน วิธีนี้ n8n ไม่มีทาง Error เรื่อง Memory/Timeout แน่นอนครับ!

### 2. การจัดการ Impulse Responses (IR)
ระบบได้สร้างไฟล์ตัวอย่างจำลองไว้ให้ที่ `assets/impulse_responses/`
คุณสามารถเอาไฟล์เสียง IR จริงๆ (หาโหลดฟรีได้ตามเน็ต เช่น เสียงสะท้อนโบสถ์, เสียงสะท้อนห้องอัดแพงๆ) ไปวางไว้ แล้วสั่งใช้งานผ่าน API เพื่อให้เสียง AI หรือเสียงมนุษย์ของคุณมีมิติตามห้องนั้นๆ ได้เป๊ะ 100% ครับ