# 📝 Changelog (บันทึกการปรับปรุงระบบ)

All notable changes to the **AI Media Studio Pro** project will be documented in this file.

## [Unreleased / Latest] - 2026-08-22
### 🚀 Added
- **Auto Model Provisioning:** ระบบดาวน์โหลด Base Model (Piper TTS: `en_US-lessac-medium` และ `en_US-ryan-medium`) อัตโนมัติ ป้องกันปัญหา No Models Found ตอน Deploy โฮสต์ใหม่
- **Faster-Whisper Engine (`subtitle_engine.py`):** เพิ่มเอนจิ้น CTranslate2 ขนาดจิ๋วแต่ทรงพลัง สำหรับสกัดเสียงเป็นซับไตเติลไฟล์ `.ass` แบบ Word-level Timestamp พร้อมเอฟเฟกต์ Karaoke Style สีเหลืองขอบดำ (สไตล์ TikTok)
- **API Endpoint `quote916`:** สร้าง Template อัตโนมัติสำหรับทำคลิปคำคม/สอนเทคนิค โดยการนำวิดีโอมาขยายฉากหลังเบลอ (Smart Blur) ตีเข้ากรอบแนวตั้ง 9:16 แปะตัวอักษร และลบเสียง
- **API Endpoint `pro_vlog`:** สร้าง Template อัตโนมัติสำหรับทำคลิป Vlog ดึงเสียงไปผ่าน Pedalboard (Mastering) แล้วเบิร์นซับไตเติลเด้งตามคำลงไป
- **Dynamic Path for Fonts:** รองรับการเรียกใช้ฟอนต์ภาษาไทย (`Sarabun-Bold.ttf`) แบบ Dynamic (`BASE_DIR`) เพื่อให้ฟอนต์ไม่เป็นกล่องสี่เหลี่ยม แม้จะย้ายโปรเจกต์ไปรันบนเซิร์ฟเวอร์อื่น

### 🔄 Changed
- **Removed MoviePy:** ถอดการพึ่งพา MoviePy ในการประมวลผลวิดีโอออก 100% 
- **Upgraded to FFmpeg Complex Filter:** เปลี่ยนมาใช้ FFmpeg แบบ Single-pass processing (คำสั่งเดียวม้วนเดียวจบ) ลด I/O และประหยัด RAM เซิร์ฟเวอร์มหาศาล
- **API Documentation (`API_EXAMPLES.md`):** เพิ่มคู่มือพร้อม 5 ตัวอย่าง Use Case จริง สำหรับการทำงานร่วมกับ n8n

### 🛠️ Fixed
- **Font Rendering Bug:** แก้ไขปัญหา FFmpeg หาระบบ Font ภาษาไทยไม่เจอจนแสดงผลเป็นกล่องสี่เหลี่ยม โดยอ้างอิงผ่าน `os.path.join(BASE_DIR)` ทำให้ชัวร์ว่าเจอไฟล์ทุกระบบ OS
- **Video Aspect Ratio Bug:** แก้ไขความเข้าใจผิดเรื่องสัดส่วนภาพจากแนวนอน 16:9 เป็นบังคับแนวตั้ง 9:16 (เหมาะสำหรับ TikTok, Reels, Shorts) เสมอ
