# Tai Chi Master Pro: AI-Powered Biomechanical Analysis
**AI-Based Tai Chi Training & Pose Tracking Web Application**

Tai Chi Master Pro เป็นแอปพลิเคชันเว็บที่พัฒนาขึ้นภายใต้โครงการวิจัยของ **KMCH (King Mongkut's Chaokhun Thahan Hospital)** เพื่อวิเคราะห์พิกัดการเคลื่อนไหวของร่างกาย (Biometrics) โดยใช้เทคโนโลยี Computer Vision ในการตรวจสอบความถูกต้องของการร่ายรำไทเก็กแบบ Real-time

## Features
* **Real-time Hand Tracking**: ใช้ MediaPipe ในการตรวจจับพิกัดข้อมือและมืออย่างแม่นยำ
* **Dynamic Guidelines**: ดึงพิกัดอ้างอิง (Guidelines) จาก Firebase Realtime Database มาแสดงผลเป็นวงกลมไกด์ไลน์บนหน้าจอ
* **Live Scoring System**: คำนวณความแม่นยำ (Accuracy) แยกตามมือซ้ายและขวาตามช่วงเฟรมของท่ารำ
* **Global Leaderboard**: ระบบบันทึกคะแนนและจัดอันดับผู้ใช้งานผ่าน Cloud Database
* **Cyberpunk UI**: อินเทอร์เฟซทันสมัยสไตล์ Modern Neon พร้อมฟอนต์ Orbitron

## Tech Stack
* **Frontend**: Streamlit
* **Computer Vision**: OpenCV, MediaPipe
* **Backend & Database**: Firebase Realtime Database (Region: asia-southeast1)
* **WebRTC**: Streamlit-WebRTC สำหรับการประมวลผลวิดีโอผ่านบราวเซอร์

## Installation & Local Development

1. **Clone the repository**
   ```bash
   git clone [https://github.com/Obanez/TaiChi_Game.git)
   cd taichiweb
2.**Create Virtual Environment**
python -m venv .venv
source .venv/bin/activate  # สำหรับ Windows: .venv\Scripts\activate
3.**Install Dependencies**
pip install -r requirements.txt
4.**Configuration**
สร้างไฟล์ .streamlit/secrets.toml และใส่ค่า Firebase Service Account ของคุณลงไป (ดูตัวอย่างในโค้ด)
5.**Run Application**
streamlit run streamlit_app.py