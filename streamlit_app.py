import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import threading
import time
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth

# --- 1. Firebase Configuration (Region: asia-southeast1) ---
def init_firebase():
    if not firebase_admin._apps:
        try:
            fb_conf = st.secrets["firebase"]
            creds_dict = {
                "type": fb_conf["type"],
                "project_id": fb_conf["project_id"],
                "private_key_id": fb_conf["private_key_id"],
                "private_key": fb_conf["private_key"], # ใช้ค่าจาก Triple Quotes ใน TOML ได้เลย
                "client_email": fb_conf["client_email"],
                "client_id": fb_conf["client_id"],
                "auth_uri": fb_conf["auth_uri"],
                "token_uri": fb_conf["token_uri"],
                "auth_provider_x509_cert_url": fb_conf["auth_provider_x509_cert_url"],
                "client_x509_cert_url": fb_conf["client_x509_cert_url"],
            }
            creds = credentials.Certificate(creds_dict)
            firebase_admin.initialize_app(creds, {
                'databaseURL': 'https://taichigmae-default-rtdb.asia-southeast1.firebasedatabase.app'
            })
        except Exception as e:
            st.error(f"❌ Firebase Init Error: {e}")

init_firebase()

# --- 2. Database Helpers ---
def fetch_guide_line(pose_name):
    """ ดึงพิกัดไกด์ไลน์จาก Firebase """
    ref = db.reference(f'guidelines/{pose_name}')
    data = ref.get()
    if not data:
        return np.array([]), np.array([])
    
    left_hand, right_hand = [], []
    for entry in data:
        hand_type = entry.get("HandType")
        pos = [entry.get("X"), entry.get("Y")]
        if hand_type == "Left":
            left_hand.append(pos)
        else:
            right_hand.append(pos)
            
    return np.array(left_hand, dtype=np.int32), np.array(right_hand, dtype=np.int32)

def save_score_to_firebase(username, score):
    """ บันทึกคะแนนลง Firebase """
    ref = db.reference('scores')
    ref.push({
        'username': username,
        'score': round(score, 2),
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

def get_leaderboard():
    """ ดึง Top 10 Leaderboard """
    ref = db.reference('scores')
    data = ref.get()
    if not data: return []
    
    scores_list = [val for val in data.values()]
    scores_list.sort(key=lambda x: x['score'], reverse=True)
    return scores_list[:10]

# --- 3. MediaPipe & Video Processor ---
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
@st.cache_data
def get_ice_servers():
    try:
        account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
        auth_token = st.secrets["TWILIO_AUTH_TOKEN"]
        
        # ยิงคำขอไปขอช่องทางพิเศษจาก Twilio
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Tokens.json",
            auth=HTTPBasicAuth(account_sid, auth_token)
        )
        response.raise_for_status()
        return response.json()["ice_servers"]
    except Exception as e:
        st.warning(f"⚠️ Twilio API Error: {e}")
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

class TaiChiVideoProcessor(VideoProcessorBase):
    def __init__(self, guide_line_left, guide_line_right):
        self.guide_line_left = guide_line_right
        self.guide_line_right = guide_line_left
        self.hands = mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.lock = threading.Lock()
        self.accuracy = {"Left": 0, "Right": 0}
        self.frames_in_circle_left = 0
        self.frames_in_circle_right = 0
        self.total_frames = 0
        self.frame_index = 0
        self.game_finished = False
        self.max_frames = 3200 # ความยาวเกมมาตรฐาน

    def is_inside_circle(self, hand_pos, guide_pos, radius):
        return np.linalg.norm(np.array(hand_pos) - np.array(guide_pos)) < radius

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        game_width, game_height = 960, 720
        img = cv2.resize(img, (game_width, game_height))
        img = cv2.flip(img, 1)

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        with self.lock:
            if not self.game_finished:
                self.total_frames += 1
                
                if results.multi_hand_landmarks:
                    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        hand_x = int(hand_landmarks.landmark[0].x * game_width)
                        hand_y = int(hand_landmarks.landmark[0].y * game_height)
                        hand_pos = (hand_x, hand_y)
                        
                        hand_label = results.multi_handedness[idx].classification[0].label
                        guide_line = self.guide_line_right if hand_label == "Right" else self.guide_line_left

                        if guide_line.size > 0:
                            position = guide_line[self.frame_index % len(guide_line)]
                            
                            is_in = self.is_inside_circle(hand_pos, position, 40)
                            circle_color = (0, 255, 0) if is_in else (0, 0, 255)
                            
                            cv2.circle(img, (int(position[0]), int(position[1])), 40, circle_color, 2)
                            cv2.circle(img, hand_pos, 10, (255, 0, 0), -1)

                            if is_in:
                                if hand_label == "Right":
                                    self.frames_in_circle_right += 1
                                else:
                                    self.frames_in_circle_left += 1

                    self.frame_index += 1

                # คำนวณ Accuracy
                if self.total_frames > 0:
                    self.accuracy["Left"] = (self.frames_in_circle_left / self.total_frames) * 100
                    self.accuracy["Right"] = (self.frames_in_circle_right / self.total_frames) * 100

                if self.total_frames >= self.max_frames:
                    self.game_finished = True

        return frame.from_ndarray(img, format="bgr24")

# --- 4. Streamlit UI Styling ---
st.set_page_config(page_title="TAI CHI CYBERPUNK", layout="wide")

# Custom CSS for Cyberpunk Theme
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    
    <style>
    /* Global Background and Typography */
    .stApp {
        background-color: #1a1a2e;
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
    }
    
    h1, h2, h3, .stHeader, .stMetric label, button {
        font-family: 'Orbitron', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    h1 {
        color: #00bcd4;
        text-shadow: 0 0 10px #00bcd4, 0 0 20px #00bcd4;
        text-align: center;
        padding-bottom: 30px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #16213e !important;
        border-right: 2px solid #00bcd4;
    }
    
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #00bcd4;
        font-size: 1.2rem;
        border-bottom: 1px solid #00bcd4;
        padding-bottom: 10px;
    }

    /* Video Frame */
    .video-container {
        border: 4px solid #00bcd4;
        border-radius: 15px;
        box-shadow: 0 0 20px #00bcd4, inset 0 0 10px #00bcd4;
        overflow: hidden;
        margin: auto;
    }

    /* Metric Cards (Glassmorphism) */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 188, 212, 0.3);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
        transition: 0.3s;
    }
    
    .metric-card:hover {
        border-color: #00bcd4;
        box-shadow: 0 0 15px rgba(0, 188, 212, 0.5);
        transform: translateY(-5px);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00bcd4;
        text-shadow: 0 0 5px #00bcd4;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        margin-bottom: 5px;
    }

    /* Leaderboard Table */
    .leaderboard-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
        background: rgba(0, 0, 0, 0.2);
    }
    
    .leaderboard-table th {
        background-color: rgba(0, 188, 212, 0.2);
        color: #00bcd4;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #00bcd4;
    }
    
    .leaderboard-table td {
        padding: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .leaderboard-table tr:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.02);
    }
    
    .leaderboard-table tr:hover {
        background-color: rgba(0, 188, 212, 0.1);
        cursor: default;
    }

    /* Buttons */
    .stButton>button {
        background: transparent !important;
        color: #00bcd4 !important;
        border: 2px solid #00bcd4 !important;
        border-radius: 5px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: #00bcd4 !important;
        color: #1a1a2e !important;
        box-shadow: 0 0 15px #00bcd4 !important;
        transform: translateY(-3px) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #ff8c00;
        font-size: 0.7rem;
        margin-top: 50px;
        text-shadow: 0 0 5px #ff8c00;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Tai Chi Game</h1>", unsafe_allow_html=True)

# --- Sidebar Content ---
with st.sidebar:
    st.markdown("<h2>Player Settings</h2>", unsafe_allow_html=True)
    username = st.text_input("Access Identity", value="KMCH_Player")
    
    st.markdown("<br><h2>Pro Dashboard</h2>", unsafe_allow_html=True)
    leaderboard_data = get_leaderboard()
    
    if leaderboard_data:
        table_html = """
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>User</th>
                    <th>Acc</th>
                </tr>
            </thead>
            <tbody>
        """
        for i, entry in enumerate(leaderboard_data):
            table_html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{entry['username']}</td>
                    <td style="color:#00bcd4;">{entry['score']:.1f}%</td>
                </tr>
            """
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
    else:
        st.write("No data in matrix.")

    st.markdown('<div class="footer">KMCH Tai Chi By Oat</div>', unsafe_allow_html=True)

# --- Main Layout ---
col_vid, col_stats = st.columns([3, 1])

with col_stats:
    # Placeholders for Streamlit updates with initial styled containers
    l_acc_p = st.empty()
    r_acc_p = st.empty()
    t_acc_p = st.empty()
    
    l_acc_p.markdown("""<div class="metric-card"><div class="metric-label">L-HAND SYNC</div><div class="metric-value">0.00%</div></div>""", unsafe_allow_html=True)
    r_acc_p.markdown("""<div class="metric-card"><div class="metric-label">R-HAND SYNC</div><div class="metric-value">0.00%</div></div>""", unsafe_allow_html=True)
    t_acc_p.markdown("""<div class="metric-card" style="border-color: #00bcd4;"><div class="metric-label">TOTAL ACCURACY</div><div class="metric-value">0.00%</div></div>""", unsafe_allow_html=True)
    
    save_p = st.empty()

with col_vid:
    pose_name = "TaiChiMaster720p"
    g_left, g_right = fetch_guide_line(pose_name)

    if g_left.size == 0:
        st.warning("⚠️ SYNC DATA MISSING. CHECK FIREBASE UPLOAD.")

    st.markdown('<div class="video-container">', unsafe_allow_html=True)
    ctx = webrtc_streamer(
            key="taichi-cyber",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: TaiChiVideoProcessor(g_left, g_right),
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )
    st.markdown('</div>', unsafe_allow_html=True)



# --- UI Update Logic ---
if ctx.video_processor:
    while ctx.state.playing:
        with ctx.video_processor.lock:
            left_acc = ctx.video_processor.accuracy["Left"]
            right_acc = ctx.video_processor.accuracy["Right"]
            finished = ctx.video_processor.game_finished
            total_score = (left_acc + right_acc) / 2
            
        # Update metrics using markdown to keep styling consistent
        l_acc_p.markdown(f"""<div class="metric-card"><div class="metric-label">L-HAND SYNC</div><div class="metric-value">{left_acc:.2f}%</div></div>""", unsafe_allow_html=True)
        r_acc_p.markdown(f"""<div class="metric-card"><div class="metric-label">R-HAND SYNC</div><div class="metric-value">{right_acc:.2f}%</div></div>""", unsafe_allow_html=True)
        t_acc_p.markdown(f"""<div class="metric-card" style="border-color:#00bcd4; box-shadow: 0 0 15px #00bcd4;"><div class="metric-label">TOTAL ACCURACY</div><div class="metric-value">{total_score:.2f}%</div></div>""", unsafe_allow_html=True)
        
        if finished:
            st.balloons()
            if save_p.button("UPLOAD DATA TO CLOUD"):
                save_score_to_firebase(username, total_score)
                st.success("DATA UPLOADED SUCCESSFULLY.")
                time.sleep(2)
                st.rerun()
            break
        time.sleep(0.5)
