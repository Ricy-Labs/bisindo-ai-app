import streamlit as st
import os as _os
from groq import Groq
import datetime
import av
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os
import json
import pickle
import numpy as np
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ===== KONFIGURASI =====
try:
    API_KEY = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
except:
    API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=API_KEY)

SYSTEM_PROMPT = """Kamu adalah BISINDO Bot, asisten AI ahli Bahasa Isyarat Indonesia (BISINDO).
Sistem BISINDO AI ini dibuat oleh Ricy Ikada Saputra dari Universitas Mercu Buana, jurusan Teknik Informatika 2026.
Sistem ini dapat mendeteksi alfabet A-Z dalam BISINDO secara real-time menggunakan:
- MediaPipe Hand Landmarker untuk deteksi 21 titik landmark PER TANGAN (2 tangan sekaligus)
- Random Forest Classifier dengan 7.600+ data rekaman gesture 2 tangan
- EMA Smoothing, Majority Voting, dan Confidence Threshold untuk stabilisasi deteksi
- AI Agent berbasis LLaMA 3.3 70B via Groq untuk menyusun kalimat dari rangkaian huruf
- Overlay subtitle real-time (Tkinter) yang tampil di atas PPT/PDF/Word saat presentasi
BISINDO (Bahasa Isyarat Indonesia) adalah bahasa isyarat yang digunakan oleh komunitas Tuli di Indonesia.
Sistem ini KHUSUS dirancang untuk membantu guru/dosen tunarungu agar dapat mengajar dengan lebih inklusif.
Setiap huruf A-Z dideteksi menggunakan KEDUA TANGAN secara bersamaan.
Jawab dalam Bahasa Indonesia, ramah, informatif dan singkat.
Jika ditanya tentang gesture spesifik, jelaskan cara membentuk kedua tangan untuk huruf tersebut."""

# ===== AGENTIC AI FUNCTIONS =====
def agent_build_sentence(raw_text, context=None):
    if context is None:
        context = []
    context_str = ""
    if context:
        context_str = "Konteks percakapan sebelumnya:\n" + "\n".join(context[-3:]) + "\n\n"
    prompt = f"""{context_str}Teks mentah dari gesture BISINDO: "{raw_text}"

Kamu adalah agen komunikasi inklusif untuk pengguna tunarungu.
Lakukan 3 tugas berikut dan balas HANYA dalam format JSON:
{{
  "kalimat": "kalimat yang sudah diperbaiki dan dirapikan",
  "prediksi": ["kata1", "kata2", "kata3"],
  "konteks": "penjelasan singkat maksud kalimat ini"
}}
Rules:
- kalimat: perbaiki ejaan, buat kalimat Indonesia yang natural
- prediksi: 3 kata yang paling mungkin menjadi kata selanjutnya
- konteks: jelaskan maksud kalimat secara singkat
- Balas HANYA JSON, tidak ada teks lain"""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Kamu adalah agen komunikasi inklusif BISINDO. Selalu balas dalam format JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300, temperature=0.3
        )
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except:
        return {"kalimat": raw_text, "prediksi": [], "konteks": ""}

def agent_refine_sentence(sentence):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Ubah kalimat menjadi versi yang lebih sopan dan natural dalam Bahasa Indonesia. Balas HANYA kalimatnya."},
                {"role": "user", "content": sentence}
            ],
            max_tokens=150, temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except:
        return sentence

# ===== GESTURE INFO =====
ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# ══ GESTURE_DESC — BISINDO 2 Tangan (sesuai sistem rekaman) ══
# Deskripsi berdasarkan gesture yang direkam dengan 2 tangan
GESTURE_DESC = {
    "A": "Kedua tangan mengepal dengan ibu jari ke atas, saling bersentuhan di bagian ibu jari",
    "B": "Tangan kanan: empat jari rapat lurus ke atas, ibu jari ditekuk ke dalam · Tangan kiri: mengepal di samping kanan",
    "C": "Tangan kanan: semua jari melengkung membentuk huruf C (satu tangan)",
    "D": "Tangan kanan: telunjuk lurus ke atas, ibu jari & jari tengah menyentuh · Tangan kiri: mengepal di samping",
    "E": "Kedua tangan: semua jari lurus rapat ke atas, telapak menghadap depan, berdampingan",
    "F": "Tangan kanan: telunjuk ditekuk menyentuh ibu jari, jari lain lurus · Tangan kiri: mengepal di samping",
    "G": "Kedua tangan mengepal, ditumpuk vertikal — tangan kiri di bawah, tangan kanan di atas",
    "H": "Kedua tangan: telunjuk & jari tengah lurus ke atas berdampingan, saling bersentuhan",
    "I": "Tangan kanan: kelingking lurus ke atas, jari lain mengepal · Tangan kiri: mengepal di samping (gerakan ke atas)",
    "J": "Tangan kanan: kelingking lurus membuat gerakan melingkar seperti huruf J · Tangan kiri: mengepal (ada gerakan)",
    "K": "Tangan kanan: telunjuk & jari tengah membentuk V ke atas, ibu jari menyentuh jari tengah · Tangan kiri: mengepal",
    "L": "Kedua tangan: ibu jari & telunjuk membentuk L, saling berhadapan membentuk bingkai",
    "M": "Kedua tangan terbuka lebar, semua jari merenggang, telapak menghadap depan (seperti huruf M)",
    "N": "Kedua tangan terbuka, semua jari merenggang ke atas, telapak menghadap depan berdampingan",
    "O": "Tangan kanan: semua jari & ibu jari membentuk lingkaran O (satu tangan)",
    "P": "Tangan kanan: ibu jari & telunjuk membentuk lingkaran menghadap bawah · Tangan kiri: mengepal di bawah",
    "Q": "Tangan kanan: telunjuk & ibu jari membentuk lingkaran, jari lain mengepal, mengarah ke bawah · Tangan kiri: mengepal",
    "R": "Tangan kanan: telunjuk ditekuk membentuk kait (ada gerakan) · Tangan kiri: mengepal di bawah",
    "S": "Kedua tangan: jari-jari ditekuk seperti cakar, saling berhadapan & bergerak",
    "T": "Tangan kanan: telunjuk horizontal, ibu jari vertikal membentuk T · Tangan kiri: mengepal di bawah",
    "U": "Kedua tangan: telapak terbuka menghadap atas, berdampingan sejajar (seperti mangkuk)",
    "V": "Kedua tangan: telunjuk & jari tengah membentuk V, saling berhadapan",
    "W": "Kedua tangan: telunjuk, jari tengah & manis lurus ke atas, saling berdampingan membentuk W",
    "X": "Kedua tangan: telunjuk disilangkan membentuk X, saling berhadapan",
    "Y": "Kedua tangan: telunjuk & jari tengah membentuk V ke atas, saling berhadapan",
    "Z": "Tangan kanan: telunjuk membuat gerakan zigzag seperti huruf Z (satu tangan, ada gerakan)",
}

GESTURE_COLORS = {
    c: (
        int(255 * abs(np.sin(i * 0.5))),
        int(200 * abs(np.cos(i * 0.4))),
        int(255 * abs(np.sin(i * 0.3 + 1)))
    )
    for i, c in enumerate(ALPHABET)
}

DATA_FILE = "gesture_data.json"

def load_gesture_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                for g in ALPHABET:
                    if g not in data["counts"]:
                        data["counts"][g] = 0
                return data
    except:
        pass
    return {"counts": {g: 0 for g in ALPHABET}, "total": 0}

def save_gesture_data(counts, total):
    try:
        with open(DATA_FILE, "w") as f:
            json.dump({"counts": counts, "total": total}, f)
    except:
        pass

# ===== LOAD CUSTOM MODEL =====
@st.cache_resource
def load_model():
    try:
        with open("bisindo_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("bisindo_classes.pkl", "rb") as f:
            classes = pickle.load(f)
        # Pastikan classes adalah list of str
        classes = [str(c) for c in classes]
        return model, classes
    except Exception as e:
        return None, None

# ===== DOWNLOAD MEDIAPIPE LANDMARKER =====
lm_model_path = "hand_landmarker.task"
if not os.path.exists(lm_model_path):
    with st.spinner("⬇️ Downloading hand landmarker model..."):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            lm_model_path
        )

# ===================================================
# ✅ FUNGSI NORMALISASI LANDMARK (Fix #4)
# Koordinat relatif ke wrist, scale relatif ke hand size
# Membuat prediksi tidak terpengaruh posisi/jarak kamera
# ===================================================
def normalize_landmarks(hand_landmarks):
    """
    Normalisasi landmark tangan:
    - Titik asal (0,0,0) = wrist (landmark 0)
    - Scale = jarak wrist ke middle finger MCP (landmark 9)
    Hasil: invariant terhadap posisi & jarak tangan ke kamera
    """
    if not hand_landmarks:
        return [0.0] * 63

    wrist = hand_landmarks[0]
    ref = hand_landmarks[9]  # middle finger MCP

    # Hitung scale (jarak wrist ke ref point)
    scale = np.sqrt(
        (ref.x - wrist.x) ** 2 +
        (ref.y - wrist.y) ** 2 +
        (ref.z - wrist.z) ** 2
    )
    if scale < 1e-6:
        scale = 1.0  # hindari division by zero

    normalized = []
    for lm in hand_landmarks:
        normalized.extend([
            (lm.x - wrist.x) / scale,
            (lm.y - wrist.y) / scale,
            (lm.z - wrist.z) / scale,
        ])
    return normalized


# ===================================================
# ✅ VIDEO PROCESSOR DENGAN SEMUA FIX TERINTEGRASI
# ===================================================
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.gesture = ""
        self.stable_gesture = ""
        self.hand_count = 0
        self.confidence = 0.0
        self.stable_confidence = 0.0
        self.stable_progress = 0.0
        self._last_gesture = ""
        self._frame_count = 0
        _loaded = load_model()
        self.model   = _loaded[0]
        self.classes = _loaded[1]
        # Load scaler kalau ada
        self._scaler = None
        try:
            import os as _os
            if _os.path.exists('bisindo_scaler.pkl'):
                with open('bisindo_scaler.pkl','rb') as _f:
                    self._scaler = pickle.load(_f)
        except:
            pass

        # ── Fix #1: EMA Smoothing ──────────────────────────
        # Alpha kecil = lebih smooth tapi sedikit lag
        # Alpha besar = lebih responsif tapi lebih noise
        self._ema_landmarks = None
        self._EMA_ALPHA = 0.5

        # ── Fix #2: Confidence Threshold ──────────────────
        # Prediksi di bawah threshold langsung dibuang
        self._CONF_THRESHOLD = 0.75

        # ── Fix #3: Majority Voting Buffer ────────────────
        # Gesture harus menang vote 70% dari N frame terakhir
        self._vote_buffer = []
        self._VOTE_SIZE = 8    # 8 frame @ 30fps ≈ 0.27 detik
        self._VOTE_MIN_RATIO = 0.65

        # Untuk stable gesture (sudah di-vote)
        self._last_stable = ""
        self._hand_label_cache = {}  # Cache handedness

        # MediaPipe detector
        base_options = python.BaseOptions(model_asset_path=lm_model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.6,   # ✅ naikkan dari default 0.5
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        # Sharpening ringan
        kernel = np.array([[0, -0.5, 0],
                           [-0.5, 3, -0.5],
                           [0, -0.5, 0]])
        img = cv2.filter2D(img, -1, kernel)

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self._frame_count += 1

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.detector.detect(mp_image)
            self.hand_count = len(result.hand_landmarks)

            if result.hand_landmarks:
                # Gambar skeleton semua tangan
                conns = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17)
                ]
                colors_hand = [(200, 200, 255), (255, 200, 100)]
                for idx, hand in enumerate(result.hand_landmarks):
                    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
                    color = colors_hand[idx % 2]
                    for a, b in conns:
                        cv2.line(img, points[a], points[b], color, 1)
                    for pt in points:
                        cv2.circle(img, pt, 4, (0, 255, 150), -1)

                if self.model is not None:
                    # ── Sort by wrist X → stabil saat 2 tangan nempel ──────────
                    left_lm  = [0.0] * 63
                    right_lm = [0.0] * 63
                    hands_with_x = [(hand[0].x, i, hand)
                                    for i, hand in enumerate(result.hand_landmarks)]
                    hands_with_x.sort(key=lambda t: t[0])

                    for rank, (wx, i, hand) in enumerate(hands_with_x):
                        norm = normalize_landmarks(hand)
                        if len(hands_with_x) == 2:
                            sort_label = "Left" if rank == 0 else "Right"
                        else:
                            cached   = self._hand_label_cache.get(i)
                            mp_label = "Right"
                            if result.handedness and i < len(result.handedness):
                                mp_label = result.handedness[i][0].category_name
                            sort_label = cached if cached else mp_label
                        self._hand_label_cache[i] = sort_label
                        if sort_label == "Left":
                            right_lm = norm
                        else:
                            left_lm = norm

                    raw_row = np.array(left_lm + right_lm, dtype=np.float32)

                    # ✅ Fix #1: EMA Smoothing
                    if self._ema_landmarks is None:
                        self._ema_landmarks = raw_row.copy()
                    else:
                        self._ema_landmarks = (
                            self._EMA_ALPHA * raw_row +
                            (1 - self._EMA_ALPHA) * self._ema_landmarks
                        )

                    X = self._ema_landmarks.reshape(1, -1)
                    if self._scaler is not None:
                        X = self._scaler.transform(X)
                    pred_idx = self.model.predict(X)[0]
                    proba    = self.model.predict_proba(X)[0]
                    conf     = float(np.max(proba))
                    # Konversi index → huruf A-Z
                    if hasattr(pred_idx, 'item'):
                        pred_idx = pred_idx.item()
                    if (isinstance(pred_idx, (int, float)) and
                            self.classes is not None and
                            int(pred_idx) < len(self.classes)):
                        pred = str(self.classes[int(pred_idx)])
                    else:
                        pred = str(pred_idx)

                    # ✅ Fix #2: Confidence Threshold
                    if conf < self._CONF_THRESHOLD:
                        # Gesture terlalu ragu-ragu, skip frame ini
                        self.gesture = ""
                        self.confidence = 0.0
                        self._vote_buffer.clear()
                        self.stable_progress = 0.0
                    else:
                        self.gesture = pred
                        self.confidence = conf

                        # ✅ Fix #3: Majority Voting Buffer
                        self._vote_buffer.append(pred)
                        if len(self._vote_buffer) > self._VOTE_SIZE:
                            self._vote_buffer.pop(0)

                        # Hitung progress dari vote terbanyak
                        if self._vote_buffer:
                            winner, count = Counter(self._vote_buffer).most_common(1)[0]
                            vote_ratio = count / len(self._vote_buffer)
                            self.stable_progress = min(
                                len(self._vote_buffer) / self._VOTE_SIZE,
                                vote_ratio
                            )

                            # Reset stable saat gesture berubah → bisa input ulang
                            if winner != pred:
                                self.stable_gesture = ""

                            # Gesture stabil kalau sudah full buffer & ratio cukup
                            if (len(self._vote_buffer) == self._VOTE_SIZE and
                                    vote_ratio >= self._VOTE_MIN_RATIO and
                                    winner == pred):
                                self.stable_gesture = winner
                                self.stable_confidence = conf
                            else:
                                if count < self._VOTE_SIZE * self._VOTE_MIN_RATIO:
                                    self.stable_gesture = ""

                        # Catat statistik saat gesture berubah
                        if pred != self._last_gesture:
                            data = load_gesture_data()
                            counts = data["counts"]
                            total = data["total"]
                            if pred in counts:
                                counts[pred] += 1
                                total += 1
                                save_gesture_data(counts, total)
                            self._last_gesture = pred

            else:
                # Tidak ada tangan — reset semua state
                self.gesture = ""
                self.stable_gesture = ""
                self.confidence = 0.0
                self.stable_progress = 0.0
                self._vote_buffer.clear()
                self._last_gesture = ""
                self._ema_landmarks    = None
                self._hand_label_cache = {}  # reset cache saat tangan hilang

        except Exception as e:
            import traceback
            traceback.print_exc()

        # ===== OVERLAY UI =====
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (8, 8, 18), -1)
        cv2.rectangle(overlay, (0, h - 95), (w, h), (8, 8, 18), -1)
        img = cv2.addWeighted(overlay, 0.8, img, 0.2, 0)

        cv2.putText(img, "BISINDO AI", (15, 32), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, "A-Z Sign Language Detector", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 160), 1)
        cv2.putText(img, "by Second Byte", (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 100), 1)

        dot_color = (0, 255, 100) if self.hand_count > 0 else (60, 60, 80)
        cv2.circle(img, (w - 185, 35), 7, dot_color, -1)
        cv2.putText(img, f"Tangan: {self.hand_count}", (w - 170, 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, dot_color, 2)
        cv2.line(img, (0, 85), (w, 85), (30, 30, 50), 1)
        cv2.line(img, (0, h - 95), (w, h - 95), (30, 30, 50), 1)

        if self.gesture:
            color = GESTURE_COLORS.get(self.gesture, (180, 180, 180))
            cv2.putText(img, self.gesture, (w // 2 - 30, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 0), 6)
            cv2.putText(img, self.gesture, (w // 2 - 30, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 2.5, color, 3)
            desc = GESTURE_DESC.get(self.gesture, "")
            cv2.putText(img, desc[:35], (10, h - 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 180), 1)

            # Confidence bar
            bar_w = 220
            bar_x = (w - bar_w) // 2
            bar_y = h - 78
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6), (40, 40, 40), -1)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * self.confidence), bar_y + 6), color, -1)
            cv2.putText(img, f"{int(self.confidence * 100)}%",
                        (bar_x + bar_w + 8, bar_y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

            # ✅ Vote progress indicator (di atas confidence bar)
            vote_bar_y = bar_y - 14
            cv2.putText(img, "Stabilitas:", (bar_x, vote_bar_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 100, 130), 1)
            cv2.rectangle(img, (bar_x + 65, vote_bar_y - 7),
                          (bar_x + bar_w, vote_bar_y + 1), (30, 30, 30), -1)
            stab_color = (0, 255, 100) if self.stable_progress >= 1.0 else (0, 180, 255)
            cv2.rectangle(img, (bar_x + 65, vote_bar_y - 7),
                          (bar_x + 65 + int((bar_w - 65) * self.stable_progress), vote_bar_y + 1),
                          stab_color, -1)
        else:
            idle = "Tunjukkan tangan ke kamera..."
            ts = cv2.getTextSize(idle, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(img, idle, ((w - ts[0]) // 2, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 90), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="BISINDO AI",
    page_icon="👐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══ CSS VARIABLES — Single source of truth ══ */
:root {
    /* Base surfaces */
    --bg-base:       #070B18;
    --bg-surface:    #0D1426;
    --bg-overlay:    #121C34;
    --bg-hover:      #172240;

    /* Primary accent — Emerald (action, positive) */
    --emerald-400:   #34D39A;
    --emerald-500:   #00D68F;
    --emerald-600:   #00B87A;
    --emerald-glow:  rgba(0, 214, 143, 0.15);
    --emerald-border:rgba(0, 214, 143, 0.25);

    /* Secondary accent — Cyan (info, tech) */
    --cyan-400:      #22D3EE;
    --cyan-500:      #00B4D8;
    --cyan-glow:     rgba(0, 180, 216, 0.12);

    /* Tertiary — Violet (decorative, premium) */
    --violet-400:    #A78BFA;
    --violet-500:    #7C3AED;
    --violet-glow:   rgba(124, 58, 237, 0.12);

    /* Warm accents */
    --amber-400:     #FCD34D;
    --amber-500:     #F59E0B;
    --coral-500:     #F97316;

    /* Text hierarchy */
    --text-primary:  #EEF2FF;
    --text-secondary:#8FA3BF;
    --text-muted:    #3D5A80;

    /* Borders */
    --border:        rgba(255, 255, 255, 0.07);
    --border-active: rgba(255, 255, 255, 0.14);
    --border-accent: rgba(0, 214, 143, 0.3);

    /* Typography */
    --font-display:  'Syne', sans-serif;
    --font-body:     'Plus Jakarta Sans', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;

    /* Spacing */
    --radius-sm:     8px;
    --radius-md:     14px;
    --radius-lg:     20px;
    --radius-xl:     28px;

    /* Shadows */
    --shadow-sm:     0 2px 8px rgba(0,0,0,0.4);
    --shadow-md:     0 8px 24px rgba(0,0,0,0.5);
    --shadow-lg:     0 20px 60px rgba(0,0,0,0.6);
    --shadow-glow:   0 0 40px rgba(0,214,143,0.12);
}

/* ══ HIDE STREAMLIT DEFAULT UI ══ */
#MainMenu {visibility: hidden !important;}
header[data-testid="stHeader"] {background: transparent !important; height: 0 !important;}
footer {visibility: hidden !important;}
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}

/* ══ FIX CHAT INPUT ══ */
[data-testid="stChatInput"] {
    background: var(--bg-surface) !important;
}
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    color: #0a0f1e !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6b7280 !important;
}
[data-testid="stChatInputContainer"] {
    background: var(--bg-base) !important;
}
[data-testid="stBottom"] {
    background: #0a0f1e !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    padding-bottom: 8px !important;
}
[data-testid="stBottom"] > div {
    background: #0a0f1e !important;
}
[data-testid="stChatInputContainer"] > div {
    background: #0a0f1e !important;
}
section[data-testid="stBottom"] {
    background: #0a0f1e !important;
}

/* ══ GLOBAL RESET & BASE ══ */
html, body, [class*="css"] {
    font-family: var(--font-body);
    font-size: 15px;
    line-height: 1.65;
    color: var(--text-primary);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ══ APP BACKGROUND — Layered depth ══ */
.stApp {
    background-color: var(--bg-base);
    background-image:
        /* Subtle noise texture via SVG */
        url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E"),
        /* Top-left emerald glow */
        radial-gradient(ellipse 70% 50% at 0% 0%, rgba(0,214,143,0.06) 0%, transparent 65%),
        /* Top-right cyan glow */
        radial-gradient(ellipse 50% 40% at 100% 5%, rgba(0,180,216,0.05) 0%, transparent 60%),
        /* Bottom violet glow */
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(124,58,237,0.04) 0%, transparent 55%),
        /* Center dark base */
        radial-gradient(ellipse 100% 80% at 50% 50%, #0A1020 0%, var(--bg-base) 100%);
}

/* ══ SIDEBAR ══ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #080E1C 100%) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px);
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--emerald-500), transparent);
    opacity: 0.5;
}

/* ══ SCROLLBAR ══ */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--emerald-500), var(--cyan-500));
    border-radius: 99px;
}

/* ══ FOCUS — Accessibility ══ */
*:focus-visible {
    outline: 2px solid var(--emerald-500) !important;
    outline-offset: 3px !important;
    border-radius: var(--radius-sm) !important;
}

/* ══ BRAND ══ */
.brand {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 30px;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--emerald-400) 0%, var(--cyan-400) 50%, var(--violet-400) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}

.brand-sub {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
    opacity: 0.8;
}

/* ══ PAGE TITLES ══ */
.page-title {
    font-family: var(--font-display);
    font-weight: 800;
    font-size: 42px;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 6px;
    background: linear-gradient(135deg, #FFFFFF 0%, rgba(255,255,255,0.65) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.page-sub {
    font-size: 14px;
    color: var(--text-secondary);
    margin-bottom: 32px;
    font-weight: 400;
    letter-spacing: 0.1px;
}

/* ══ METRIC CARDS ══ */
.metric-card {
    position: relative;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 22px;
    text-align: center;
    overflow: hidden;
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-active), transparent);
}

.metric-card::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: var(--radius-lg);
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(255,255,255,0.03) 0%, transparent 70%);
    pointer-events: none;
}

.metric-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-active);
    box-shadow: var(--shadow-md);
}

.metric-num {
    font-family: var(--font-display);
    font-size: 46px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 8px;
    letter-spacing: -1px;
}

.metric-lbl {
    font-family: var(--font-mono);
    font-size: 9px;
    color: var(--text-muted);
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ══ SECTION HEADINGS ══ */
.section-head {
    font-family: var(--font-mono);
    font-weight: 500;
    font-size: 11px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 2.5px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 18px;
}

/* ══ RANK CARDS ══ */
.rank-card {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 12px 16px;
    margin-bottom: 6px;
    transition: all 0.15s ease;
}

.rank-card:hover {
    background: var(--bg-overlay);
    border-color: var(--border-active);
    transform: translateX(2px);
}

/* ══ ALPHABET CARDS ══ */
.alpha-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 20px 14px;
    text-align: center;
    margin-bottom: 12px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

.alpha-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 50% 0%, rgba(0,214,143,0.05) 0%, transparent 70%);
    opacity: 0;
    transition: opacity 0.2s;
}

.alpha-card:hover {
    border-color: var(--emerald-border);
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4), var(--shadow-glow);
}

.alpha-card:hover::before { opacity: 1; }

.alpha-letter {
    font-family: var(--font-display);
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 8px;
}

.alpha-desc {
    font-size: 11px;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* ══ CHAT BUBBLES ══ */
.chat-bubble-wrap {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.chat-bubble-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--emerald-500), var(--cyan-500), var(--violet-400));
}

/* ══ WORD BUILDER ══ */
.word-builder {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}

.word-builder::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse 60% 40% at 50% 0%, rgba(0,214,143,0.04) 0%, transparent 70%);
}

.word-display {
    font-family: var(--font-display);
    font-size: 52px;
    font-weight: 800;
    letter-spacing: 10px;
    background: linear-gradient(135deg, var(--emerald-400) 0%, var(--cyan-400) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    min-height: 68px;
    text-align: center;
    padding: 12px 0;
    line-height: 1.1;
}

/* ══ MODEL BADGE ══ */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--emerald-glow);
    border: 1px solid var(--emerald-border);
    border-radius: 99px;
    padding: 6px 14px;
    font-size: 12px;
    font-family: var(--font-mono);
    color: var(--emerald-400);
    font-weight: 500;
    letter-spacing: 0.5px;
}

/* ══ BUTTONS ══ */
.stButton > button {
    background: var(--bg-overlay) !important;
    border: 1px solid var(--border-active) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important;
    font-family: var(--font-body) !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 18px !important;
    min-height: 44px !important;
    transition: all 0.18s ease !important;
    letter-spacing: 0.2px !important;
    position: relative !important;
    overflow: hidden !important;
}

.stButton > button:hover {
    background: var(--emerald-glow) !important;
    border-color: var(--emerald-border) !important;
    color: var(--emerald-400) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4), 0 0 20px rgba(0,214,143,0.1) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Primary button (type=primary) */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--emerald-600), var(--emerald-500)) !important;
    border-color: var(--emerald-500) !important;
    color: #000 !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 20px rgba(0,214,143,0.25) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--emerald-500), var(--emerald-400)) !important;
    box-shadow: 0 8px 30px rgba(0,214,143,0.35) !important;
    color: #000 !important;
}

/* ══ TEXT INPUT ══ */
.stTextInput > div > div > input {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-active) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 15px !important;
    padding: 10px 14px !important;
    min-height: 46px !important;
    transition: border-color 0.15s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--emerald-500) !important;
    box-shadow: 0 0 0 3px rgba(0,214,143,0.1) !important;
}

.stTextInput > div > div > input::placeholder {
    color: var(--text-muted) !important;
}

/* ══ CHAT INPUT ══ */
.stChatInput > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-active) !important;
    border-radius: var(--radius-lg) !important;
}

/* ══ CHAT MESSAGES ══ */
.stChatMessage {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 16px 20px !important;
}

/* ══ RADIO (sidebar nav) ══ */
.stRadio > div {
    gap: 4px !important;
}

.stRadio > div > label {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-md) !important;
    padding: 10px 14px !important;
    transition: all 0.15s ease !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}

.stRadio > div > label:hover {
    background: var(--bg-overlay) !important;
    border-color: var(--border) !important;
}

.stRadio > div > label[data-checked="true"] {
    background: var(--emerald-glow) !important;
    border-color: var(--emerald-border) !important;
    color: var(--emerald-400) !important;
}

/* ══ SELECT SLIDER ══ */
.stSlider > div > div > div {
    background: var(--emerald-500) !important;
}

/* ══ BAR CHART ══ */
.stVegaLiteChart {
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}

/* ══ ALERTS ══ */
.stAlert {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    font-size: 14px !important;
}

/* ══ SPINNER ══ */
.stSpinner > div {
    border-top-color: var(--emerald-500) !important;
}

/* ══ DIVIDER ══ */
hr {
    border-color: var(--border) !important;
    margin: 24px 0 !important;
}

/* ══ MARKDOWN TEXT ══ */
.stMarkdown p {
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: var(--text-secondary) !important;
}

/* ══ ANIMATIONS ══ */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 20px rgba(0,214,143,0.1); }
    50%       { box-shadow: 0 0 40px rgba(0,214,143,0.2); }
}

.metric-card { animation: fadeSlideUp 0.4s ease both; }
.metric-card:nth-child(1) { animation-delay: 0.05s; }
.metric-card:nth-child(2) { animation-delay: 0.10s; }
.metric-card:nth-child(3) { animation-delay: 0.15s; }
.metric-card:nth-child(4) { animation-delay: 0.20s; }

.model-badge { animation: pulseGlow 3s ease-in-out infinite; }

</style>
""", unsafe_allow_html=True)

# ===== SESSION STATE =====
for key, default in [
    ("messages", []),
    ("session_start", datetime.datetime.now()),
    ("word_buffer", ""),
    ("last_added_letter", ""),
    ("agent_result", None),
    ("conversation_context", []),
    ("word_history", []),
    ("overlay_running", False),
    ("overlay_pid", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ===== LOAD MODEL STATUS =====
bisindo_model, bisindo_classes = load_model()
model_loaded = bisindo_model is not None

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown('<div class="brand">BISINDO AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Deteksi Isyarat 2 Tangan · A–Z · Indonesia</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if model_loaded:
        st.markdown('<div class="model-badge">✦ Custom Model A–Z Aktif</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ bisindo_model.pkl tidak ditemukan")

    st.markdown("<br>", unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Dashboard",
        "📷  Gesture Detector",
        "🎬  Overlay Subtitle",
        "💬  BISINDO Bot",
        "📖  Panduan A–Z"
    ], label_visibility="collapsed")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:10px;color:rgba(255,255,255,0.15);letter-spacing:1.5px;line-height:2">
    SECOND BYTE<br>
    TEKNIK INFORMATIKA<br>
    2026
    </div>""", unsafe_allow_html=True)


# ==========================
# PAGE: DASHBOARD
# ==========================
if page == "📊  Dashboard":
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Sesi dimulai {st.session_state.session_start.strftime("%d %B %Y · %H:%M")}</div>', unsafe_allow_html=True)

    data = load_gesture_data()
    gesture_counts = data["counts"]
    for g in ALPHABET:
        if g not in gesture_counts:
            gesture_counts[g] = 0
    total = data["total"]

    col_r, _ = st.columns([1, 5])
    with col_r:
        if st.button("🔄 Refresh"):
            st.rerun()

    unique = sum(1 for v in gesture_counts.values() if v > 0)
    most = max(gesture_counts, key=gesture_counts.get) if total > 0 else "—"
    elapsed = int((datetime.datetime.now() - st.session_state.session_start).total_seconds() // 60)

    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (c1, str(total), "TOTAL DETEKSI", "linear-gradient(135deg,#00e5a0,#00c8f0)"),
        (c2, str(unique) + "/26", "HURUF TERJANGKAU", "linear-gradient(135deg,#8080ff,#c080ff)"),
        (c3, most, "HURUF FAVORIT", "linear-gradient(135deg,#ffb347,#ffcc02)"),
        (c4, str(elapsed) + " mnt", "DURASI SESI", "linear-gradient(135deg,#f78ca0,#f9748f)"),
    ]
    for col, num, lbl, grad in metrics:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-num" style="background:{grad};-webkit-background-clip:text;-webkit-text-fill-color:transparent">{num}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1.7, 1])

    with col_l:
        st.markdown('<div class="section-head">📈 Statistik Penggunaan</div>', unsafe_allow_html=True)
        if total > 0:
            st.bar_chart({g: gesture_counts[g] for g in ALPHABET}, height=250)
        else:
            st.markdown("""<div style="background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                border-radius:14px;padding:40px;text-align:center;color:rgba(255,255,255,0.2);font-size:13px">
                Belum ada data — buka Gesture Detector untuk mulai 🤟
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head">🎮 Simulator Manual</div>', unsafe_allow_html=True)
        rows = [ALPHABET[i:i + 7] for i in range(0, 26, 7)]
        for row in rows:
            cols_btn = st.columns(len(row))
            for col, letter in zip(cols_btn, row):
                with col:
                    if st.button(letter, key=f"sim_{letter}", use_container_width=True):
                        gesture_counts[letter] += 1
                        total += 1
                        save_gesture_data(gesture_counts, total)
                        st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺  Reset Semua Data", use_container_width=True):
            save_gesture_data({g: 0 for g in ALPHABET}, 0)
            st.rerun()

    with col_r:
        st.markdown('<div class="section-head">🏆 Ranking Huruf</div>', unsafe_allow_html=True)
        sorted_g = sorted(gesture_counts.items(), key=lambda x: x[1], reverse=True)
        medals = ["🥇", "🥈", "🥉"]
        for rank, (letter, count) in enumerate(sorted_g[:10], 1):
            pct = (count / max(total, 1)) * 100
            badge = medals[rank - 1] if rank <= 3 else f"{rank}"
            color = f"hsl({(ord(letter) - 65) * 14}, 70%, 65%)"
            st.markdown(f"""<div class="rank-card">
                <span style="font-size:13px;color:rgba(255,255,255,0.25);width:20px">{badge}</span>
                <span style="font-family:'Syne',sans-serif;font-size:22px;font-weight:800;color:{color}">{letter}</span>
                <div style="flex:1">
                    <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden">
                        <div style="width:{pct}%;height:100%;background:{color};border-radius:99px"></div>
                    </div>
                </div>
                <span style="font-size:11px;color:rgba(255,255,255,0.25)">{count}x</span>
            </div>""", unsafe_allow_html=True)


# ==========================
# PAGE: GESTURE DETECTOR
# ==========================
elif page == "📷  Gesture Detector":
    st.markdown('<div class="page-title">Gesture Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Deteksi alfabet A–Z BISINDO secara real-time langsung dari browser</div>', unsafe_allow_html=True)

    if not model_loaded:
        st.error("⚠️ File `bisindo_model.pkl` dan `bisindo_classes.pkl` tidak ditemukan.")
        st.stop()

    col_cam, col_info = st.columns([2, 1])

    with col_cam:
        st.markdown('<div class="section-head">📷 Kamera Live</div>', unsafe_allow_html=True)
        ctx = webrtc_streamer(
            key="bisindo-az",
            video_processor_factory=GestureProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [
                {"urls": ["stun:stun.relay.metered.ca:80"]},
                {"urls": ["turn:global.relay.metered.ca:80"], "username": "3ae2008838fdc2997a49ff80", "credential": "NTcQ/vikRr+BfhSZ"},
                {"urls": ["turn:global.relay.metered.ca:80?transport=tcp"], "username": "3ae2008838fdc2997a49ff80", "credential": "NTcQ/vikRr+BfhSZ"},
                {"urls": ["turn:global.relay.metered.ca:443"], "username": "3ae2008838fdc2997a49ff80", "credential": "NTcQ/vikRr+BfhSZ"},
                {"urls": ["turns:global.relay.metered.ca:443?transport=tcp"], "username": "3ae2008838fdc2997a49ff80", "credential": "NTcQ/vikRr+BfhSZ"},
            ]}),
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "min": 320},
                    "height": {"ideal": 480, "min": 240},
                    "frameRate": {"ideal": 15, "min": 10},
                    "facingMode": "user",
                },
                "audio": False,
            },
            async_processing=False,
        )

    with col_info:
        st.markdown('<div class="section-head">📡 Status Deteksi</div>', unsafe_allow_html=True)

        if ctx.video_processor:
            proc = ctx.video_processor
            gesture = proc.gesture
            confidence = proc.confidence
            hand_count = proc.hand_count
            stable_progress = proc.stable_progress

            if gesture:
                color = f"hsl({(ord(gesture) - 65) * 14}, 70%, 65%)"
                desc = GESTURE_DESC.get(gesture, "")
                stable_pct = int(stable_progress * 100)
                stable_color = "#00ff88" if stable_progress >= 1.0 else "#ffb347"
                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                    border-radius:20px;padding:28px;text-align:center;margin-bottom:14px">
                    <div style="font-family:'Syne',sans-serif;font-size:80px;font-weight:800;
                        color:{color};line-height:1">{gesture}</div>
                    <div style="font-size:12px;color:rgba(255,255,255,0.4);margin-top:8px">{desc}</div>
                    <div style="margin-top:14px;background:rgba(255,255,255,0.05);border-radius:99px;
                        height:6px;overflow:hidden">
                        <div style="width:{int(confidence * 100)}%;height:100%;background:{color};border-radius:99px"></div>
                    </div>
                    <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:6px">
                        Confidence: {int(confidence * 100)}%</div>
                    <div style="margin-top:10px;background:rgba(255,255,255,0.05);border-radius:99px;
                        height:4px;overflow:hidden">
                        <div style="width:{stable_pct}%;height:100%;background:{stable_color};border-radius:99px"></div>
                    </div>
                    <div style="font-size:10px;color:rgba(255,255,255,0.25);margin-top:4px">
                        Stabilitas: {stable_pct}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div style="background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                    border-radius:20px;padding:40px;text-align:center;color:rgba(255,255,255,0.25);font-size:13px">
                    Menunggu gesture tangan...
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="metric-card" style="margin-top:12px">
                <div class="metric-num" style="color:#00e5a0">{hand_count}</div>
                <div class="metric-lbl">TANGAN TERDETEKSI</div>
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""<div style="background:rgba(255,255,255,0.02);border:1px dashed rgba(255,255,255,0.08);
                border-radius:20px;padding:40px;text-align:center;color:rgba(255,255,255,0.25);font-size:13px">
                Klik <strong>START</strong> untuk aktifkan kamera
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head">📋 Semua Huruf</div>', unsafe_allow_html=True)
        rows = [ALPHABET[i:i + 4] for i in range(0, 26, 4)]
        for row in rows:
            cols = st.columns(4)
            for i, letter in enumerate(row):
                color = f"hsl({(ord(letter) - 65) * 14}, 70%, 65%)"
                with cols[i]:
                    st.markdown(f"""<div style="text-align:center;padding:6px 0;
                        border-bottom:1px solid rgba(255,255,255,0.04)">
                        <span style="font-family:'Syne',sans-serif;font-size:18px;
                            font-weight:800;color:{color}">{letter}</span>
                    </div>""", unsafe_allow_html=True)


# ==========================
# PAGE: OVERLAY SUBTITLE
# ==========================
elif page == "🎬  Overlay Subtitle":
    st.markdown('<div class="page-title">Overlay Subtitle</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Subtitle real-time yang tampil di atas PPT, PDF, atau Word saat mengajar</div>', unsafe_allow_html=True)

    # ── Cara pakai ──────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(0,214,143,0.06);border:1px solid rgba(0,214,143,0.2);
        border-radius:16px;padding:24px 28px;margin-bottom:24px;position:relative;overflow:hidden">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
            background:linear-gradient(90deg,#00D68F,#22D3EE)"></div>
        <div style="font-family:'Syne',sans-serif;font-size:15px;font-weight:700;
            color:#00D68F;margin-bottom:14px;letter-spacing:0.5px">
            🎬 Cara Menggunakan Overlay Subtitle
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px">
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:28px;margin-bottom:8px">1️⃣</div>
                <div style="font-size:13px;font-weight:600;color:#EEF2FF;margin-bottom:4px">Buka Presentasi</div>
                <div style="font-size:11px;color:#8FA3BF;line-height:1.5">Buka PPT, PDF, atau Word terlebih dahulu sampai tampil fullscreen</div>
            </div>
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:28px;margin-bottom:8px">2️⃣</div>
                <div style="font-size:13px;font-weight:600;color:#EEF2FF;margin-bottom:4px">Jalankan Overlay</div>
                <div style="font-size:11px;color:#8FA3BF;line-height:1.5">Buka terminal, jalankan <code style="background:rgba(255,255,255,0.1);padding:2px 6px;border-radius:4px">python overlay.py</code></div>
            </div>
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:28px;margin-bottom:8px">3️⃣</div>
                <div style="font-size:13px;font-weight:600;color:#EEF2FF;margin-bottom:4px">Mulai Gesture</div>
                <div style="font-size:11px;color:#8FA3BF;line-height:1.5">Tunjukkan gesture huruf BISINDO ke kamera — huruf muncul otomatis</div>
            </div>
            <div style="background:rgba(255,255,255,0.04);border-radius:12px;padding:16px;text-align:center">
                <div style="font-size:28px;margin-bottom:8px">4️⃣</div>
                <div style="font-size:13px;font-weight:600;color:#EEF2FF;margin-bottom:4px">Tekan Enter</div>
                <div style="font-size:11px;color:#8FA3BF;line-height:1.5">AI Agent menyusun kalimat otomatis dan tampil sebagai subtitle</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fitur overlay ────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-head">✨ Fitur Overlay</div>', unsafe_allow_html=True)
        features = [
            ("📺", "Kamera PiP", "Kamera kecil di pojok kanan bawah layar, selalu tampil di atas presentasi"),
            ("📝", "Subtitle Animasi", "Kalimat muncul dengan efek typewriter di bawah tengah layar"),
            ("🤖", "AI Agent", "LLaMA 3.3 70B menyusun huruf menjadi kalimat Indonesia yang natural"),
            ("⏸", "Pause / Lanjut", "Tekan P untuk jeda saat tanya jawab, P lagi untuk lanjut"),
            ("🖱️", "Draggable", "Semua window bisa dipindah ke posisi yang paling nyaman"),
            ("⌨️", "Hotkey Lengkap", "Enter = proses AI · Backspace = hapus · Space = spasi · Esc = reset"),
        ]
        for icon, title, desc in features:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:14px;
                background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                border-radius:12px;padding:14px 16px;margin-bottom:8px">
                <span style="font-size:22px;flex-shrink:0">{icon}</span>
                <div>
                    <div style="font-size:13px;font-weight:600;color:#EEF2FF;margin-bottom:3px">{title}</div>
                    <div style="font-size:12px;color:#8FA3BF;line-height:1.5">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-head">⌨️ Shortcut Keyboard</div>', unsafe_allow_html=True)
        shortcuts = [
            ("Enter", "Proses huruf dengan AI Agent → tampil sebagai kalimat di subtitle"),
            ("Backspace", "Hapus 1 huruf terakhir dari buffer"),
            ("Space", "Tambah spasi antar kata"),
            ("P", "Pause / Lanjut deteksi gesture"),
            ("Esc", "Reset semua buffer & mulai dari awal"),
        ]
        for key, desc in shortcuts:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;
                padding:12px 0;border-bottom:1px solid rgba(255,255,255,0.06)">
                <kbd style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
                    border-radius:6px;padding:4px 12px;font-family:'JetBrains Mono',monospace;
                    font-size:13px;font-weight:600;color:#00D68F;min-width:90px;text-align:center;
                    display:inline-block">{key}</kbd>
                <span style="font-size:13px;color:#8FA3BF;line-height:1.5">{desc}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-head">🖥️ Tampilan Overlay</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#070B18;border:1px solid rgba(255,255,255,0.1);
            border-radius:14px;padding:20px;position:relative;overflow:hidden">

            <!-- Simulasi kamera PiP -->
            <div style="position:absolute;top:14px;right:14px;
                background:#0D1426;border:2px solid #00D68F;border-radius:8px;
                width:90px;height:66px;display:flex;align-items:center;justify-content:center">
                <div style="text-align:center">
                    <div style="font-size:18px">📷</div>
                    <div style="font-size:9px;color:#00D68F;margin-top:2px">LIVE</div>
                </div>
            </div>

            <!-- Simulasi konten PPT -->
            <div style="background:#0D1426;border-radius:8px;padding:20px 16px;
                margin-bottom:12px;min-height:80px;display:flex;align-items:center;justify-content:center">
                <div style="text-align:center">
                    <div style="font-size:13px;font-weight:600;color:#8FA3BF">[ Slide Presentasi ]</div>
                    <div style="font-size:11px;color:#3D5A80;margin-top:4px">PPT / PDF / Word</div>
                </div>
            </div>

            <!-- Simulasi subtitle bar -->
            <div style="background:#0D1426;border:1px solid #00D68F;border-radius:8px;
                padding:10px 16px;text-align:center">
                <div style="font-size:10px;color:#00D68F;margin-bottom:4px;letter-spacing:1px">
                    BISINDO AI · LIVE SUBTITLE
                </div>
                <div style="font-size:15px;font-weight:700;color:white">
                    Selamat pagi, mari kita mulai pelajaran hari ini.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Launcher tombol ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-head">🚀 Jalankan Overlay</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:rgba(252,211,77,0.06);border:1px solid rgba(252,211,77,0.25);
        border-radius:14px;padding:20px 24px;margin-top:8px">
        <div style="font-size:15px;font-weight:700;color:#FCD34D;margin-bottom:8px">
            🖥️ Fitur Overlay hanya tersedia di versi Desktop
        </div>
        <div style="font-size:13px;color:#8FA3BF;line-height:1.7">
            Overlay subtitle memerlukan akses ke desktop lokal (Tkinter window).
            Fitur ini tidak dapat dijalankan di platform cloud seperti Streamlit Cloud.<br><br>
            Untuk menggunakan overlay, jalankan aplikasi secara lokal dengan perintah:<br>
            <code style="background:rgba(255,255,255,0.08);padding:4px 10px;border-radius:6px;
                font-family:monospace;color:#00D68F">streamlit run main.py</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tips
    st.markdown("""
    <div style="background:rgba(252,211,77,0.06);border:1px solid rgba(252,211,77,0.15);
        border-radius:12px;padding:14px 18px;font-size:13px;color:#FCD34D;margin-top:12px">
        💡 <strong>Tips:</strong> Buka file presentasi (PPT/PDF/Word) terlebih dahulu,
        lalu klik tombol <strong>AKTIFKAN OVERLAY</strong> — subtitle akan langsung muncul
        di atas presentasi kamu.
    </div>
    """, unsafe_allow_html=True)

# PAGE: BISINDO BOT
# ==========================
elif page == "💬  BISINDO Bot":
    st.markdown('<div class="page-title">BISINDO Bot</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Asisten AI untuk belajar Bahasa Isyarat Indonesia</div>', unsafe_allow_html=True)

    st.markdown("""<div class="chat-bubble-wrap">
        <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;
            color:rgba(255,255,255,0.6);margin-bottom:12px;text-transform:uppercase;letter-spacing:1.5px">
            💡 Contoh pertanyaan</div>
        <div style="display:flex;flex-wrap:wrap;gap:8px">
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                border-radius:99px;padding:5px 14px;font-size:12px;color:rgba(255,255,255,0.5)">
                Bagaimana cara buat huruf A?</span>
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                border-radius:99px;padding:5px 14px;font-size:12px;color:rgba(255,255,255,0.5)">
                Apa itu BISINDO?</span>
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                border-radius:99px;padding:5px 14px;font-size:12px;color:rgba(255,255,255,0.5)">
                Bedanya BISINDO dan SIBI?</span>
            <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                border-radius:99px;padding:5px 14px;font-size:12px;color:rgba(255,255,255,0.5)">
                Cara belajar BISINDO pemula?</span>
        </div>
    </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Tanya seputar BISINDO..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner(""):
                try:
                    history = [{"role": "system", "content": SYSTEM_PROMPT}]
                    for msg in st.session_state.messages:
                        history.append({"role": msg["role"], "content": msg["content"]})
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=history,
                        max_tokens=600
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.messages:
        if st.button("🗑️ Hapus riwayat chat"):
            st.session_state.messages = []
            st.rerun()


# ==========================
# PAGE: PANDUAN A-Z
# ==========================
elif page == "📖  Panduan A–Z":
    st.markdown('<div class="page-title">Panduan A–Z</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">26 huruf alfabet BISINDO dengan gesture 2 tangan — sesuai data rekaman sistem</div>', unsafe_allow_html=True)

    search = st.text_input("🔍 Cari huruf...", placeholder="Ketik huruf, contoh: A").upper().strip()
    filtered = [l for l in ALPHABET if not search or l == search]

    cols = st.columns(4)
    for i, letter in enumerate(filtered):
        color = f"hsl({(ord(letter) - 65) * 14}, 70%, 65%)"
        desc = GESTURE_DESC.get(letter, "")
        with cols[i % 4]:
            st.markdown(f"""<div class="alpha-card">
                <div class="alpha-letter" style="color:{color}">{letter}</div>
                <div class="alpha-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
        border-radius:20px;padding:28px 32px">
        <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:16px;
            color:rgba(255,255,255,0.7);margin-bottom:18px">⚙️ Tech Stack</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
            <div style="background:rgba(0,214,143,0.05);border:1px solid rgba(0,214,143,0.15);border-radius:14px;padding:18px">
                <div style="font-size:9px;letter-spacing:2.5px;color:rgba(0,214,143,0.6);margin-bottom:8px;font-weight:600">DETEKSI LANDMARK</div>
                <div style="color:rgba(255,255,255,0.8);font-size:13px;font-weight:500">MediaPipe Hand Landmarker</div>
                <div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:4px">21 titik · 2 tangan · Float16 model</div>
            </div>
            <div style="background:rgba(0,214,143,0.05);border:1px solid rgba(0,214,143,0.15);border-radius:14px;padding:18px">
                <div style="font-size:9px;letter-spacing:2.5px;color:rgba(0,214,143,0.6);margin-bottom:8px;font-weight:600">KLASIFIKASI GESTURE</div>
                <div style="color:rgba(255,255,255,0.8);font-size:13px;font-weight:500">Random Forest Classifier</div>
                <div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:4px">EMA Smoothing · Majority Voting · Confidence Threshold · Sort-by-X Stabilization</div>
            </div>
            <div style="background:rgba(0,214,143,0.05);border:1px solid rgba(0,214,143,0.15);border-radius:14px;padding:18px">
                <div style="font-size:9px;letter-spacing:2.5px;color:rgba(0,214,143,0.6);margin-bottom:8px;font-weight:600">AI AGENT & CHATBOT</div>
                <div style="color:rgba(255,255,255,0.8);font-size:13px;font-weight:500">Groq · LLaMA 3.3 70B Versatile</div>
                <div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:4px">Agentic sentence building · Word prediction · Natural language refinement</div>
            </div>
            <div style="background:rgba(0,214,143,0.05);border:1px solid rgba(0,214,143,0.15);border-radius:14px;padding:18px">
                <div style="font-size:9px;letter-spacing:2.5px;color:rgba(0,214,143,0.6);margin-bottom:8px;font-weight:600">FITUR UTAMA</div>
                <div style="color:rgba(255,255,255,0.8);font-size:13px;font-weight:500">Overlay Subtitle Real-Time</div>
                <div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:4px">Picture-in-Picture kamera · Subtitle animasi · Always-on-top · Berjalan di atas PPT/PDF</div>
            </div>
            <div style="background:rgba(0,214,143,0.05);border:1px solid rgba(0,214,143,0.15);border-radius:14px;padding:18px">
                <div style="font-size:9px;letter-spacing:2.5px;color:rgba(0,214,143,0.6);margin-bottom:8px;font-weight:600">DATA & TRAINING</div>
                <div style="color:rgba(255,255,255,0.8);font-size:13px;font-weight:500">Custom Dataset BISINDO A–Z</div>
                <div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:4px">7.600+ samples · 2 tangan · Normalisasi koordinat · StandardScaler</div>
            </div>
            <div style="background:rgba(0,214,143,0.05);border:1px solid rgba(0,214,143,0.15);border-radius:14px;padding:18px">
                <div style="font-size:9px;letter-spacing:2.5px;color:rgba(0,214,143,0.6);margin-bottom:8px;font-weight:600">PLATFORM</div>
                <div style="color:rgba(255,255,255,0.8);font-size:13px;font-weight:500">Streamlit · WebRTC · Tkinter</div>
                <div style="color:rgba(255,255,255,0.4);font-size:11px;margin-top:4px">Web dashboard · Desktop overlay · Python 3.13</div>
            </div>
        </div>
        <div style="margin-top:24px;padding-top:18px;border-top:1px solid rgba(255,255,255,0.06);
            text-align:center;font-size:12px;color:rgba(255,255,255,0.25);font-style:italic;line-height:2.2">
            "Teknologi terbaik adalah yang memberdayakan mereka yang paling membutuhkan."<br>
            <span style="font-size:11px;color:rgba(0,214,143,0.5)">— Second Byte · Teknik Informatika · Universitas Mercu Buana · 2026</span>
        </div>
    </div>""", unsafe_allow_html=True)
