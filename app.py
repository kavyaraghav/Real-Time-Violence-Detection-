"""
Streamlit app — Real-Life Violence Detector (MobileNetV2 + LSTM)
"""

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2

# ---------------- CONFIG (must match training) ----------------
MODEL_PATH = "violence_lstm_model.keras"
IMG_SIZE = 224
SEQUENCE_LENGTH = 20
VIOLENCE_THRESHOLD = 0.5

st.set_page_config(page_title="Violence Detector", page_icon="🚨", layout="centered")


@st.cache_resource(show_spinner="Loading MobileNetV2 feature extractor...")
def load_feature_extractor():
    base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    return base


@st.cache_resource(show_spinner="Loading trained LSTM classifier...")
def load_classifier():
    return tf.keras.models.load_model(MODEL_PATH,compile=False)


def sample_frame_indices(total_frames, n_samples):
    if total_frames <= 0:
        return np.zeros(n_samples, dtype=int)

    if total_frames < n_samples:
        idx = np.arange(total_frames)
        idx = np.pad(idx, (0, n_samples-total_frames), mode="edge")
        return idx

    return np.linspace(0, total_frames-1, n_samples).astype(int)


def read_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(total, SEQUENCE_LENGTH)
    idx_set = set(idxs.tolist())
    max_idx = int(max(idxs))
    frame_cache = {}
    current = 0
    while current <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if current in idx_set:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame_cache[current] = frame
        current += 1
    cap.release()
    frames = []
    for i in idxs:
        if i in frame_cache:
            frames.append(frame_cache[i])
        elif frame_cache:
            frames.append(list(frame_cache.values())[-1])
        else:
            frames.append(np.zeros((IMG_SIZE, IMG_SIZE,3),dtype=np.uint8))
    return np.array(frames)



def predict_video(video_path, extractor, classifier):

    frames = read_frames(video_path)

    if len(frames) == 0:
        return None

    x = preprocess_input(frames.astype(np.float32))

    features = extractor.predict(
        x,
        batch_size=SEQUENCE_LENGTH,
        verbose=0
    )

    features = np.expand_dims(features, axis=0)

    probability = float(
        classifier.predict(features, verbose=0)[0][0]
    )

    return probability

def play_siren():
    """Generates a siren tone in the browser with the Web Audio API (no audio file needed)."""
    components.html(
        """
        <script>
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        function beep(freqStart, freqEnd, duration, startAt) {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(freqStart, ctx.currentTime + startAt);
            osc.frequency.linearRampToValueAtTime(freqEnd, ctx.currentTime + startAt + duration);
            gain.gain.setValueAtTime(0.15, ctx.currentTime + startAt);
            osc.connect(gain); gain.connect(ctx.destination);
            osc.start(ctx.currentTime + startAt);
            osc.stop(ctx.currentTime + startAt + duration);
        }
        for (let i = 0; i < 3; i++) {
            beep(600, 1200, 0.5, i * 1.0);
            beep(1200, 600, 0.5, i * 1.0 + 0.5);
        }
        </script>
        """,
        height=0,
    )


# ---------------- UI ----------------
st.title("🚨 Real-Life Violence Detector")
st.caption("MobileNetV2 (per-frame features) + LSTM (temporal classifier)")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.video(video_path)

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Place it next to app.py.")
    else:
        extractor = load_feature_extractor()
        classifier = load_classifier()

        with st.spinner("Analyzing video..."):
            probability = predict_video( video_path,extractor,classifier)


        if probability is None:
            st.error("Could not read video.")
        else:
            is_violent = probability >= VIOLENCE_THRESHOLD
            st.metric( "Violence Confidence",f"{probability:.1%}")
            if is_violent:
                st.error(f"⚠️ VIOLENCE DETECTED (confidence {probability:.1%})")
                play_siren()
            else:
                st.success(f"✅ No violence detected (confidence {probability:.1%})")


    os.unlink(video_path)
else:
    st.info("Upload a video to run detection.")
