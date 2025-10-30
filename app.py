import streamlit as st
import cv2
import tempfile
import numpy as np

# ----------------------------
# Load Model
# ----------------------------
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

model = cv2.dnn_DetectionModel(frozen_model, config_file)
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# ----------------------------
# Load Labels
# ----------------------------
classLabels = []
with open('labels.txt', 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

font = cv2.FONT_HERSHEY_COMPLEX

# ----------------------------
# Streamlit Page Settings
# ----------------------------
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("🎯 Real-Time Object Detection")

# ✨ Decorative Title Section
st.markdown("""
<h2 style="
    text-align:center;
    font-family:'Poppins', sans-serif;
    font-size:1.8rem;
    font-weight:700;
    color:#1abc9c;
    background: linear-gradient(90deg, #1abc9c, #16a085, #2ecc71);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 10px rgba(26,188,156,0.6), 0 0 20px rgba(22,160,133,0.5);
    animation: glow 3s ease-in-out infinite alternate;
    margin-top: -10px;
">
✨ Try it out — and enjoy the magic! ✨
</h2>

<p style='text-align:center; font-size:22px;'>🌟💫🚀</p>

<style>
@keyframes glow {
  from { text-shadow: 0 0 8px rgba(26,188,156,0.5), 0 0 15px rgba(22,160,133,0.3); }
  to { text-shadow: 0 0 20px rgba(26,188,156,0.8), 0 0 35px rgba(22,160,133,0.6); }
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Styling + Mode Selection
# ----------------------------
st.markdown("""
<style>
/* Hide Streamlit toolbar/search */
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebarNav"] {
    display: none !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0e1117;
    border-right: 2px solid #1abc9c;
    box-shadow: 2px 0px 12px rgba(26, 188, 156, 0.25);
    padding-top: 20px;
}

/* Custom box around radio buttons */
.mode-box {
    border: 2px solid #1abc9c;
    border-radius: 15px;
    padding: 25px 20px;
    background: linear-gradient(145deg, #10151b, #0e1117);
    box-shadow: 0px 0px 15px rgba(26, 188, 156, 0.35);
    transition: all 0.3s ease-in-out;
    margin: 25px auto;
    width: 90%;
    text-align: center;
    transform: translateX(-5%);
}

/* Title inside box */
.mode-title {
    color: #1abc9c;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 15px;
    text-shadow: 0px 0px 10px rgba(26, 188, 156, 0.5);
}

/* Align radio group and options */
div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    padding-left: 8px;
}
div[role="radiogroup"] label {
    background-color: #14181e;
    border-radius: 12px;
    padding: 10px 15px;
    width: 100%;
    transition: all 0.3s ease-in-out;
    border: 1px solid #1f2a35;
    color: #EAEAEA;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Hover */
div[role="radiogroup"] label:hover {
    transform: scale(1.03);
    border-color: #1abc9c;
    box-shadow: 0px 0px 8px rgba(26, 188, 156, 0.4);
}

/* Selected */
div[role="radiogroup"] label:has(input:checked) {
    border: 2px solid #1abc9c;
    transform: scale(1.08);
    box-shadow: 0px 0px 15px rgba(26, 188, 156, 0.6);
    background: linear-gradient(135deg, #10231b, #0f2923);
    color: #1abc9c;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Options
# ----------------------------
with st.sidebar:
    st.markdown("<div class='mode-title'>🎮 Choose Detection Mode</div>", unsafe_allow_html=True)
    mode = st.radio("", ["Webcam", "Video", "Image"])
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Webcam Mode
# ----------------------------
if mode == "Webcam":
    st.write("### Live Webcam Object Detection")
    st.info("Click **Start Webcam** to activate, unclick to stop.")
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Webcam not detected!")
            break

        ClassIndex, conf, bbox = model.detect(frame, confThreshold=0.55)
        if len(ClassIndex) != 0:
            for classId, confidence, box in zip(ClassIndex.flatten(), conf.flatten(), bbox):
                if classId <= len(classLabels):
                    cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(frame, classLabels[classId - 1].upper(),
                                (box[0] + 5, box[1] + 20),
                                font, fontScale=0.7, color=(0, 255, 0), thickness=2)

        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()

# ----------------------------
# Video Mode
# ----------------------------
elif mode == "Video":
    st.write("### 🎥 Object Detection on Video")

    option = st.selectbox("Select an option:", ["Default Sample", "Upload your own video"])

    if option == "Default Sample":
        default_path = "pexels-george-morina-5330833.mp4"
        st.video(default_path)
        cap = cv2.VideoCapture(default_path)

    elif option == "Upload your own video":
        uploaded_file = st.file_uploader("Upload your video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = None

    if 'cap' in locals() and cap is not None:
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ClassIndex, conf, bbox = model.detect(frame, confThreshold=0.55)
            if len(ClassIndex) != 0:
                for classId, confidence, box in zip(ClassIndex.flatten(), conf.flatten(), bbox):
                    if classId <= len(classLabels):
                        cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                        cv2.putText(frame, classLabels[classId - 1].upper(),
                                    (box[0] + 5, box[1] + 20),
                                    font, fontScale=0.7, color=(0, 255, 0), thickness=2)

            stframe.image(frame, channels="BGR")

        cap.release()
        st.success("✅ Video processed successfully — detection completed.")

# ----------------------------
# Image Mode
# ----------------------------
elif mode == "Image":
    st.write("### 🖼️ Object Detection on Image")

    option = st.selectbox("Select an option:", ["Default Sample", "Upload your own image"])

    if option == "Default Sample":
        default_img_path = "traffic.jpg"
        img = cv2.imread(default_img_path)
        st.image(default_img_path, caption="Default Image", use_container_width=True)

    elif option == "Upload your own image":
        uploaded_file = st.file_uploader("Upload your image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
        else:
            img = None

    if img is not None:
        ClassIndex, conf, bbox = model.detect(img, confThreshold=0.55)
        if len(ClassIndex) != 0:
            for classId, confidence, box in zip(ClassIndex.flatten(), conf.flatten(), bbox):
                if classId <= len(classLabels):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classLabels[classId - 1].upper(),
                                (box[0] + 5, box[1] + 20),
                                font, fontScale=0.7, color=(0, 255, 0), thickness=2)
        st.image(img, channels="BGR", caption="Detected Image", use_container_width=True)
