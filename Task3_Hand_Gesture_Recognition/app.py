
import streamlit as st
import cv2
import mediapipe as mp
import av

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    WebRtcMode,
)

st.set_page_config(
    page_title="AI Hand Gesture Recognition",
    page_icon="✋",
    layout="wide",
)

st.title("✋ AI Hand Gesture Recognition")
st.write(
    "Real-time hand gesture detection using Python, "
    "OpenCV and MediaPipe."
)

st.info(
    "Click START, allow camera permission, and show your hand "
    "in front of the camera."
)

st.subheader("Supported Gestures")
st.write("✋ Open Palm  |  ✊ Fist  |  ☝️ One Finger  |  ✌️ Victory")

def identify_gesture(hand_landmarks):
    points = hand_landmarks.landmark

    tips = [8, 12, 16, 20]
    joints = [6, 10, 14, 18]

    extended = [
        points[tip].y < points[joint].y
        for tip, joint in zip(tips, joints)
    ]

    count = sum(extended)

    if count == 0:
        return "Fist"
    elif count == 4:
        return "Open Palm"
    elif (
        extended[0]
        and extended[1]
        and not extended[2]
        and not extended[3]
    ):
        return "Victory / Peace"
    elif extended[0] and not any(extended[1:]):
        return "One Finger"
    else:
        return "Other Gesture"


class HandGestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        gesture = "No Hand Detected"

        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                gesture = identify_gesture(hand)

                self.drawing.draw_landmarks(
                    image,
                    hand,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )

        cv2.putText(
            image,
            gesture,
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


st.subheader("Live Camera")

webrtc_streamer(
    key="hand-gesture-camera",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=HandGestureProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.markdown("### How to use")
st.markdown(
    """
    1. Click **START** in the camera section.
    2. Allow camera permission in your browser.
    3. Show your hand clearly in front of the camera.
    4. Try a fist, open palm, one finger or peace sign.
    """
)

st.divider()
st.caption("Task 3 | AI Hand Gesture Recognition")

