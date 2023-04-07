import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from faceDetect import FaceDetector
import pandas as pd
import numpy as np

st.title("Nose Job")



detector = FaceDetector()

class VideoProcessor:
    def __init__(self) -> None:
        pass
    
    def check_collision_circle_rect(circle_center, circle_radius, rect_points):
    # Check if center of circle is inside rectangle
        if circle_center[0] >= rect_points[0] and circle_center[0] <= rect_points[2] \
        and circle_center[1] >= rect_points[1] and circle_center[1] <= rect_points[3]:
            return True

        # Check if any edge of the rectangle intersects with the circle
        rect_edges = [(rect_points[0], rect_points[1], rect_points[2], rect_points[1]), # Top edge
                    (rect_points[2], rect_points[1], rect_points[2], rect_points[3]), # Right edge
                    (rect_points[2], rect_points[3], rect_points[0], rect_points[3]), # Bottom edge
                    (rect_points[0], rect_points[3], rect_points[0], rect_points[1])] # Left edge

        for edge in rect_edges:
            pt1 = np.array([edge[0], edge[1]])
            pt2 = np.array([edge[2], edge[3]])
            dist = cv2.pointPolygonTest(np.array([pt1, pt2]), circle_center, True)
            if dist >= -circle_radius:
                return True

        return False


    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cap = cv2.flip(img, 1)
        frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                
        frame_num = 0

        a1, a2, a3, a4 = 800, 0, 750, int(frameHeight/2)-50
        # rectangles = [(a1, a2, a3, a4)]
        rectangles = []


        b1, b2, b3, b4 = 800, int(frameHeight/2)+50, 750, int(frameHeight)
        # inverse_rectangles = [(b1, b2, b3, b4)]
        inverse_rectangles = []
        cap = detector.findFace(cap, draw=True)
        return av.VideoFrame.from_ndarray(cap, format="bgr24")


ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
if ctx.video_processor:
    pass
    
