import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import av
import time
import numpy as np
import pandas as pd
from faceDetect import FaceDetector
import random


st.title("Nose Job")

start_time = 0
end_time = 0

class VideoProcessor:
    def __init__(self) -> None:
        try:
            self.df = pd.read_csv("data.csv")
        except:
            self.df = pd.DataFrame(columns=['Name', 'Score'])
            
        self.frame_num = 0
        
        self.frameWidth = 680
        self.frameHeight = 480
        
        self.a1, self.a2, self.a3, self.a4 = 800, 0, 750, int(self.frameHeight/2)-50
        # rectangles = [(a1, a2, a3, a4)]
        self.rectangles = []


        self.b1, self.b2, self.b3, self.b4 = 800, int(self.frameHeight/2)+50, 750, int(self.frameHeight)
        # inverse_rectangles = [(b1, b2, b3, b4)]
        self.inverse_rectangles = []
        
        self.detector = FaceDetector()
        self.collision_num = 0
        self.collision = False
        self.Score = 0
        self.speed = 5
        
    def check_collision_circle_rect(self, circle_center, circle_radius, rect_points):
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
        start_time = time.time()
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        frame = self.detector.findFace(img, draw=True)
        lmlist = self.detector.findPosition(frame, draw=False)
        
        
        
        if len(lmlist)>0:
            cv2.putText(frame, f'Score: {self.Score - 2}', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            if self.frame_num % 40 == 0:
                # a1, a2, a3, a4 = a1, a2, a3, a4+random.randint(50,100)
                temp_y = random.randint(int(self.frameHeight/4),int(self.frameHeight * 0.75))
                a1, a2, a3, a4 = self.a1, self.a2, self.a3, temp_y-50
                b1, b2, b3, b4 = self.b1, temp_y+50, self.b3, self.b4
                self.rectangles.append((a1, a2, a3, a4))
                self.inverse_rectangles.append((b1, b2, b3, b4))
                self.Score += 1

            if len(self.rectangles) > 4:
                self.rectangles = self.rectangles[1:]
                self.inverse_rectangles = self.inverse_rectangles[1:]
            
            # speed = 5
            
            if self.frame_num % 400 == 0:
                self.speed += 1
            
            arr = np.array(self.rectangles)
            arr[:, [0, 2]] -= self.speed
            self.rectangles = [tuple(row) for row in arr]
            
            arr = np.array(self.inverse_rectangles)
            arr[:, [0, 2]] -= self.speed
            self.inverse_rectangles = [tuple(row) for row in arr]
            
            for x1, y1, x2, y2 in self.rectangles:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)
            for x1, y1, x2, y2 in self.inverse_rectangles:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), -1)

            self.frame_num += 1   

            # Collision detection between rectangles and landmarks
            
            px, py = int((lmlist[0][1][0] + lmlist[0][1][0] + lmlist[0][1][2]) / 2), int((lmlist[0][1][1] + lmlist[0][1][1] + lmlist[0][1][3]) / 2) 
            point = (px, py)
            frame = cv2.circle(frame, point, 10, (255,0,0), -1)
            
            for rect in self.rectangles:          
                if self.check_collision_circle_rect(point, 10, rect):
                    self.collision = True
                    self.collision_num += 1
                    print("Collision ->", self.collision_num)

            if self.collision == False:
                for rect in self.inverse_rectangles:
                    if self.check_collision_circle_rect(point, 10, rect):
                        self.collision = True
                        self.collision_num += 1
                        print("Collision ->", self.collision_num)     
            
            if self.collision == True:
                time.sleep(5) 
            
        end_time = time.time()
        fps = 1/(end_time - start_time)
        end_time = start_time
        img = cv2.putText(img, str(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video" : True,
        "audio" : False
    }
)
if ctx.video_processor:
    pass
    
