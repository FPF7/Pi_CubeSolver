import cv2
import json
import time
from typing import Dict, List, Tuple

WINDOW_NAME = "Position Calibration"
DEFAULT_RADIUS = 6

# Mapping of labels for the two camera views
CAM1_LABELS = [
    "F top-left", "F top-middle", "F top-right", "F middle-left", "F center", "F middle-right", "F bottom-left", "F bottom-middle", "F bottom-right",
    "R top-left", "R top-middle", "R top-right", "R middle-left", "R center", "R middle-right", "R bottom-left", "R bottom-middle", "R bottom-right",
    "D top-left", "D top-middle", "D top-right", "D middle-left", "D center", "D middle-right", "D bottom-left", "D bottom-middle", "D bottom-right",
]

CAM2_LABELS = [
    "L top-left", "L top-middle", "L top-right", "L middle-left", "L center", "L middle-right", "L bottom-left", "L bottom-middle", "L bottom-right",
    "B top-left", "B top-middle", "B top-right", "B middle-left", "B center", "B middle-right", "B bottom-left", "B bottom-middle", "B bottom-right",
    "U top-left", "U top-middle", "U top-right", "U middle-left", "U center", "U middle-right", "U bottom-left", "U bottom-middle", "U bottom-right",
]

def draw_boxed_text(image, text, org, text_color=(255, 255, 255), font_scale=0.6, thickness=1, padding=3):
    x, y = org
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(image, (x - padding, y - text_h - padding), (x + text_w + padding, y + baseline + padding), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)

def capture_frame(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {index}")
        return None

    time.sleep(1.0) # Give the camera time to warm up
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"ERROR: Camera at index {index} opened, but returned no data.")
        return None
        
    return frame

class PositionCalibrator:
    def __init__(self, frame, labels, name):
        self.frame = frame
        self.labels = labels
        self.name = name
        self.index = 0
        self.coords = {}
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.click)

    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.index < len(self.labels):
            label = self.labels[self.index]
            self.coords[label] = {"x": int(x), "y": int(y)}
            print(f"{label}: ({x}, {y})")
            self.index += 1

    def run(self):
        while self.index < len(self.labels):
            display = self.frame.copy()
            for i, label in enumerate(self.labels):
                if label in self.coords:
                    pt = self.coords[label]
                    cv2.circle(display, (pt["x"], pt["y"]), 5, (0, 255, 0), -1)
            
            curr = self.labels[self.index] if self.index < len(self.labels) else "DONE"
            draw_boxed_text(display, f"{self.name} - Click: {curr}", (10, 30))
            
            cv2.imshow(WINDOW_NAME, display)
            if cv2.waitKey(20) & 0xFF == ord('q'): break
        return self.coords

if __name__ == "__main__":
    # Try index 1 for FRONT and 2 for BACK
    print("Calibrating Camera FRONT...")
    f1 = capture_frame(1) 
    if f1 is None: 
        print("Still getting None for Cam 1. Check index or USB connection.")
        exit()
    data1 = PositionCalibrator(f1, CAM1_LABELS, "Cam 1").run()
    
    print("Calibrating Camera BACK...")
    f2 = capture_frame(0)
    f2 = cv2.flip(f2,-1)
    if f2 is None:
        print("Still getting None for Cam 2. Check index or USB connection.")
        exit()
    data2 = PositionCalibrator(f2, CAM2_LABELS, "Cam 2").run()

    with open("positions.json", "w") as f:
        json.dump({"cam1": data1, "cam2": data2}, f, indent=2)
    print("Successfully saved positions.json")
    
