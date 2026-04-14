import cv2
import numpy as np
import json

# --- CONFIGURATION ---
CAMERA_INDEX = 0          # Change to 0 or 1 depending on which camera you want to tune
POSITIONS_FILE = "positions.json" # Or whatever you named the output from calib_pos.py
CAMERA_KEY = "cam1"       # The key in your json file for this camera's positions

def nothing(x): pass

def classify_hsv_live(h, s, v, hsv_cfg):
    # Same logic as your cube_to_kociemba.py
    if v < hsv_cfg['v_min'] or v > hsv_cfg['v_max']: return "X" # X = Dark/Unknown
    if s < hsv_cfg['s_min']: return "W"                         # W = White
    if 0 <= h < hsv_cfg['r_o']: return "R"
    if hsv_cfg['r_o'] <= h < hsv_cfg['o_y']: return "O"
    if hsv_cfg['o_y'] <= h < hsv_cfg['y_g']: return "Y"
    if hsv_cfg['y_g'] <= h < hsv_cfg['g_b']: return "G"
    if hsv_cfg['g_b'] <= h < hsv_cfg['b_r']: return "B"
    if hsv_cfg['b_r'] <= h <= 179: return "R" 
    return "X"

def main():
    global CAMERA_KEY, CAMERA_INDEX
    # 1. Load Positions
    try:
        with open(POSITIONS_FILE, "r") as f:
            pos_data = json.load(f)[CAMERA_KEY]
    except Exception as e:
        print(f"Error loading {POSITIONS_FILE}. Make sure the file exists and keys match. {e}")
        return

    cv2.namedWindow('Tuning Feed')
    
    # 2. Sliders
    cv2.createTrackbar('red-orange', 'Tuning Feed', 10, 179, nothing)
    cv2.createTrackbar('orange-yell', 'Tuning Feed', 25, 179, nothing)
    cv2.createTrackbar('yellow-gree', 'Tuning Feed', 40, 179, nothing)
    cv2.createTrackbar('green-blue', 'Tuning Feed', 90, 179, nothing)
    cv2.createTrackbar('blue-red', 'Tuning Feed', 150, 179, nothing)
    cv2.createTrackbar('black-filter', 'Tuning Feed', 50, 255, nothing) 
    cv2.createTrackbar('white-filter', 'Tuning Feed', 60, 255, nothing) 
    cv2.createTrackbar('val-max', 'Tuning Feed', 255, 255, nothing)     

    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Read Sliders
        hsv_cfg = {
            "r_o": cv2.getTrackbarPos('red-orange', 'Tuning Feed'),
            "o_y": cv2.getTrackbarPos('orange-yell', 'Tuning Feed'),
            "y_g": cv2.getTrackbarPos('yellow-gree', 'Tuning Feed'),
            "g_b": cv2.getTrackbarPos('green-blue', 'Tuning Feed'),
            "b_r": cv2.getTrackbarPos('blue-red', 'Tuning Feed'),
            "v_min": cv2.getTrackbarPos('black-filter', 'Tuning Feed'),
            "s_min": cv2.getTrackbarPos('white-filter', 'Tuning Feed'),
            "v_max": cv2.getTrackbarPos('val-max', 'Tuning Feed')
        }

        # 4. Draw Positions and Live Predictions
        for label, pt in pos_data.items():
            x, y = pt['x'], pt['y']
            
            # Sample the center pixel at this position
            h, s, v = current_hsv[y, x] 
            
            # Predict the color based on current sliders
            predicted_letter = classify_hsv_live(h, s, v, hsv_cfg)
            
            # Draw it on screen
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
            cv2.putText(frame, predicted_letter, (x - 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, predicted_letter, (x - 10, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Tuning Feed', frame)

        key = cv2.waitKey(30) & 0xFF
        
        # Save and quit
        if key == ord('x'):
            with open("hsv_config.json", "w") as f:
                json.dump(hsv_cfg, f, indent=2)
            print("Saved tuned lighting to hsv_config.json")
            break
            
        # Swap Cameras!
        elif key == ord('c'):
            # Toggle the configuration
            if CAMERA_KEY == "cam1":
                CAMERA_KEY = "cam2"
                CAMERA_INDEX = 0 # Or whatever index your second camera uses
            else:
                CAMERA_KEY = "cam1"
                CAMERA_INDEX = 1 
                
            print(f"Switched to {CAMERA_KEY}")
            
            # Restart the video capture with the new camera
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()