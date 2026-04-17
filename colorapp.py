import cv2
import numpy as np
import json

# --- CONFIGURATION ---
CAMERA_INDEX = 0          
POSITIONS_FILE = "positions.json" 
CAMERA_KEY = "cam2"       

def nothing(x): pass

def classify_hsv_live(h, s, v, hsv_cfg):
    # --- DEBUG MODE ---
    # Commented out the brightness filter so it never returns "X"
    # if v < hsv_cfg['v_min'] or v > hsv_cfg['v_max']: return "X" 
    
    # 1. Check if it's washed out enough to be White
    if s < hsv_cfg['s_min']: return "W"                         
    
    # 2. Force a Hue guess based on your slider boundaries
    if 0 <= h < hsv_cfg['r_o']: return "R"
    if hsv_cfg['r_o'] <= h < hsv_cfg['o_y']: return "O"
    if hsv_cfg['o_y'] <= h < hsv_cfg['y_g']: return "Y"
    if hsv_cfg['y_g'] <= h < hsv_cfg['g_b']: return "G"
    if hsv_cfg['g_b'] <= h < hsv_cfg['b_r']: return "B"
    
    # 3. If it's above b_r, it wraps back around to Red. 
    # (Replaced "X" with "R" here to guarantee a color is always returned)
    return "R"

def main():
    global CAMERA_KEY, CAMERA_INDEX
    
    # Load ALL position data into memory once at the beginning
    try:
        with open(POSITIONS_FILE, "r") as f:
            all_pos_data = json.load(f)
        # Set the active positions to the current camera key
        pos_data = all_pos_data[CAMERA_KEY]
    except Exception as e:
        print(f"Error loading {POSITIONS_FILE}. Make sure the file exists and keys match. {e}")
        return

    cv2.namedWindow('Tuning Feed')
    
    # Sliders
    cv2.createTrackbar('red-orange', 'Tuning Feed', 8, 179, nothing)
    cv2.createTrackbar('orange-yell', 'Tuning Feed', 22, 179, nothing)
    cv2.createTrackbar('yellow-gree', 'Tuning Feed', 45, 179, nothing)
    cv2.createTrackbar('green-blue', 'Tuning Feed', 90, 179, nothing)
    cv2.createTrackbar('blue-red', 'Tuning Feed', 150, 179, nothing)
    cv2.createTrackbar('white-filter', 'Tuning Feed', 40, 255, nothing) 
    cv2.createTrackbar('val-max', 'Tuning Feed', 255, 255, nothing)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- FLIP LOGIC ---
        if CAMERA_INDEX == 0:
            frame = cv2.flip(frame, -1)  # Flip Camera 0 vertically
        # elif CAMERA_INDEX == 1:
        #     frame = cv2.flip(frame, 1)  # Keep Camera 1 horizontally mirrored (or remove if you don't want this)
            
        current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        hsv_cfg = {
            "r_o": cv2.getTrackbarPos('red-orange', 'Tuning Feed'),
            "o_y": cv2.getTrackbarPos('orange-yell', 'Tuning Feed'),
            "y_g": cv2.getTrackbarPos('yellow-gree', 'Tuning Feed'),
            "g_b": cv2.getTrackbarPos('green-blue', 'Tuning Feed'),
            "b_r": cv2.getTrackbarPos('blue-red', 'Tuning Feed'),
            "s_min": cv2.getTrackbarPos('white-filter', 'Tuning Feed'),
            "v_max": cv2.getTrackbarPos('val-max', 'Tuning Feed')
        }

        # Draw Positions and Live Predictions using the active pos_data
        # Draw Positions and Live Predictions using the active pos_data
        for label, pt in pos_data.items():
            x, y = pt['x'], pt['y']
            
            # --- BRUTE FORCE THE CENTERS ---
            # If the label is a center sticker, ignore the camera and hardcode the letter.
            # (Assuming standard orientation: U=White, F=Green, R=Red, D=Yellow, L=Orange, B=Blue)
            if label.endswith("center"):
                if label.startswith("U"): predicted_letter = "W" # top is white
                elif label.startswith("R"): predicted_letter = "R" # right is red
                elif label.startswith("F"): predicted_letter = "G" # front is green
                elif label.startswith("D"): predicted_letter = "Y" # bottom is yellow
                elif label.startswith("L"): predicted_letter = "O" # left is orange
                elif label.startswith("B"): predicted_letter = "B" # back is blue
                else: predicted_letter = "?"
                
            # --- READ THE CAMERA FOR EVERYTHING ELSE ---
            else:
                h, s, v = current_hsv[y, x] 
                predicted_letter = classify_hsv_live(h, s, v, hsv_cfg)
            
            # Draw the circles and text
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1)
            cv2.putText(frame, predicted_letter, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            cv2.putText(frame, predicted_letter, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        cv2.imshow('Tuning Feed', frame)

        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('x'):
            with open("hsv_config.json", "w") as f:
                json.dump(hsv_cfg, f, indent=2)
            print("Saved tuned lighting to hsv_config.json")
            break
            
        elif key == ord('c'):
            # BUG 2 & 3 FIXED: Toggle logic aligns with 0 and 1, and reloads coordinates!
            if CAMERA_KEY == "cam1":
                CAMERA_KEY = "cam2"
                CAMERA_INDEX = 0 
            else:
                CAMERA_KEY = "cam1"
                CAMERA_INDEX = 1 
                
            # Update the coordinate map to match the new camera!
            pos_data = all_pos_data[CAMERA_KEY] 
            
            print(f"Switched to {CAMERA_KEY} (Index {CAMERA_INDEX})")
            
            cap.release()
            cap = cv2.VideoCapture(CAMERA_INDEX)
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
