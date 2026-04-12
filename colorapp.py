
import cv2
import numpy as np

# --- CONFIGURATION ---
VIDEO_FILE = 'test2.mp4' 

# Global state
paused = False
current_hsv = None

def nothing(x): pass

def main():
    global paused, current_hsv
    cv2.startWindowThread() 
    
    # 1. Setup the Windows
    cv2.namedWindow('Calibration Feed')
    cv2.namedWindow('Color Segments') # This shows all colors at once

    # 2. Create Sliders matching your screenshot
    # Column 1: Hue Transitions (0-179)
    cv2.createTrackbar('red-orange', 'Calibration Feed', 10, 179, nothing)
    cv2.createTrackbar('orange-yell', 'Calibration Feed', 25, 179, nothing)
    cv2.createTrackbar('yellow-gree', 'Calibration Feed', 40, 179, nothing)
    cv2.createTrackbar('green-blue', 'Calibration Feed', 90, 179, nothing)
    cv2.createTrackbar('blue-red', 'Calibration Feed', 150, 179, nothing)

    # Column 2: Filters (0-255)
    cv2.createTrackbar('black-filter', 'Calibration Feed', 50, 255, nothing) # Value Min
    cv2.createTrackbar('white-filter', 'Calibration Feed', 60, 255, nothing) # Saturation Min
    cv2.createTrackbar('val-max', 'Calibration Feed', 255, 255, nothing)     # Brightness Max

    cap = cv2.VideoCapture(0)
    
    frame = cap.read()
    while True:
        
        
                
        current_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 3. Read the border values from sliders
        r_o = cv2.getTrackbarPos('red-orange', 'Calibration Feed')
        o_y = cv2.getTrackbarPos('orange-yell', 'Calibration Feed')
        y_g = cv2.getTrackbarPos('yellow-gree', 'Calibration Feed')
        g_b = cv2.getTrackbarPos('green-blue', 'Calibration Feed')
        b_r = cv2.getTrackbarPos('blue-red', 'Calibration Feed')
        
        v_min = cv2.getTrackbarPos('black-filter', 'Calibration Feed')
        s_min = cv2.getTrackbarPos('white-filter', 'Calibration Feed')
        v_max = cv2.getTrackbarPos('val-max', 'Calibration Feed')

        # 4. Color Logic: Create masks for each segment
        # We define each color as the space between two sliders
        mask_red    = cv2.inRange(current_hsv, np.array([0, s_min, v_min]), np.array([r_o, 255, v_max]))
        mask_orange = cv2.inRange(current_hsv, np.array([r_o, s_min, v_min]), np.array([o_y, 255, v_max]))
        mask_yellow = cv2.inRange(current_hsv, np.array([o_y, s_min, v_min]), np.array([y_g, 255, v_max]))
        mask_green  = cv2.inRange(current_hsv, np.array([y_g, s_min, v_min]), np.array([g_b, 255, v_max]))
        mask_blue   = cv2.inRange(current_hsv, np.array([g_b, s_min, v_min]), np.array([b_r, 255, v_max]))
        mask_red2   = cv2.inRange(current_hsv, np.array([b_r, s_min, v_min]), np.array([179, 255, v_max]))
        
        # White is special: It's anything with very low saturation
        mask_white  = cv2.inRange(current_hsv, np.array([0, 0, v_min]), np.array([179, s_min, 255]))

        # 5. Visualizer: Colorize the segments so you can see them all at once!
        # This creates a "Heatmap" where each color is clearly separated
        output = np.zeros_like(frame)
        output[mask_red > 0]    = (0, 0, 255)    # Red
        output[mask_orange > 0] = (0, 165, 255)  # Orange
        output[mask_yellow > 0] = (0, 255, 255)  # Yellow
        output[mask_green > 0]  = (0, 255, 0)    # Green
        output[mask_blue > 0]   = (255, 0, 0)    # Blue
        output[mask_red2 > 0]   = (0, 0, 255)    # Red (wrap-around)
        output[mask_white > 0]  = (255, 255, 255)# White

        cv2.imshow('Calibration Feed', frame)
        cv2.imshow('Color Segments', output)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('x'):
            print(f"Borders: R-O:{r_o}, O-Y:{o_y}, Y-G:{y_g}, G-B:{g_b}, B-R:{b_r}")
            break
        elif key == ord(' '):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
