import cv2
import json
import time
import numpy as np
import kociemba

import serial
import time

DEFAULT_SAMPLE_RADIUS = 6

# Fixed cube orientation / color scheme
COLOR_TO_FACE = {
    "white": "U", "red": "R", "green": "F",
    "yellow": "D", "orange": "L", "blue": "B",
}

KOCIEMBA_FACE_ORDER = ["U", "R", "F", "D", "L", "B"]

FACE_LABEL_ORDER = {
    "U": ["U top-left", "U top-middle", "U top-right", "U middle-left", "U center", "U middle-right", "U bottom-left", "U bottom-middle", "U bottom-right"],
    "R": ["R top-left", "R top-middle", "R top-right", "R middle-left", "R center", "R middle-right", "R bottom-left", "R bottom-middle", "R bottom-right"],
    "F": ["F top-left", "F top-middle", "F top-right", "F middle-left", "F center", "F middle-right", "F bottom-left", "F bottom-middle", "F bottom-right"],
    "D": ["D top-left", "D top-middle", "D top-right", "D middle-left", "D center", "D middle-right", "D bottom-left", "D bottom-middle", "D bottom-right"],
    "L": ["L top-left", "L top-middle", "L top-right", "L middle-left", "L center", "L middle-right", "L bottom-left", "L bottom-middle", "L bottom-right"],
    "B": ["B top-left", "B top-middle", "B top-right", "B middle-left", "B center", "B middle-right", "B bottom-left", "B bottom-middle", "B bottom-right"],
}

def capture_frame(camera_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    time.sleep(1.0) # Let the camera adjust to lighting
    ret, frame = cap.read()
    cap.release()
    
    if not ret or frame is None:
        raise RuntimeError(f"Failed to capture frame from camera {camera_index}")

    # --- FLIP LOGIC ---
    if camera_index == 0:
        frame = cv2.flip(frame, -1)  # Flips Camera 0 vertically

    return frame

def classify_hsv(bgr_pixel, hsv_cfg):
    pixel_np = np.array([[bgr_pixel]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel_np, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv

    if s < hsv_cfg['s_min']: return "white"

    if 0 <= h < hsv_cfg['r_o']: return "red"
    if hsv_cfg['r_o'] <= h < hsv_cfg['o_y']: return "orange"
    if hsv_cfg['o_y'] <= h < hsv_cfg['y_g']: return "yellow"
    if hsv_cfg['y_g'] <= h < hsv_cfg['g_b']: return "green"
    if hsv_cfg['g_b'] <= h < hsv_cfg['b_r']: return "blue"
    if hsv_cfg['b_r'] <= h <= 179: return "red"
    
    return "unknown"

def forced_center_color(label):
    if label == "D center": return "yellow"
    if label == "F center": return "green"
    if label == "R center": return "red"
    if label == "U center": return "white"
    if label == "L center": return "orange"
    if label == "B center": return "blue"
    return "unknown"

def read_stickers(frame, positions, hsv_cfg):
    results = {}
    for label, pt in positions.items():
        x, y = pt['x'], pt['y']
        
        # Grab a small patch around the coordinate to get a stable average
        patch = frame[max(0, y-DEFAULT_SAMPLE_RADIUS):y+DEFAULT_SAMPLE_RADIUS+1, 
                      max(0, x-DEFAULT_SAMPLE_RADIUS):x+DEFAULT_SAMPLE_RADIUS+1]
        avg_bgr = [int(v) for v in patch.mean(axis=(0,1))]
        
        # Centers are hardcoded; everything else uses your sliders
        if label.endswith("center"):
            color = forced_center_color(label)
        else:
            color = classify_hsv(avg_bgr, hsv_cfg)
            
        # Translate the color to a Kociemba letter (U, R, F, D, L, B)
        results[label] = COLOR_TO_FACE.get(color, "?")
    return results


# You may need to change this port depending on your Pi's configuration
# e.g., '/dev/serial0', '/dev/ttyAMA0', or '/dev/ttyUSB0'
UART_PORT = '/dev/serial0' 
BAUD_RATE = 115200

# def send_to_stm32(solution_string):
#     # Split the string "R2 U' F" into a list: ['R2', "U'", 'F']
#     moves = solution_string.split()
    
#     try:
#         # Open the serial port
#         ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
#         time.sleep(2) # Give the connection a moment to initialize
        
#         print("Sending moves to STM32...")
        
#         for move in moves:
#             # Add a newline or delimiter so the STM32 knows the command is complete
#             command = f"{move}\n" 
#             ser.write(command.encode('utf-8'))
            
#             print(f"Sent: {move}")


# # U2 converts to double -> U U 
# # anything prime converts to lowercase -> F' = f
# # remove spaces
# # random character at end symbolizing end of string
# # send something to stm if fails kociemba

#             # Optional: Wait for an "ACK" (acknowledgement) from the STM32 
#             # before sending the next move so you don't overflow the RX buffer
#             while True:
#                 response = ser.readline().decode('utf-8').strip()
#                 if response == "DONE":
#                     break

#         print("All moves sent!")
#         ser.close()
        
#     except serial.SerialException as e:
#         print(f"UART Error: {e}")

# # Usage:
# # solution = "R2 U' F L2 D"
# # send_to_stm32(solution)

def send_to_stm32(solution_string, is_success=True):
    try:
        # Open the serial port
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Give the connection a moment to initialize
        
        # --- KOCIEMBA FAILURE HANDLING ---
        if not is_success:
            print("Sending FAILURE flag to STM32...")
            # Send a specific string so your C code knows to flash a red LED or abort
            ser.write(b"FAIL!\n") 
            ser.close()
            return
            
        # --- FORMATTING THE STRING ---
        moves = solution_string.split()
        formatted_string = ""
        
        for move in moves:
            face = move[0]
            if len(move) == 1:       # Standard move (e.g., 'F')
                formatted_string += face
            elif move[1] == "'":     # Prime move (e.g., 'F') -> 'f'
                formatted_string += face.lower()
            elif move[1] == '2':     # Double move (e.g., 'U2') -> 'UU'
                formatted_string += face + face
                
        # Append the random character at the end (using '!' as the terminator)
        formatted_string += "!"
        
        # --- SENDING THE PAYLOAD ---
        print(f"Sending to STM32: {formatted_string}")
        
        # We add \n so the STM32 UART interrupt knows the transmission is complete
        ser.write((formatted_string + "\n").encode('utf-8')) 
        
        # Wait for an "ACK" from the STM32 after it finishes the ENTIRE sequence
        print("Waiting for STM32 to finish...")
        while True:
            response = ser.readline().decode('utf-8').strip()
            if response == "DONE":
                print("Solve complete!")
                break

        ser.close()
        
    except serial.SerialException as e:
        print(f"UART Error: {e}")



def main():
    print("Loading configuration files...")
    try:
        with open("positions.json", "r") as f: 
            pos_data = json.load(f)
        with open("hsv_config.json", "r") as f: 
            hsv_cfg = json.load(f)
    except Exception as e:
        print(f"Error loading config files. Make sure positions.json and hsv_config.json exist! Error: {e}")
        return

    print("Capturing Camera FRONT...")
    frame1 = capture_frame(1) 
    print("Capturing Camera BACK...")
    frame2 = capture_frame(0) 

    print("Analyzing colors...")
    faces1 = read_stickers(frame1, pos_data['cam1'], hsv_cfg)
    faces2 = read_stickers(frame2, pos_data['cam2'], hsv_cfg)
    
    # Combine the data from both cameras
    all_stickers = {**faces1, **faces2}
    
    # Build Kociemba String in the strict URFDLB order
    cube_string = ""
    for face in KOCIEMBA_FACE_ORDER:
        for label in FACE_LABEL_ORDER[face]:
            cube_string += all_stickers.get(label, "?")
            
    print(f"\nScanned Cube String: {cube_string}")
    
    # Check for unrecognized colors
    if "?" in cube_string:
        print("ERROR: Some stickers were unreadable.")
        send_to_stm32("", is_success=False) # <--- SEND FAILURE
    else:
        print("Validating and Solving with Kociemba...")
        try:
            solution = kociemba.solve(cube_string)
            print(f"\n>>> SUCCESS! Solution: {solution} <<<")
            send_to_stm32(solution, is_success=True) # <--- SEND SUCCESS
        except Exception as e:
            print(f"Kociemba failed: {e}")
            send_to_stm32("", is_success=False) # <--- SEND FAILURE

if __name__ == "__main__":
    main()
