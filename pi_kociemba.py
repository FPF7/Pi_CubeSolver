import cv2
import json
import time
import numpy as np
import kociemba
import serial

CUBE_JSON = "cube.json"
UART_PORT = '/dev/serial0'
BAUD_RATE = 115200
RECALIBRATE = True  # Set True first run on Pi, False after

COLOR_NAME_TO_FACE = {
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

CENTER_COLORS = {
    "U center": "U", "R center": "R", "F center": "F",
    "D center": "D", "L center": "L", "B center": "B",
}

def closest_color(bgr, color_refs):
    bgr = np.array(bgr, dtype=float)
    best_name = None
    best_dist = float('inf')
    for name, ref_bgr in color_refs.items():
        dist = np.linalg.norm(bgr - np.array(ref_bgr, dtype=float))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return COLOR_NAME_TO_FACE.get(best_name, "?")

def capture_frame(camera_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    time.sleep(1.0)
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        raise RuntimeError(f"Failed to capture frame from camera {camera_index}")
    return frame

def read_stickers(frame, stickers, color_refs):
    results = {}
    for label, pt in stickers.items():
        if label in CENTER_COLORS:
            results[label] = CENTER_COLORS[label]
            continue
        x, y = pt['x'], pt['y']
        r = pt.get('sample_radius', 6)
        h, w = frame.shape[:2]
        patch = frame[max(0,y-r):min(h,y+r+1), max(0,x-r):min(w,x+r+1)]
        if patch.size == 0:
            results[label] = "?"
            continue
        avg_bgr = [int(v) for v in patch.mean(axis=(0,1))]
        results[label] = closest_color(avg_bgr, color_refs)
    return results

def recalibrate_colors(frame1, frame2, data):
    color_refs = {}
    for img_key, frame in [("first_image", frame1), ("second_image", frame2)]:
        colors = data[img_key].get("colors", {})
        if not colors:
            continue
        display = frame.copy()
        pending = list(colors.keys())
        clicked = {}

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and pending:
                name = pending[0]
                r = colors[name].get("sample_radius", 6)
                h, w = frame.shape[:2]
                patch = frame[max(0,y-r):min(h,y+r+1), max(0,x-r):min(w,x+r+1)]
                avg = [int(v) for v in patch.mean(axis=(0,1))]
                clicked[name] = avg
                print(f"Updated {name}: BGR={avg}")
                pending.pop(0)

        cv2.namedWindow(f"Recalibrate - {img_key}")
        cv2.setMouseCallback(f"Recalibrate - {img_key}", on_click)

        for name, pt in colors.items():
            cv2.circle(display, (pt['x'], pt['y']), 8, (0,255,0), 2)
            cv2.putText(display, name, (pt['x']+10, pt['y']),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        print(f"\nClick colors in order: {list(colors.keys())}")

        while pending:
            img = display.copy()
            cv2.putText(img, f"Click: {pending[0]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.imshow(f"Recalibrate - {img_key}", img)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        for name, avg in clicked.items():
            color_refs[name] = avg
        for name, pt in colors.items():
            if name not in color_refs:
                color_refs[name] = pt['avg_bgr']

    return color_refs

def send_to_stm32(solution_string, is_success=True):
    TIMEOUT_SECONDS = 60
    try:
        ser = serial.Serial(UART_PORT, BAUD_RATE, timeout=1)
        time.sleep(1)
        print("UART Port Opened Successfully!")

        if not is_success:
            print("Sending FAIL to STM32...")
            ser.write(b"FAIL!\n")
            ser.flush()
            ser.close()
            return

        moves = solution_string.split()
        formatted = ""
        for move in moves:
            face = move[0]
            if len(move) == 1:
                formatted += face
            elif move[1] == "'":
                formatted += face.lower()
            elif move[1] == '2':
                formatted += face + face
        formatted += "!"

        print(f"Pi → STM32: {formatted}")
        ser.write((formatted + "\n").encode('utf-8'))
        ser.flush()

        print(f"Waiting up to {TIMEOUT_SECONDS}s for STM32 to finish...")
        start = time.time()
        while time.time() - start < TIMEOUT_SECONDS:
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8').strip()
                print(f"STM32 → Pi: '{response}'")
                if response.upper() == "DONE":
                    print("Solve complete!")
                    break
            time.sleep(0.05)
        else:
            print("TIMEOUT: STM32 did not finish in time.")

        ser.close()

    except serial.SerialException as e:
        print(f"UART Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    print("Loading cube.json...")
    try:
        with open(CUBE_JSON, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {CUBE_JSON}: {e}")
        return

    first  = data.get("first_image", {})
    second = data.get("second_image", {})

    cam0_idx = second.get("camera_index", 0)
    cam1_idx = first.get("camera_index", 1)

    print(f"Capturing camera {cam1_idx} (first_image)...")
    frame1 = capture_frame(cam1_idx)
    print(f"Capturing camera {cam0_idx} (second_image)...")
    frame2 = capture_frame(cam0_idx)

    # --- RECALIBRATE or load existing color refs ---
    if RECALIBRATE:
        color_refs = recalibrate_colors(frame1, frame2, data)
        for name in color_refs:
            if name in data["first_image"]["colors"]:
                data["first_image"]["colors"][name]["avg_bgr"] = color_refs[name]
            if name in data["second_image"]["colors"]:
                data["second_image"]["colors"][name]["avg_bgr"] = color_refs[name]
        with open(CUBE_JSON, "w") as f:
            json.dump(data, f, indent=2)
        print("Saved updated color refs to cube.json — set RECALIBRATE=False for next run")
    else:
        color_refs = {}
        for name, pt in first.get("colors", {}).items():
            color_refs[name] = pt['avg_bgr']
        for name, pt in second.get("colors", {}).items():
            color_refs[name] = pt['avg_bgr']

    print("Analyzing colors...")
    stickers1 = read_stickers(frame1, first.get("stickers", {}), color_refs)
    stickers2 = read_stickers(frame2, second.get("stickers", {}), color_refs)
    all_stickers = {**stickers1, **stickers2}

    cube_string = ""
    for face in KOCIEMBA_FACE_ORDER:
        for label in FACE_LABEL_ORDER[face]:
            cube_string += all_stickers.get(label, "?")

    print(f"\nScanned Cube String: {cube_string}")

    if "?" in cube_string:
        print("ERROR: Some stickers were unreadable.")
        send_to_stm32("", is_success=False)
    else:
        print("Solving with Kociemba...")
        try:
            solution = kociemba.solve(cube_string)
            print(f"\n>>> SUCCESS! Solution: {solution} <<<")
            send_to_stm32(solution, is_success=True)
        except Exception as e:
            print(f"Kociemba failed: {e}")
            send_to_stm32("", is_success=False)

if __name__ == "__main__":
    main()