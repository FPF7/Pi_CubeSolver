import cv2
import json
import numpy as np
import kociemba
import serial
import time

COLOR_NAME_TO_FACE = {
    "white": "U",
    "red": "R",
    "green": "F",
    "yellow": "D",
    "orange": "L",
    "blue": "B",
}

COLOR_NAME_TO_LETTER = {
    "white": "w",
    "red": "r",
    "green": "g",
    "yellow": "y",
    "orange": "o",
    "blue": "b",
}

FIXED_FACE_COLORS = {
    "F": "green",
    "U": "white",
    "B": "blue",
    "R": "red",
    "L": "orange",
    "D": "yellow",
}

# Based on your current calibration grouping:
# camera index 0 -> F R U
# camera index 1 -> B D L
FRU_CAMERA_INDEX = 0
BDL_CAMERA_INDEX = 2

def convert_uart_moves(sequence: str) -> str:
    result = ""

    moves = sequence.split()

    for move in moves:
        face = move[0]

        # Check for modifiers
        if len(move) == 1:
            result += face
        elif move[1] == '2':
            result += face * 2
        elif move[1] == "'":
            result += face.lower()

    return result
def draw_boxed_text(
    image,
    text,
    org,
    text_color=(255, 255, 255),
    font_scale=0.5,
    thickness=1,
    padding=3,
):
    x, y = org
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )

    cv2.rectangle(
        image,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + baseline + padding),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def capture_single_frame(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    ret, frame = cap.read()
    cap.release()
    if camera_index == FRU_CAMERA_INDEX:
        frame = cv2.flip(frame, -1)

    if not ret or frame is None:
        raise RuntimeError(f"Could not capture frame from camera {camera_index}")

    return frame


def bgr_to_lab(bgr):
    pixel = np.array([[bgr]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    return np.array([float(lab[0]), float(lab[1]), float(lab[2])], dtype=np.float32)


def median_bgr_patch(frame, x: int, y: int, radius: int):
    h, w = frame.shape[:2]

    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return [0, 0, 0]

    median = np.median(patch.reshape(-1, 3), axis=0)
    return [int(round(v)) for v in median.tolist()]


def lab_distance(lab1, lab2):
    return float(np.linalg.norm(lab1 - lab2))


def classify_color_from_lab(sample_lab, color_refs):
    best_color = None
    best_distance = None

    for color_name, ref_lab in color_refs.items():
        dist = lab_distance(sample_lab, ref_lab)
        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_color = color_name

    return best_color, best_distance


def get_face_stickers_in_order(sticker_dict, face_name):
    order = [
        f"{face_name} top-left",
        f"{face_name} top-middle",
        f"{face_name} top-right",
        f"{face_name} middle-left",
        f"{face_name} center",
        f"{face_name} middle-right",
        f"{face_name} bottom-left",
        f"{face_name} bottom-middle",
        f"{face_name} bottom-right",
    ]

    missing = [label for label in order if label not in sticker_dict]
    if missing:
        raise KeyError(
            f"Missing labels for face {face_name}: {missing}\n"
            f"Available labels: {list(sticker_dict.keys())}"
        )

    return [sticker_dict[label] for label in order]


def rotate_face_180(facelets):
    if len(facelets) != 9:
        raise ValueError(f"Expected 9 facelets, got {len(facelets)}")

    return [
        facelets[8], facelets[7], facelets[6],
        facelets[5], facelets[4], facelets[3],
        facelets[2], facelets[1], facelets[0],
    ]


def rotate_point_180(x: int, y: int, width: int, height: int):
    return width - 1 - x, height - 1 - y


def maybe_rotate_point_from_json(x: int, y: int, frame, invert: bool):
    if not invert:
        return x, y

    h, w = frame.shape[:2]
    return rotate_point_180(x, y, w, h)


def load_color_refs(colors_json_path):
    with open(colors_json_path, "r", encoding="utf-8") as f:
        colors_data = json.load(f)

    merged = colors_data["merged"]

    color_refs = {}
    for color_name in COLOR_NAME_TO_FACE.keys():
        median_lab = merged[color_name]["median_lab"]
        if median_lab is None:
            raise ValueError(f"Missing median_lab for color {color_name} in colors.json")
        color_refs[color_name] = np.array(median_lab, dtype=np.float32)

    return color_refs


def classify_face(frame, sticker_dict, face_name, color_refs, invert=False):
    sticker_positions = get_face_stickers_in_order(sticker_dict, face_name)

    color_names = []
    face_letters = []
    debug_info = []

    for idx, pos in enumerate(sticker_positions):
        json_x = int(pos["x"])
        json_y = int(pos["y"])
        radius = int(pos.get("sample_radius", 1))

        x, y = maybe_rotate_point_from_json(json_x, json_y, frame, invert)

        median_bgr = median_bgr_patch(frame, x, y, radius)
        sample_lab = bgr_to_lab(median_bgr)

        best_color, best_distance = classify_color_from_lab(sample_lab, color_refs)

        if idx == 4:
            best_color = FIXED_FACE_COLORS[face_name]

        face_letter = COLOR_NAME_TO_FACE[best_color]

        color_names.append(best_color)
        face_letters.append(face_letter)

        debug_info.append({
            "index": idx,
            "json_x": json_x,
            "json_y": json_y,
            "sample_x": x,
            "sample_y": y,
            "sample_radius": radius,
            "median_bgr": median_bgr,
            "sample_lab": [float(v) for v in sample_lab.tolist()],
            "classified_color": best_color,
            "face_letter": face_letter,
            "distance": float(best_distance),
        })

    return color_names, face_letters, debug_info


def save_labeled_image(frame, sticker_dict, faces, debug_by_face, output_path, invert=False):
    display = frame.copy()
    h, w = display.shape[:2]

    for face in faces:
        face_debug = debug_by_face[face]
        positions = get_face_stickers_in_order(sticker_dict, face)

        for idx, pos in enumerate(positions):
            json_x = int(pos["x"])
            json_y = int(pos["y"])
            radius = int(pos.get("sample_radius", 1))

            draw_x, draw_y = maybe_rotate_point_from_json(json_x, json_y, frame, invert)

            info = face_debug[idx]
            color_name = info["classified_color"]
            letter = COLOR_NAME_TO_LETTER[color_name]

            cv2.circle(display, (draw_x, draw_y), radius, (0, 255, 0), 1)
            cv2.circle(display, (draw_x, draw_y), 2, (0, 255, 0), -1)

            draw_boxed_text(
                display,
                letter,
                (draw_x + 6, draw_y - 8),
                text_color=(255, 255, 255),
                font_scale=0.7,
                thickness=2,
                padding=3,
            )

            draw_boxed_text(
                display,
                str(idx + 1),
                (draw_x + 6, draw_y + 18),
                text_color=(180, 255, 180),
                font_scale=0.45,
                thickness=1,
                padding=2,
            )

            if idx == 4:
                draw_boxed_text(
                    display,
                    face,
                    (draw_x - 10, draw_y - 14),
                    text_color=(0, 255, 255),
                    font_scale=0.55,
                    thickness=2,
                    padding=2,
                )

    cv2.imwrite(output_path, display)

def kociemba_to_faces(k_string):
    # Mapping faces → slices
    face_ranges = {
        "U": (0, 9),
        "R": (9, 18),
        "F": (18, 27),
        "D": (27, 36),
        "L": (36, 45),
        "B": (45, 54),
    }

    # Convert face letter → color name
    result = {}

    for face, (start, end) in face_ranges.items():
        stickers = k_string[start:end]
        colors = [FIXED_FACE_COLORS[c] for c in stickers]
        result[face] = colors

    return result
def cube_to_kociemba(position_json_path="position.json", colors_json_path="colors.json"):
    with open(position_json_path, "r", encoding="utf-8") as f:
        position_data = json.load(f)

    color_refs = load_color_refs(colors_json_path)

    sticker_sets = {}
    invert_flags = {}

    for block_name in ["camera_1", "camera_2"]:
        block = position_data[block_name]
        cam_idx = int(block["camera_index"])
        sticker_sets[cam_idx] = block["stickers"]
        invert_flags[cam_idx] = bool(block.get("invert", False))

    if FRU_CAMERA_INDEX not in sticker_sets:
        raise ValueError(f"No sticker set found for camera {FRU_CAMERA_INDEX}")
    if BDL_CAMERA_INDEX not in sticker_sets:
        raise ValueError(f"No sticker set found for camera {BDL_CAMERA_INDEX}")

    frame_fru = capture_single_frame(FRU_CAMERA_INDEX)
    stickers_fru = sticker_sets[FRU_CAMERA_INDEX]
    invert_fru = invert_flags[FRU_CAMERA_INDEX]

    frame_bdl = capture_single_frame(BDL_CAMERA_INDEX)
    stickers_bdl = sticker_sets[BDL_CAMERA_INDEX]
    invert_bdl = invert_flags[BDL_CAMERA_INDEX]

    cv2.imwrite("cam0_raw.png", frame_fru)
    cv2.imwrite("cam1_raw.png", frame_bdl)

    classified_color_names = {}
    classified_face_letters = {}
    debug = {}

    for face in ["F", "R", "U"]:
        color_names, face_letters, debug_info = classify_face(
            frame_fru, stickers_fru, face, color_refs, invert=invert_fru
        )
        classified_color_names[face] = color_names
        classified_face_letters[face] = face_letters
        debug[face] = debug_info

    for face in ["B", "D", "L"]:
        color_names, face_letters, debug_info = classify_face(
            frame_bdl, stickers_bdl, face, color_refs, invert=invert_bdl
        )
        classified_color_names[face] = color_names
        classified_face_letters[face] = face_letters
        debug[face] = debug_info

    save_labeled_image(
        frame=frame_fru,
        sticker_dict=stickers_fru,
        faces=["F", "R", "U"],
        debug_by_face=debug,
        output_path="cam0_labeled.png",
        invert=invert_fru,
    )

    save_labeled_image(
        frame=frame_bdl,
        sticker_dict=stickers_bdl,
        faces=["B", "D", "L"],
        debug_by_face=debug,
        output_path="cam1_labeled.png",
        invert=invert_bdl,
    )

    # If a camera was calibrated/viewed inverted, the face order seen from that camera
    # is reversed relative to the physical cube, so rotate those faces 180 degrees
    # before building the Kociemba string.
    if invert_fru:
        for face in ["F", "R", "U"]:
            classified_color_names[face] = rotate_face_180(classified_color_names[face])
            classified_face_letters[face] = rotate_face_180(classified_face_letters[face])
            debug[face] = rotate_face_180(debug[face])

    if invert_bdl:
        for face in ["B", "D", "L"]:
            classified_color_names[face] = rotate_face_180(classified_color_names[face])
            classified_face_letters[face] = rotate_face_180(classified_face_letters[face])
            debug[face] = rotate_face_180(debug[face])

    cube_string = "".join(
        "".join(classified_face_letters[face])
        for face in ["U", "R", "F", "D", "L", "B"]
    )

    if len(cube_string) != 54:
        raise ValueError(f"Kociemba string must be 54 characters, got {len(cube_string)}")

    valid_letters = set("URFDLB")
    bad_chars = [c for c in cube_string if c not in valid_letters]
    if bad_chars:
        raise ValueError(f"Invalid letters in cube string: {bad_chars}")

    counts = {c: cube_string.count(c) for c in "URFDLB"}
    for c, n in counts.items():
        if n != 9:
            raise ValueError(f"Face letter {c} appears {n} times, expected 9")

    result = {
        "cube_string": cube_string,
        "classified_color_names": classified_color_names,
        "classified_face_letters": classified_face_letters,
        "debug": debug,
        "invert_flags": invert_flags,
        "saved_images": [
            "cam0_raw.png",
            "cam1_raw.png",
            "cam0_labeled.png",
            "cam1_labeled.png",
        ],
    }

    return result

def transform(s):
    result = list(s)

    for start in [0, 9, 18]:  # blocks: 1–9, 10–18, 19–27
        result[start:start+9] = reversed(result[start:start+9])

    return ''.join(result)
def final():
    result = cube_to_kociemba("position.json", "colors.json")

    print("\nKociemba string:")
    kociemba_string = transform(result["cube_string"])
    print(kociemba_string)
    face_dict = (kociemba_to_faces(kociemba_string))
    print("Per-face classified colors:")
    for face in ["U", "R", "F", "D", "L", "B"]:
        print(f"{face} {face_dict[face]}")
    solve = kociemba.solve(kociemba_string)
    if kociemba_string == "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB": 
        stm_solve = "SOLVED"
    else:
        stm_solve = convert_uart_moves(solve)
        #put size of string at the beginning of it
        stm_solve = f"{len(stm_solve)}{stm_solve}"
    return stm_solve


if __name__ == "__main__":
        ser = serial.Serial('/dev/serial0', 115200, timeout=1)
        print("Waiting for STM32 to be ready...")
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if line == "Start":
                print("STM32 is ready. Starting cube scanning and solving...")
                try:
                    uart = final()
                except ValueError as e:
                    print(f"ValueError occurred: {e}")
                    uart = "E"   # 👈 send error signal

                except Exception as e:
                    print(f"Unexpected error: {e}")
                    uart = "E"   # 👈 catch EVERYTHING else too
                print(f"Sending solution to STM32: {uart}")
                ser.write(uart.encode())
                print("Solution sent")
                break
            else:
                print(f"Received from STM32: {line}")