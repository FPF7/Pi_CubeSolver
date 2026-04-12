import cv2
import json
import sys
import time
import numpy as np
from typing import Dict, List, Tuple
import kociemba

DEFAULT_SAMPLE_RADIUS = 6

# Fixed cube orientation / color scheme
COLOR_TO_FACE = {
    "white": "U",
    "red": "R",
    "green": "F",
    "yellow": "D",
    "orange": "L",
    "blue": "B",
}

KOCIEMBA_FACE_ORDER = ["U", "R", "F", "D", "L", "B"]

FACE_LABEL_ORDER = {
    "U": [
        "U top-left", "U top-middle", "U top-right",
        "U middle-left", "U center", "U middle-right",
        "U bottom-left", "U bottom-middle", "U bottom-right",
    ],
    "R": [
        "R top-left", "R top-middle", "R top-right",
        "R middle-left", "R center", "R middle-right",
        "R bottom-left", "R bottom-middle", "R bottom-right",
    ],
    "F": [
        "F top-left", "F top-middle", "F top-right",
        "F middle-left", "F center", "F middle-right",
        "F bottom-left", "F bottom-middle", "F bottom-right",
    ],
    "D": [
        "D top-left", "D top-middle", "D top-right",
        "D middle-left", "D center", "D middle-right",
        "D bottom-left", "D bottom-middle", "D bottom-right",
    ],
    "L": [
        "L top-left", "L top-middle", "L top-right",
        "L middle-left", "L center", "L middle-right",
        "L bottom-left", "L bottom-middle", "L bottom-right",
    ],
    "B": [
        "B top-left", "B top-middle", "B top-right",
        "B middle-left", "B center", "B middle-right",
        "B bottom-left", "B bottom-middle", "B bottom-right",
    ],
}


def bgr_to_lab(bgr: List[int]) -> np.ndarray:
    pixel = np.array([[bgr]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return lab[0][0].astype(float)


def sample_patch_average(image: np.ndarray, x: int, y: int, radius: int) -> List[int]:
    h, w = image.shape[:2]
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return [0, 0, 0]

    mean = patch.mean(axis=(0, 1))
    return [int(round(v)) for v in mean.tolist()]


def capture_frame(camera_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    time.sleep(1.0)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Failed to capture frame from camera {camera_index}")

    return frame


def load_calibration(calibration_path: str) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    refs: Dict[str, Dict[str, object]] = {}
    for key in ["first_image", "second_image"]:
        image_section = data.get(key, {})
        for color_name, color_data in image_section.get("colors", {}).items():
            refs[color_name] = color_data

    if not refs:
        raise ValueError("No color references found in calibration file")

    return data, refs


def classify_color_name(avg_bgr: List[int], refs: Dict[str, Dict[str, object]]) -> str:
    sample_lab = bgr_to_lab(avg_bgr)

    best_label = "unknown"
    best_dist = float("inf")

    for label, entry in refs.items():
        ref_bgr = entry.get("avg_bgr")
        if not isinstance(ref_bgr, list) or len(ref_bgr) != 3:
            continue

        ref_lab = bgr_to_lab(ref_bgr)
        dist = float(np.linalg.norm(sample_lab - ref_lab))
        if dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label


def forced_center_color(sticker_label: str) -> str:
    if sticker_label == "D center":
        return "yellow"
    if sticker_label == "F center":
        return "green"
    if sticker_label == "R center":
        return "red"
    if sticker_label == "U center":
        return "white"
    if sticker_label == "L center":
        return "orange"
    if sticker_label == "B center":
        return "blue"
    return "unknown"


def read_section_labels(
    image: np.ndarray,
    sticker_points: Dict[str, Dict[str, object]],
    refs: Dict[str, Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}

    for sticker_label, point in sticker_points.items():
        x = int(point["x"])
        y = int(point["y"])
        r = int(point.get("sample_radius", DEFAULT_SAMPLE_RADIUS))

        avg_bgr = sample_patch_average(image, x, y, r)

        if sticker_label.endswith("center"):
            color_name = forced_center_color(sticker_label)
        else:
            color_name = classify_color_name(avg_bgr, refs)

        face_letter = COLOR_TO_FACE.get(color_name, "?")

        results[sticker_label] = {
            "color_name": color_name,
            "face_letter": face_letter,
            "avg_bgr": avg_bgr,
            "x": x,
            "y": y,
            "sample_radius": r,
        }

    return results


def save_annotated_image(
    image: np.ndarray,
    sticker_results: Dict[str, Dict[str, object]],
    output_path: str,
    title: str,
) -> None:
    annotated = image.copy()

    for sticker_label, entry in sticker_results.items():
        x = int(entry["x"])
        y = int(entry["y"])
        r = int(entry.get("sample_radius", DEFAULT_SAMPLE_RADIUS))
        color_name = str(entry.get("color_name", "unknown"))
        face_letter = str(entry.get("face_letter", "?"))
        text = color_name[:1].upper()

        cv2.circle(annotated, (x, y), r, (0, 255, 0), 1)
        cv2.circle(annotated, (x, y), 2, (0, 255, 0), -1)

        cv2.putText(
            annotated,
            text,
            (x - 8, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            text,
            (x - 8, y + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

        small_text = f"{sticker_label}={color_name}/{face_letter}"
        cv2.putText(
            annotated,
            small_text,
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            small_text,
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(annotated, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(annotated, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    ok = cv2.imwrite(output_path, annotated)
    if not ok:
        raise RuntimeError(f"Failed to write annotated image: {output_path}")


def assemble_faces(all_stickers: Dict[str, Dict[str, object]]) -> Dict[str, List[str]]:
    faces: Dict[str, List[str]] = {}

    for face, labels in FACE_LABEL_ORDER.items():
        values: List[str] = []
        for label in labels:
            if label not in all_stickers:
                raise ValueError(f"Missing sticker label: {label}")
            values.append(all_stickers[label]["face_letter"])
        faces[face] = values

    return faces


def build_kociemba_string(faces: Dict[str, List[str]]) -> str:
    return "".join(
        face_letter
        for face in KOCIEMBA_FACE_ORDER
        for face_letter in faces[face]
    )


def validate_cube_string(cube_string: str) -> List[str]:
    errors: List[str] = []

    if len(cube_string) != 54:
        errors.append(f"Cube string length is {len(cube_string)}, expected 54")

    for face_letter in "URFDLB":
        count = cube_string.count(face_letter)
        if count != 9:
            errors.append(f"Face letter {face_letter} appears {count} times, expected 9")

    if "?" in cube_string:
        errors.append("Cube string contains '?' labels, so some colors were not mapped")

    return errors


def maybe_solve_kociemba(cube_string: str) -> str:
    try:
        return kociemba.solve(cube_string)
    except Exception as e:
        return f"solve failed: {e}"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 cube_to_kociemba.py calibration.json [output.json] [first_annotated.png] [second_annotated.png]")
        sys.exit(1)

    calibration_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) >= 3 else "cube_state.json"
    first_annotated_path = sys.argv[3] if len(sys.argv) >= 4 else "first_annotated.png"
    second_annotated_path = sys.argv[4] if len(sys.argv) >= 5 else "second_annotated.png"

    calibration, refs = load_calibration(calibration_path)

    print("Capturing from camera 2...")
    first_image = capture_frame(1)

    print("Capturing from camera 3...")
    second_image = capture_frame(2)

    cv2.imwrite("cam2_capture.png", first_image)
    cv2.imwrite("cam3_capture.png", second_image)

    first_points = calibration.get("first_image", {}).get("stickers", {})
    second_points = calibration.get("second_image", {}).get("stickers", {})

    if not first_points:
        raise ValueError("No sticker points found in calibration for first_image")
    if not second_points:
        raise ValueError("No sticker points found in calibration for second_image")

    first_results = read_section_labels(first_image, first_points, refs)
    second_results = read_section_labels(second_image, second_points, refs)

    save_annotated_image(first_image, first_results, first_annotated_path, "First image classified")
    save_annotated_image(second_image, second_results, second_annotated_path, "Second image classified")

    all_stickers = {}
    all_stickers.update(first_results)
    all_stickers.update(second_results)

    faces = assemble_faces(all_stickers)
    cube_string = build_kociemba_string(faces)
    validation_errors = validate_cube_string(cube_string)
    solve_result = maybe_solve_kociemba(cube_string) if not validation_errors else "skipped due to validation errors"

    result = {
        "faces": faces,
        "cube_string": cube_string,
        "validation_errors": validation_errors,
        "solve_result": solve_result,
        "stickers": all_stickers,
        "color_to_face": COLOR_TO_FACE,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("Faces:")
    for face in KOCIEMBA_FACE_ORDER:
        print(f"{face}: {' '.join(faces[face])}")

    print(f"\nCube string: {cube_string}")
    if validation_errors:
        print("\nValidation errors:")
        for err in validation_errors:
            print(f"- {err}")
    else:
        print("\nCube string passed basic validation.")

    print(f"\nKociemba result: {solve_result}")
    print(f"Saved detailed output to {output_path}")
    print(f"Saved annotated first image to {first_annotated_path}")
    print(f"Saved annotated second image to {second_annotated_path}")
    print("Saved raw captures to cam2_capture.png and cam3_capture.png")


if __name__ == "__main__":
    main()