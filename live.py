import cv2
import json
import sys
import numpy as np
from typing import Dict, List, Tuple

WINDOW_NAME = "Cube Live Overlay"
DEFAULT_SAMPLE_RADIUS = 6


def bgr_to_lab(bgr: List[int]):
    pixel = np.array([[bgr]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)
    return lab[0][0].astype(float)


def classify_color_from_bgr(avg_bgr: List[int], refs: Dict[str, Dict[str, object]]) -> str:
    if not refs:
        return "?"

    sample_lab = bgr_to_lab(avg_bgr)

    best_label = "?"
    best_dist = float("inf")

    for label, entry in refs.items():
        ref_bgr = entry.get("avg_bgr")
        if not isinstance(ref_bgr, list) or len(ref_bgr) != 3:
            continue

        ref_lab = bgr_to_lab(ref_bgr)
        dist = ((sample_lab - ref_lab) ** 2).sum() ** 0.5

        if dist < best_dist:
            best_dist = dist
            best_label = label

    return best_label[:1].upper() if best_label else "?"


def load_calibration(calibration_path: str, section_name: str = "first_image") -> Tuple[Dict[str, Dict[str, object]], Dict[str, Dict[str, object]]]:
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    section = data.get(section_name, {})
    stickers = section.get("stickers", {})
    if not stickers:
        raise ValueError(f"No sticker data found for section '{section_name}'")

    refs: Dict[str, Dict[str, object]] = {}
    for key in ["first_image", "second_image"]:
        image_section = data.get(key, {})
        for color_name, color_data in image_section.get("colors", {}).items():
            refs[color_name] = color_data

    if not refs:
        raise ValueError("No color references found in calibration file")

    return stickers, refs


def sample_patch_average(frame, x: int, y: int, radius: int) -> List[int]:
    h, w = frame.shape[:2]
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return [0, 0, 0]

    mean = patch.mean(axis=(0, 1))
    return [int(round(v)) for v in mean.tolist()]


def run_live_overlay(calibration_path: str, camera_index: int = 0, section_name: str = "first_image") -> None:
    stickers, refs = load_calibration(calibration_path, section_name)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    print(f"Using camera index: {camera_index}")
    print(f"Using calibration section: {section_name}")
    print("Press Q or Esc to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from camera.")
            break

        display = frame.copy()

        for label, point in stickers.items():
            x = int(point["x"])
            y = int(point["y"])
            r = int(point.get("sample_radius", DEFAULT_SAMPLE_RADIUS))

            avg_bgr = sample_patch_average(frame, x, y, r)
            color_letter = classify_color_from_bgr(avg_bgr, refs)

            cv2.circle(display, (x, y), r, (0, 255, 0), 1)
            cv2.circle(display, (x, y), 2, (0, 255, 0), -1)

            cv2.putText(
                display,
                color_letter,
                (x - 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                color_letter,
                (x - 8, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        title = f"Live Overlay - {section_name} - camera {camera_index}"
        cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 cube_live_overlay.py calibration.json [camera_index] [first_image|second_image]")
        sys.exit(1)

    calibration_path = sys.argv[1]
    camera_index = int(sys.argv[2]) if len(sys.argv) >= 3 else 0
    section_name = sys.argv[3] if len(sys.argv) >= 4 else "first_image"

    if section_name not in ("first_image", "second_image"):
        print("section must be 'first_image' or 'second_image'")
        sys.exit(1)

    run_live_overlay(calibration_path, camera_index, section_name)


if __name__ == "__main__":
    main()
