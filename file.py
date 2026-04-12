import cv2
import json
import sys
import numpy as np
from typing import Dict, List, Tuple

DEFAULT_SAMPLE_RADIUS = 6


def bgr_to_lab(bgr: List[int]) -> np.ndarray:
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
        dist = float(np.linalg.norm(sample_lab - ref_lab))
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


def overlay_colors_on_image(calibration_path: str, image_path: str, output_path: str, section_name: str = "first_image") -> None:
    stickers, refs = load_calibration(calibration_path, section_name)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    display = image.copy()

    for label, point in stickers.items():
        x = int(point["x"])
        y = int(point["y"])
        r = int(point.get("sample_radius", DEFAULT_SAMPLE_RADIUS))

        avg_bgr = sample_patch_average(image, x, y, r)
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

    title = f"Overlay - {section_name}"
    cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1, cv2.LINE_AA)

    ok = cv2.imwrite(output_path, display)
    if not ok:
        raise RuntimeError(f"Failed to write output image: {output_path}")

    print(f"Saved overlay image to {output_path}")


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python3 cube_image_overlay.py calibration.json input.jpg [output.jpg] [first_image|second_image]")
        sys.exit(1)

    calibration_path = sys.argv[1]
    image_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) >= 4 else "overlay_output.jpg"
    section_name = sys.argv[4] if len(sys.argv) >= 5 else "first_image"

    if section_name not in ("first_image", "second_image"):
        print("section must be 'first_image' or 'second_image'")
        sys.exit(1)

    overlay_colors_on_image(calibration_path, image_path, output_path, section_name)


if __name__ == "__main__":
    main()
