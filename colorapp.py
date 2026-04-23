import cv2
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

WINDOW_NAME = "Color Calibration"
DEFAULT_RADIUS = 6

# Click as many samples as you want for each color.
# Press N to move to the next color.
COLOR_ORDER = ["white", "red", "green", "yellow", "orange", "blue"]


def draw_boxed_text(
    image,
    text: str,
    org: Tuple[int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_scale: float = 0.6,
    thickness: int = 1,
    padding: int = 3,
) -> None:
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
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    time.sleep(1.0)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Could not capture frame from camera {camera_index}")

    return frame


def bgr_to_lab(bgr: List[int]) -> List[float]:
    pixel = np.array([[bgr]], dtype=np.uint8)
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    return [float(lab[0]), float(lab[1]), float(lab[2])]


def median_bgr_patch(frame, x: int, y: int, radius: int) -> List[int]:
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


class ColorCalibration:
    def __init__(self, frame, name: str, camera_index: int):
        self.frame = frame
        self.name = name
        self.camera_index = camera_index

        self.radius = DEFAULT_RADIUS
        self.color_index = 0

        self.samples: Dict[str, List[Dict[str, object]]] = {
            color: [] for color in COLOR_ORDER
        }

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.click)

    def get_current_color(self) -> Optional[str]:
        if self.color_index < len(COLOR_ORDER):
            return COLOR_ORDER[self.color_index]
        return None

    def click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        current_color = self.get_current_color()
        if current_color is None:
            return

        median_bgr = median_bgr_patch(self.frame, x, y, self.radius)
        lab = bgr_to_lab(median_bgr)

        sample = {
            "x": int(x),
            "y": int(y),
            "sample_radius": int(self.radius),
            "median_bgr": median_bgr,
            "lab": lab,
        }

        self.samples[current_color].append(sample)

        print(
            f"{self.name} -> {current_color}: "
            f"({x}, {y}) BGR={median_bgr} LAB={[round(v, 2) for v in lab]}"
        )

    def next_color(self):
        if self.color_index < len(COLOR_ORDER) - 1:
            self.color_index += 1
            print(f"Now sampling color: {COLOR_ORDER[self.color_index]}")
        else:
            self.color_index = len(COLOR_ORDER)
            print(f"{self.name}: done")

    def undo_last(self):
        current_color = self.get_current_color()

        if current_color is not None and self.samples[current_color]:
            removed = self.samples[current_color].pop()
            print(f"Undid sample from {current_color}: ({removed['x']}, {removed['y']})")
            return

        if self.color_index > 0:
            prev_color = COLOR_ORDER[self.color_index - 1]
            if self.samples[prev_color]:
                self.color_index -= 1
                removed = self.samples[prev_color].pop()
                print(f"Undid sample from {prev_color}: ({removed['x']}, {removed['y']})")
                return

        print("Nothing to undo.")

    def draw_sample(self, img, sample: Dict[str, object], idx: int, color: Tuple[int, int, int]):
        x = int(sample["x"])
        y = int(sample["y"])
        r = int(sample.get("sample_radius", self.radius))

        cv2.circle(img, (x, y), r, color, 1)
        cv2.circle(img, (x, y), 2, color, -1)

        draw_boxed_text(
            img,
            str(idx + 1),
            (x + 6, y - 6),
            text_color=color,
            font_scale=0.5,
            thickness=1,
            padding=3,
        )

    def draw(self):
        display = self.frame.copy()

        current_color = self.get_current_color()

        for color_name, sample_list in self.samples.items():
            draw_color = (0, 255, 0) if color_name == current_color else (180, 180, 180)
            for i, sample in enumerate(sample_list):
                self.draw_sample(display, sample, i, draw_color)

        y = 25
        lines = [
            f"{self.name} (camera {self.camera_index})",
            f"Radius: {self.radius}",
            f"Current color: {current_color}",
            f"Samples for current color: {0 if current_color is None else len(self.samples[current_color])}",
            "Click=save sample  N=next color",
            "U=undo  +/- radius  Q=quit",
            "Take several samples per color from both cameras",
        ]

        for line in lines:
            draw_boxed_text(display, line, (10, y), text_color=(255, 255, 255), font_scale=0.6)
            y += 26

        return display

    def summarize(self) -> Dict[str, object]:
        summary = {}

        for color_name, sample_list in self.samples.items():
            if len(sample_list) == 0:
                summary[color_name] = {
                    "samples": [],
                    "median_lab": None,
                    "median_bgr": None,
                }
                continue

            all_lab = np.array([sample["lab"] for sample in sample_list], dtype=np.float32)
            all_bgr = np.array([sample["median_bgr"] for sample in sample_list], dtype=np.float32)

            median_lab = np.median(all_lab, axis=0)
            median_bgr = np.median(all_bgr, axis=0)

            summary[color_name] = {
                "samples": sample_list,
                "median_lab": [float(v) for v in median_lab.tolist()],
                "median_bgr": [int(round(v)) for v in median_bgr.tolist()],
            }

        return summary

    def run(self) -> Dict[str, object]:
        print(f"\n=== {self.name} ===")
        print(f"Using frozen frame from camera {self.camera_index}")
        print("Controls:")
        print("  Left click = save sample")
        print("  N = next color")
        print("  U = undo last")
        print("  +/- = radius")
        print("  Q = quit")
        print(f"Start with color: {COLOR_ORDER[0]}")

        while True:
            display = self.draw()
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("q"), 27):
                raise KeyboardInterrupt("Calibration cancelled by user.")
            elif key in (ord("+"), ord("=")):
                self.radius += 1
            elif key == ord("-"):
                self.radius = max(1, self.radius - 1)
            elif key == ord("u"):
                self.undo_last()
            elif key == ord("n"):
                self.next_color()

            if self.color_index >= len(COLOR_ORDER):
                break

        cv2.destroyAllWindows()

        return {
            "camera_index": self.camera_index,
            "colors": self.summarize(),
        }


def merge_camera_colors(cam1: Dict[str, object], cam2: Dict[str, object]) -> Dict[str, object]:
    merged = {}

    colors1 = cam1["colors"]
    colors2 = cam2["colors"]

    for color_name in COLOR_ORDER:
        samples_1 = colors1[color_name]["samples"]
        samples_2 = colors2[color_name]["samples"]

        combined_samples = samples_1 + samples_2

        if len(combined_samples) == 0:
            merged[color_name] = {
                "samples": [],
                "median_lab": None,
                "median_bgr": None,
            }
            continue

        all_lab = np.array([sample["lab"] for sample in combined_samples], dtype=np.float32)
        all_bgr = np.array([sample["median_bgr"] for sample in combined_samples], dtype=np.float32)

        median_lab = np.median(all_lab, axis=0)
        median_bgr = np.median(all_bgr, axis=0)

        merged[color_name] = {
            "samples": combined_samples,
            "median_lab": [float(v) for v in median_lab.tolist()],
            "median_bgr": [int(round(v)) for v in median_bgr.tolist()],
        }

    return merged


def main():
    print("Capturing one frame from camera 1...")
    first_frame = capture_single_frame(1)

    first = ColorCalibration(
        first_frame,
        "Camera 1 Colors",
        1,
    ).run()

    print("Capturing one frame from camera 2...")
    second_frame = capture_single_frame(0)

    second = ColorCalibration(
        second_frame,
        "Camera 2 Colors",
        0   ,
    ).run()

    merged_colors = merge_camera_colors(first, second)

    data = {
        "camera_1": first,
        "camera_2": second,
        "merged": merged_colors,
        "color_order": COLOR_ORDER,
        "color_space": "LAB",
        "sample_method": "median patch",
    }

    with open("colors.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved colors.json")


if __name__ == "__main__":
    main()