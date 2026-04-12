import cv2
import json
import time
from typing import Dict, List, Optional, Tuple

WINDOW_NAME = "Calibration"
DEFAULT_RADIUS = 6

FIRST_LABELS = [
    "F top-left", "F top-middle", "F top-right",
    "F middle-left", "F center", "F middle-right",
    "F bottom-left", "F bottom-middle", "F bottom-right",

    "R top-left", "R top-middle", "R top-right",
    "R middle-left", "R center", "R middle-right",
    "R bottom-left", "R bottom-middle", "R bottom-right",

    "D top-left", "D top-middle", "D top-right",
    "D middle-left", "D center", "D middle-right",
    "D bottom-left", "D bottom-middle", "D bottom-right",
]

SECOND_LABELS = [
    "L top-left", "L top-middle", "L top-right",
    "L middle-left", "L center", "L middle-right",
    "L bottom-left", "L bottom-middle", "L bottom-right",

    "B top-left", "B top-middle", "B top-right",
    "B middle-left", "B center", "B middle-right",
    "B bottom-left", "B bottom-middle", "B bottom-right",

    "U top-left", "U top-middle", "U top-right",
    "U middle-left", "U center", "U middle-right",
    "U bottom-left", "U bottom-middle", "U bottom-right",
]

FIRST_COLORS = ["green", "red", "yellow"]
SECOND_COLORS = ["blue", "orange", "white"]


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


class FrameCalibration:
    def __init__(self, frame, labels: List[str], colors: List[str], name: str, camera_index: int):
        self.frame = frame
        self.labels = labels
        self.colors = colors
        self.name = name
        self.camera_index = camera_index

        self.mode = "stickers"
        self.index = 0
        self.radius = DEFAULT_RADIUS

        self.stickers: Dict[str, Dict[str, object]] = {}
        self.color_refs: Dict[str, Dict[str, object]] = {}

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.click)

    def get_labels(self) -> List[str]:
        return self.labels if self.mode == "stickers" else self.colors

    def get_current_label(self) -> Optional[str]:
        labels = self.get_labels()
        if self.index < len(labels):
            return labels[self.index]
        return None

    def sample_patch_average(self, x: int, y: int, radius: int) -> List[int]:
        h, w = self.frame.shape[:2]
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)

        patch = self.frame[y1:y2, x1:x2]
        if patch.size == 0:
            return [0, 0, 0]

        mean = patch.mean(axis=(0, 1))
        return [int(round(v)) for v in mean.tolist()]

    def click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "done":
            return

        label = self.get_current_label()
        if label is None:
            return

        avg_bgr = self.sample_patch_average(x, y, self.radius)

        data = {
            "x": int(x),
            "y": int(y),
            "sample_radius": int(self.radius),
            "avg_bgr": avg_bgr,
        }

        if self.mode == "stickers":
            self.stickers[label] = data
        else:
            self.color_refs[label] = data

        print(f"{self.name} -> {label}: ({x}, {y}) BGR={avg_bgr}")

        self.index += 1
        if self.index >= len(self.get_labels()):
            self.advance()

    def advance(self):
        if self.mode == "stickers":
            self.mode = "colors"
            self.index = 0
            print(f"{self.name}: sticker positions done. Now click colors: {', '.join(self.colors)}")
        elif self.mode == "colors":
            self.mode = "done"
            print(f"{self.name}: done")

    def undo_last(self):
        if self.mode == "done":
            if self.color_refs:
                self.mode = "colors"
                self.index = len(self.color_refs)
            else:
                self.mode = "stickers"
                self.index = len(self.stickers)

        if self.mode == "colors":
            if self.index > 0:
                self.index -= 1
                label = self.colors[self.index]
                self.color_refs.pop(label, None)
                print(f"Undid color {label}")
                return
            if self.stickers:
                self.mode = "stickers"
                self.index = len(self.stickers)

        if self.mode == "stickers":
            if self.index > 0:
                self.index -= 1
                label = self.labels[self.index]
                self.stickers.pop(label, None)
                print(f"Undid sticker {label}")
            else:
                print("Nothing to undo.")

    def draw_point(self, img, point: Dict[str, object], index: int, color: Tuple[int, int, int]):
        x = int(point["x"])
        y = int(point["y"])
        r = int(point.get("sample_radius", self.radius))

        cv2.circle(img, (x, y), r, color, 1)
        cv2.circle(img, (x, y), 2, color, -1)

        draw_boxed_text(
            img,
            str(index + 1),
            (x + 6, y - 6),
            text_color=color,
            font_scale=0.5,
            thickness=1,
            padding=3,
        )

    def draw(self):
        display = self.frame.copy()

        for i, label in enumerate(self.labels):
            if label in self.stickers:
                self.draw_point(display, self.stickers[label], i, (0, 255, 0))

        for i, label in enumerate(self.colors):
            if label in self.color_refs:
                self.draw_point(display, self.color_refs[label], i, (255, 200, 0))

        y = 25
        lines = [
            f"{self.name} (camera {self.camera_index})",
            f"Mode: {self.mode}",
            f"Radius: {self.radius}",
            f"Next: {self.get_current_label()}",
            "Click=save  U=undo  +/- radius  Q=quit",
        ]

        for line in lines:
            draw_boxed_text(display, line, (10, y), text_color=(255, 255, 255), font_scale=0.6)
            y += 26

        return display

    def run(self) -> Dict[str, object]:
        print(f"\n=== {self.name} ===")
        print(f"Using frozen frame from camera {self.camera_index}")
        print("Controls: left click=save, U=undo, +/- radius, Q=quit")

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

            if self.mode == "done":
                break

        cv2.destroyAllWindows()

        return {
            "camera_index": self.camera_index,
            "stickers": self.stickers,
            "colors": self.color_refs,
        }


def main():
    print("Capturing one frame from camera 2...")
    first_frame = capture_single_frame(1)

    first = FrameCalibration(
        first_frame,
        FIRST_LABELS,
        FIRST_COLORS,
        "Camera 2: F/R/D",
        1,
    ).run()

    print("Capturing one frame from camera 3...")
    second_frame = capture_single_frame(0)

    second = FrameCalibration(
        second_frame,
        SECOND_LABELS,
        SECOND_COLORS,
        "Camera 3: L/B/U",
        0,
    ).run()

    data = {
        "first_image": first,
        "second_image": second,
    }

    with open("cube.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved cube.json")


if __name__ == "__main__":
    main()