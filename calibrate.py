import cv2
import json
import time
from typing import Dict, List, Optional, Tuple

WINDOW_NAME = "Position Calibration"
DEFAULT_RADIUS = 6

FIRST_LABELS = [
    "F top-left", "F top-middle", "F top-right",
    "F middle-left", "F center", "F middle-right",
    "F bottom-left", "F bottom-middle", "F bottom-right",

    "R top-left", "R top-middle", "R top-right",
    "R middle-left", "R center", "R middle-right",
    "R bottom-left", "R bottom-middle", "R bottom-right",

    "U top-left", "U top-middle", "U top-right",
    "U middle-left", "U center", "U middle-right",
    "U bottom-left", "U bottom-middle", "U bottom-right",
]

SECOND_LABELS = [
    "B top-left", "B top-middle", "B top-right",
    "B middle-left", "B center", "B middle-right",
    "B bottom-left", "B bottom-middle", "B bottom-right",

    "D top-left", "D top-middle", "D top-right",
    "D middle-left", "D center", "D middle-right",
    "D bottom-left", "D bottom-middle", "D bottom-right",

    "L top-left", "L top-middle", "L top-right",
    "L middle-left", "L center", "L middle-right",
    "L bottom-left", "L bottom-middle", "L bottom-right",
]


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


def rotate_point_180(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    return width - 1 - x, height - 1 - y


class FrameCalibration:
    def __init__(self, frame, labels: List[str], name: str, camera_index: int, invert: bool = False):
        self.frame = frame
        self.labels = labels
        self.name = name
        self.camera_index = camera_index
        self.invert = invert

        self.mode = "stickers"
        self.index = 0
        self.radius = DEFAULT_RADIUS

        self.stickers: Dict[str, Dict[str, int]] = {}

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.click)

    def get_current_label(self) -> Optional[str]:
        if self.index < len(self.labels):
            return self.labels[self.index]
        return None

    def display_to_storage_coords(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert clicked display coordinates back to original frame coordinates
        before saving.
        """
        if not self.invert:
            return x, y

        h, w = self.frame.shape[:2]
        return rotate_point_180(x, y, w, h)

    def storage_to_display_coords(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert stored original frame coordinates into display coordinates.
        """
        if not self.invert:
            return x, y

        h, w = self.frame.shape[:2]
        return rotate_point_180(x, y, w, h)

    def click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "done":
            return

        label = self.get_current_label()
        if label is None:
            return

        save_x, save_y = self.display_to_storage_coords(x, y)

        data = {
            "x": int(save_x),
            "y": int(save_y),
            "sample_radius": int(self.radius),
        }

        self.stickers[label] = data
        print(
            f"{self.name} -> {label}: "
            f"display=({x}, {y}) saved=({save_x}, {save_y}) radius={self.radius}"
        )

        self.index += 1
        if self.index >= len(self.labels):
            self.advance()

    def advance(self):
        self.mode = "done"
        print(f"{self.name}: done")

    def undo_last(self):
        if self.index > 0:
            self.index -= 1
            label = self.labels[self.index]
            self.stickers.pop(label, None)
            self.mode = "stickers"
            print(f"Undid sticker {label}")
        else:
            print("Nothing to undo.")

    def toggle_invert(self):
        self.invert = not self.invert
        print(f"{self.name}: invert = {self.invert}")

    def draw_point(self, img, point: Dict[str, int], index: int, color: Tuple[int, int, int]):
        x = int(point["x"])
        y = int(point["y"])
        r = int(point.get("sample_radius", self.radius))

        draw_x, draw_y = self.storage_to_display_coords(x, y)

        cv2.circle(img, (draw_x, draw_y), r, color, 1)
        cv2.circle(img, (draw_x, draw_y), 2, color, -1)

        draw_boxed_text(
            img,
            str(index + 1),
            (draw_x + 6, draw_y - 6),
            text_color=color,
            font_scale=0.5,
            thickness=1,
            padding=3,
        )

    def draw(self):
        display = self.frame.copy()

        if self.invert:
            display = cv2.rotate(display, cv2.ROTATE_180)

        for i, label in enumerate(self.labels):
            if label in self.stickers:
                self.draw_point(display, self.stickers[label], i, (0, 255, 0))

        y = 25
        lines = [
            f"{self.name} (camera {self.camera_index})",
            f"Mode: {self.mode}",
            f"Radius: {self.radius}",
            f"Invert: {self.invert}",
            f"Next: {self.get_current_label()}",
            "Click=save  U=undo  I=invert  +/- radius  Q=quit",
        ]

        for line in lines:
            draw_boxed_text(display, line, (10, y), text_color=(255, 255, 255), font_scale=0.6)
            y += 26

        return display

    def run(self) -> Dict[str, object]:
        print(f"\n=== {self.name} ===")
        print(f"Using frozen frame from camera {self.camera_index}")
        print("Controls: left click=save, U=undo, I=invert, +/- radius, Q=quit")

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
            elif key == ord("i"):
                self.toggle_invert()

            if self.mode == "done":
                break

        cv2.destroyAllWindows()

        return {
            "camera_index": self.camera_index,
            "invert": self.invert,
            "stickers": self.stickers,
        }


def main():
    print("Capturing one frame from camera 0...")
    first_frame = capture_single_frame(0)

    first = FrameCalibration(
        first_frame,
        FIRST_LABELS,
        "Camera 0: F/R/U",
        0,
        invert=False,
    ).run()

    print("Capturing one frame from camera 1...")
    second_frame = capture_single_frame(1)

    second = FrameCalibration(
        second_frame,
        SECOND_LABELS,
        "Camera 1: B/D/L",
        1,
        invert=False,
    ).run()

    data = {
        "camera_1": first,
        "camera_2": second,
    }

    with open("position.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved position.json")


if __name__ == "__main__":
    main()
    