import argparse
from pathlib import Path

import cv2
import yaml

from yolo_auto_center import AutoCenter


HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to YAML config file")
    return ap.parse_args()


def open_source(src: str):
    if isinstance(src, str) and src.isdigit():
        return cv2.VideoCapture(int(src))
    return cv2.VideoCapture(src)


def draw_quadrants(img, deadband_x_px, deadband_y_px):
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2

    # Crosshair
    cv2.line(img, (cx, 0), (cx, H - 1), (255, 255, 255), 1)
    cv2.line(img, (0, cy), (W - 1, cy), (255, 255, 255), 1)

    # Deadband rectangle in the center
    x1 = int(cx - deadband_x_px)
    x2 = int(cx + deadband_x_px)
    y1 = int(cy - deadband_y_px)
    y2 = int(cy + deadband_y_px)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Quadrant labels (simple)
    cv2.putText(img, "UP-LEFT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(img, "UP-RIGHT", (W - 160, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(img, "DOWN-LEFT", (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(img, "DOWN-RIGHT", (W - 185, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)


def draw_arrows(img, dir_x, dir_y):
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2
    L = int(0.18 * min(W, H))

    # Horizontal arrow
    if dir_x in ("LEFT", "RIGHT"):
        end = (cx - L, cy) if dir_x == "LEFT" else (cx + L, cy)
        cv2.arrowedLine(img, (cx, cy), end, (0, 255, 255), 4, tipLength=0.25)

    # Vertical arrow
    if dir_y in ("UP", "DOWN"):
        end = (cx, cy - L) if dir_y == "UP" else (cx, cy + L)
        cv2.arrowedLine(img, (cx, cy), end, (0, 255, 255), 4, tipLength=0.25)


def draw_status_box(img, text):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    pad = 10
    cv2.rectangle(img, (10, 10), (10 + tw + 2 * pad, 10 + th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(img, text, (10 + pad, 10 + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ac = AutoCenter.from_config(cfg)

    cap = open_source(str(cfg.get("source", "0")))
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir source: {cfg.get('source')}")

    show = bool(cfg.get("show_window", True))
    if not show:
        raise RuntimeError("show_window=false no tiene sentido para este example visual")

    cv2.namedWindow("yolo-auto-center (visual)", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        out = ac.process_frame(frame)

        # Draw quadrants + deadband
        deadband_x_px = cfg.get("deadband_x", 0.02) * W
        deadband_y_px = cfg.get("deadband_y", 0.02) * H
        draw_quadrants(frame, deadband_x_px, deadband_y_px)

        if out["found"]:
            # Draw selected bbox if present
            if all(k in out for k in ("x1", "y1", "x2", "y2")):
                x1, y1, x2, y2 = int(out["x1"]), int(out["y1"]), int(out["x2"]), int(out["y2"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw bbox center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

            # Arrows according to needed movement
            draw_arrows(frame, out["dir_x"], out["dir_y"])

            # Status text with error
            ex = out["err_x"] * 100.0
            ey = out["err_y"] * 100.0
            n = out.get("n", 1)  # default 1 for single target mode
            status = f"N={n}  {out['dir_x']}|{out['dir_y']}  err=({ex:+.1f}%, {ey:+.1f}%)"
        else:
            status = "NONE (no detection)"

        draw_status_box(frame, status)

        cv2.imshow("yolo-auto-center (visual)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()