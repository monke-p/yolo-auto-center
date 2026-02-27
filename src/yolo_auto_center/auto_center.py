import numpy as np
from collections import deque
from ultralytics import YOLO

class AutoCenter:
    def __init__(
        self,
        model_path: str,
        class_id: int = -1,
        conf: float = 0.5,
        iou: float = 0.5,
        imgsz: int = 640,
        device=None,
        deadband_x: float = 0.02,
        deadband_y: float = 0.02,
        smooth_frames: int = 0,
        # NEW:
        use_group: bool = True,          # True = center group centroid
        group_mode: str = "weighted",    # "weighted" | "mean" | "median"
        # (kept for backward-compat if use_group=False)
        select_mode: str = "largest",    # "largest" | "closest"
    ):
        self.model = YOLO(model_path)
        self.class_id = class_id
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device
        self.deadband_x = deadband_x
        self.deadband_y = deadband_y

        self.use_group = use_group
        self.group_mode = group_mode
        self.select_mode = select_mode

        self.use_smooth = smooth_frames > 0
        self.smooth_x = deque(maxlen=max(1, smooth_frames))
        self.smooth_y = deque(maxlen=max(1, smooth_frames))

    @classmethod
    def from_config(cls, cfg: dict):
        return cls(
            model_path=cfg["model_path"],
            class_id=cfg.get("class_id", -1),
            conf=cfg.get("conf", 0.5),
            iou=cfg.get("iou", 0.5),
            imgsz=cfg.get("imgsz", 640),
            device=cfg.get("device", None),
            deadband_x=cfg.get("deadband_x", 0.02),
            deadband_y=cfg.get("deadband_y", 0.02),
            smooth_frames=cfg.get("smooth_frames", 0),
            use_group=cfg.get("use_group", True),
            group_mode=cfg.get("group_mode", "weighted"),
            select_mode=cfg.get("select_mode", "largest"),
        )

    def process_frame(self, frame):
        H, W = frame.shape[:2]
        res = self.model.predict(
            frame, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False
        )[0]

        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return {"found": False, "dir_x": "NONE", "dir_y": "NONE", "err_x": 0.0, "err_y": 0.0}

        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)

        # Collect detections: area, cx, cy, bbox
        dets = []
        for i in range(len(xyxy)):
            if self.class_id != -1 and clss[i] != self.class_id:
                continue
            x1, y1, x2, y2 = map(float, xyxy[i])
            area = float((x2 - x1) * (y2 - y1))
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            dets.append((area, cx, cy, (x1, y1, x2, y2)))

        if not dets:
            return {"found": False, "dir_x": "NONE", "dir_y": "NONE", "err_x": 0.0, "err_y": 0.0}

        # --- Choose target center ---
        if self.use_group:
            # Group bbox (span of all)
            x1g = min(d[3][0] for d in dets)
            y1g = min(d[3][1] for d in dets)
            x2g = max(d[3][2] for d in dets)
            y2g = max(d[3][3] for d in dets)

            cxs = np.array([d[1] for d in dets], dtype=float)
            cys = np.array([d[2] for d in dets], dtype=float)

            if self.group_mode == "median":
                cx = float(np.median(cxs))
                cy = float(np.median(cys))
            elif self.group_mode == "mean":
                cx = float(np.mean(cxs))
                cy = float(np.mean(cys))
            else:
                # weighted by area (recommended)
                ws = np.array([max(d[0], 1.0) for d in dets], dtype=float)
                ws = ws / ws.sum()
                cx = float((cxs * ws).sum())
                cy = float((cys * ws).sum())

            bb = (x1g, y1g, x2g, y2g)

        else:
            # Single target (old behavior)
            if self.select_mode == "closest":
                _, cx, cy, bb = min(dets, key=lambda d: abs(d[1] - W/2) + abs(d[2] - H/2))
            else:
                _, cx, cy, bb = max(dets, key=lambda d: d[0])

        # --- Error (normalized) ---
        err_x = (cx - W/2) / W
        err_y = (cy - H/2) / H

        if self.use_smooth:
            self.smooth_x.append(err_x)
            self.smooth_y.append(err_y)
            err_x = float(np.mean(self.smooth_x))
            err_y = float(np.mean(self.smooth_y))

        dir_x = "OK" if abs(err_x) <= self.deadband_x else ("LEFT" if err_x < 0 else "RIGHT")
        dir_y = "OK" if abs(err_y) <= self.deadband_y else ("UP" if err_y < 0 else "DOWN")

        x1, y1, x2, y2 = bb
        return {
            "found": True,
            "dir_x": dir_x,
            "dir_y": dir_y,
            "err_x": float(err_x),
            "err_y": float(err_y),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "n": int(len(dets)),  # how many targets were used
        }