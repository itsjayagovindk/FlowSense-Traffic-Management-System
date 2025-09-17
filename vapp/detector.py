# detector.py
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

try:
    from ultralytics import YOLO
    _HAVE_YOLO = True
except Exception:
    YOLO = None
    _HAVE_YOLO = False

# COCO-style vehicle ids by default (adjust for your trained model)
DEFAULT_CLASS_NAMES = {
    0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck'
}

VEHICLE_SET = set(DEFAULT_CLASS_NAMES.keys())

class VehicleDetector:
    def __init__(self, model_path: Optional[str] = None, class_names: Dict[int, str] = None, conf: float = 0.35, device: Optional[str] = None):
        self.conf = conf
        self.model = None
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.device = device
        if _HAVE_YOLO and model_path:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print('[detector] Failed to load YOLO:', e)
        # simple BGSub fallback if model is None
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=36, detectShadows=True)

    def _centers_from_fg(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        fg = self.bg.apply(frame)
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for c in contours:
            if cv2.contourArea(c) < 900:
                continue
            x, y, w, h = cv2.boundingRect(c)
            centers.append((x + w/2, y + h/2))
        return centers

    def detect_centers(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        if self.model is None:
            return self._centers_from_fg(frame)
        centers: List[Tuple[float, float]] = []
        results = self.model(frame, conf=self.conf, device=self.device, verbose=False)
        for r in results:
            if getattr(r, 'boxes', None) is None:
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                if cls_id not in self.class_names:
                    # If your trained model has 4 classes indexed 0..3, this filters others
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        return centers

    @staticmethod
    def point_in_poly(pt: Tuple[float, float], poly: List[Tuple[int, int]]) -> bool:
        arr = np.array(poly, dtype=np.int32)
        return cv2.pointPolygonTest(arr, (float(pt[0]), float(pt[1])), False) > 0

    def count_in_rois(self, frame: np.ndarray, rois: Dict[str, List[Tuple[int, int]]]) -> Dict[str, int]:
        centers = self.detect_centers(frame)
        counts = {k: 0 for k in rois.keys()}
        for c in centers:
            for name, poly in rois.items():
                if self.point_in_poly(c, poly):
                    counts[name] += 1
                    break
        return counts