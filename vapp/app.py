# app.py
import sys, os, time
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from logic import TrafficController, Timing
from detector import VehicleDetector
from utils import ROIStore

# ---------- Worker thread for video processing ----------
class Processor(QtCore.QThread):
    update = QtCore.Signal(dict)  # emits {'frame': np.ndarray, 'counts': dict, 'state': export}

    def __init__(self, video_path: str, rois: Dict[str, List[Tuple[int,int]]], model_path: str, device: str = None, conf: float = 0.35):
        super().__init__()
        self.video_path = video_path
        self.rois = rois
        self.detector = VehicleDetector(model_path=model_path, conf=conf, device=device)
        self.ctrl = TrafficController(Timing())
        self._stop = False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
        frame_interval = max(1, int(fps / 10))  # process ~10 Hz
        idx = 0
        # smoothing counts
        hist = {k: [] for k in self.rois.keys()}
        while not self._stop:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if idx % frame_interval == 0:
                counts = self.detector.count_in_rois(frame, self.rois)
                for k, v in counts.items():
                    arr = hist[k]
                    arr.append(v)
                    if len(arr) > 10:
                        arr.pop(0)
                    counts[k] = int(round(sum(arr) / len(arr)))
                self.ctrl.tick(counts)
                state = self.ctrl.export()
                # draw overlays
                vis = self._draw_overlay(frame.copy(), counts, state)
                self.update.emit({'frame': vis, 'counts': counts, 'state': state})
            idx += 1
        cap.release()

    def stop(self):
        self._stop = True

    def _draw_overlay(self, frame, counts, state):
        # draw polygons
        for lbl, poly in self.rois.items():
            pts = np.array(poly, dtype=np.int32)
            color = (0, 0, 255)
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
            x, y = pts[0]
            cv2.putText(frame, f"{lbl}:{counts.get(lbl,0)}", (x, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        # draw current/phase timer
        rem = max(0, int(round(state['ends_at'] - state['now'])))
        text = f"{state['current']} {state['phase']} {rem}s"
        cv2.rectangle(frame, (10, 10), (310, 60), (0,0,0), -1)
        cv2.putText(frame, text, (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        return frame

# ---------- Polygon drawing widget ----------
class VideoCanvas(QtWidgets.QLabel):
    pointAdded = QtCore.Signal(tuple)  # (x,y)

    def __init__(self):
        super().__init__()
        self.setScaledContents(True)
        self.image = None
        self.temp_pts: List[Tuple[int,int]] = []
        self.polys: Dict[str, List[Tuple[int,int]]] = {}
        self.current_label = 'N'

    def set_frame(self, frame: np.ndarray):
        self.image = frame
        self._update_pix()

    def _update_pix(self):
        if self.image is None:
            return
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self.image is None:
            return
        pos = ev.position() if hasattr(ev, 'position') else ev.posF()
        x = int(pos.x() * (self.image.shape[1] / self.width()))
        y = int(pos.y() * (self.image.shape[0] / self.height()))
        self.temp_pts.append((x, y))
        self.pointAdded.emit((x, y))
        self.repaint()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self.image is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHints(QtGui.QPainter.Antialiasing)
        scale_x = self.width() / self.image.shape[1]
        scale_y = self.height() / self.image.shape[0]
        pen = QtGui.QPen(QtGui.QColor('lime'))
        pen.setWidth(2)
        painter.setPen(pen)
        # draw temp poly
        for i in range(1, len(self.temp_pts)):
            x1, y1 = self.temp_pts[i-1]
            x2, y2 = self.temp_pts[i]
            painter.drawLine(int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y))
        # draw committed polys
        pen2 = QtGui.QPen(QtGui.QColor('red'))
        pen2.setWidth(2)
        painter.setPen(pen2)
        font = QtGui.QFont()
        font.setPointSize(10)
        painter.setFont(font)
        for lbl, pts in self.polys.items():
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i+1) % len(pts)]
                painter.drawLine(int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y))
            if pts:
                x, y = pts[0]
                painter.fillRect(int(x*scale_x), int((y-18)*scale_y), 28, int(16*scale_y), QtGui.QColor(255,255,255,200))
                painter.setPen(QtGui.QPen(QtGui.QColor('black')))
                painter.drawText(int((x+4)*scale_x), int((y-4)*scale_y), lbl)
                painter.setPen(pen2)

    def commit_polygon(self, label: str):
        if len(self.temp_pts) < 3:
            return False
        self.polys[label] = self.temp_pts[:]
        self.temp_pts.clear()
        self.repaint()
        return True

    def clear_all(self):
        self.temp_pts.clear()
        self.polys.clear()
        self.repaint()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SmartTraffic Desktop')
        self.resize(1200, 800)

        self.video_path = None
        self.model_path = None
        self.device = None  # e.g., '0' for first GPU, or None for auto

        # central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        # left: video/roi
        left = QtWidgets.QVBoxLayout()
        self.canvas = VideoCanvas()
        self.canvas.setMinimumSize(960, 540)
        left.addWidget(self.canvas, 1)

        controls = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton('Open Video')
        self.btn_model = QtWidgets.QPushButton('Load Model (.pt)')
        self.combo_label = QtWidgets.QComboBox(); self.combo_label.addItems(['N','E','S','W'])
        self.btn_commit = QtWidgets.QPushButton('Commit Polygon')
        self.btn_clear = QtWidgets.QPushButton('Clear All')
        self.btn_save = QtWidgets.QPushButton('Save ROI')
        self.btn_load = QtWidgets.QPushButton('Load ROI')
        controls.addWidget(self.btn_open)
        controls.addWidget(self.btn_model)
        controls.addWidget(QtWidgets.QLabel('Label:'))
        controls.addWidget(self.combo_label)
        controls.addWidget(self.btn_commit)
        controls.addWidget(self.btn_clear)
        controls.addWidget(self.btn_save)
        controls.addWidget(self.btn_load)
        left.addLayout(controls)

        run_controls = QtWidgets.QHBoxLayout()
        self.btn_start = QtWidgets.QPushButton('Start')
        self.btn_stop = QtWidgets.QPushButton('Stop')
        self.spin_conf = QtWidgets.QDoubleSpinBox(); self.spin_conf.setRange(0.05, 0.95); self.spin_conf.setSingleStep(0.05); self.spin_conf.setValue(0.35)
        self.edit_device = QtWidgets.QLineEdit(); self.edit_device.setPlaceholderText("GPU device (e.g., 0) or leave blank")
        run_controls.addWidget(QtWidgets.QLabel('Conf:'))
        run_controls.addWidget(self.spin_conf)
        run_controls.addWidget(QtWidgets.QLabel('Device:'))
        run_controls.addWidget(self.edit_device)
        run_controls.addWidget(self.btn_start)
        run_controls.addWidget(self.btn_stop)
        left.addLayout(run_controls)

        h.addLayout(left, 3)

        # right: status panel
        right = QtWidgets.QVBoxLayout()
        self.lab_state = QtWidgets.QLabel('State: -')
        font = self.lab_state.font(); font.setPointSize(14); self.lab_state.setFont(font)
        right.addWidget(self.lab_state)

        self.grid = QtWidgets.QGridLayout()
        self.widgets = {}
        for i, lbl in enumerate(['N','E','S','W']):
            g = QtWidgets.QGroupBox(lbl)
            v = QtWidgets.QVBoxLayout(g)
            lamp = QtWidgets.QLabel(); lamp.setFixedSize(60, 60); lamp.setStyleSheet('background:#400; border-radius:30px;')
            count = QtWidgets.QLabel('Count: 0')
            timer = QtWidgets.QLabel('Timer: 0s')
            for w in (lamp, count, timer):
                f = w.font(); f.setPointSize(12); w.setFont(f)
            v.addWidget(lamp, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            v.addWidget(count, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            v.addWidget(timer, alignment=QtCore.Qt.AlignmentFlag.AlignHCenter)
            self.grid.addWidget(g, i, 0)
            self.widgets[lbl] = {'lamp': lamp, 'count': count, 'timer': timer}
        right.addLayout(self.grid)
        right.addStretch(1)

        h.addLayout(right, 1)

        # connections
        self.btn_open.clicked.connect(self.open_video)
        self.btn_model.clicked.connect(self.open_model)
        self.btn_commit.clicked.connect(self.commit_poly)
        self.btn_clear.clicked.connect(self.canvas.clear_all)
        self.btn_save.clicked.connect(self.save_roi)
        self.btn_load.clicked.connect(self.load_roi)
        self.btn_start.clicked.connect(self.start_proc)
        self.btn_stop.clicked.connect(self.stop_proc)

        self.proc: Processor | None = None

    # ---------- UI handlers ----------
    def open_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Video', '', 'Video Files (*.mp4 *.avi *.mov *.mkv)')
        if not path:
            return
        self.video_path = path
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if ok:
            self.canvas.set_frame(frame)
        else:
            QtWidgets.QMessageBox.critical(self, 'Error', 'Could not read first frame')

    def open_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load YOLO Model', '', 'YOLO Weights (*.pt)')
        if not path:
            return
        self.model_path = path

    def commit_poly(self):
        label = self.combo_label.currentText()
        if not self.canvas.commit_polygon(label):
            QtWidgets.QMessageBox.warning(self, 'ROI', 'Polygon needs at least 3 points.')

    def save_roi(self):
        if not self.canvas.polys:
            QtWidgets.QMessageBox.information(self, 'ROI', 'No polygons to save')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Save ROI JSON', '', 'JSON (*.json)')
        if not path:
            return
        from utils import ROIStore
        store = ROIStore()
        for k, v in self.canvas.polys.items():
            store.set_poly(k, v)
        store.save(path)

    def load_roi(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load ROI JSON', '', 'JSON (*.json)')
        if not path:
            return
        from utils import ROIStore
        store = ROIStore(); store.load(path)
        self.canvas.polys = store.polys
        self.canvas.repaint()

    def start_proc(self):
        if not self.video_path:
            QtWidgets.QMessageBox.warning(self, 'Start', 'Open a video first')
            return
        required = {'N','E','S','W'}
        if set(self.canvas.polys.keys()) != required:
            QtWidgets.QMessageBox.warning(self, 'Start', 'Please draw and commit polygons for N, E, S, W')
            return
        device = self.edit_device.text().strip() or None
        self.proc = Processor(self.video_path, self.canvas.polys, self.model_path, device=device, conf=self.spin_conf.value())
        self.proc.update.connect(self.on_update)
        self.proc.start()

    def stop_proc(self):
        if self.proc:
            self.proc.stop(); self.proc.wait(1000); self.proc = None

    def closeEvent(self, ev: QtGui.QCloseEvent):
        self.stop_proc()
        super().closeEvent(ev)

    @QtCore.Slot(dict)
    def on_update(self, payload: dict):
        frame = payload['frame']
        counts = payload['counts']
        state = payload['state']
        # update canvas
        self.canvas.set_frame(frame)
        # update lamps and timers
        current = state['current']; phase = state['phase']; rem = max(0, int(round(state['ends_at'] - state['now'])))
        self.lab_state.setText(f"Current: {current}  Phase: {phase}  Remaining: {rem}s")
        for k, w in self.widgets.items():
            w['count'].setText(f"Count: {counts.get(k,0)}")
            if k == current:
                if phase == 'GREEN': color = '#0a0'
                elif phase == 'YELLOW': color = '#aa0'
                else: color = '#a00'
                w['timer'].setText(f"Timer: {rem}s")
            else:
                color = '#400'
                w['timer'].setText('Timer: 0s')
            w['lamp'].setStyleSheet(f'background:{color}; border-radius:30px;')


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()