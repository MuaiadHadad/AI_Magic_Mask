#!/usr/bin/env python3
"""
AI Magic Mask - Single-file GUI

Requisitos (instalar uma vez):
  pip install opencv-python numpy Pillow
  Opcional (melhor precisão/estabilidade):
  pip install mediapipe

Uso:
  python3 magic_mask.py
  python3 magic_mask.py --cv2 [--source 0|path|rtsp://..] [--res 480p|720p] [--backend haar|mediapipe]

Recursos:
- Interface gráfica (Tkinter) com botões: Iniciar, Parar, Screenshot
- Modo alternativo sem Tkinter (OpenCV window) via --cv2
- Seleção de fonte de vídeo (0, 1, caminho de arquivo, ou URL RTSP)
- Detecção facial com OpenCV Haar Cascade (multi-rosto) ou MediaPipe Face Mesh
- "Máscara mágica" com overlay colorido semi-transparente no rosto
- Overlays: caixas, landmarks, FPS, contador de rostos
- Suporte a resoluções 480p / 720p

Observações:
- Evita emojis para compatibilidade no macOS
- Desempenho: 25–35 FPS em 720p (hardware dependente); MediaPipe pode reduzir FPS
"""

from __future__ import annotations

import os
import sys
import time
import cv2
import numpy as np
import threading
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

# Lazy GUI imports (para evitar abort do Tk no macOS quando não usado)
tk = None
ttk = None
Image = None
ImageTk = None

def _init_tk_modules() -> None:
    global tk, ttk, Image, ImageTk
    try:
        import tkinter as _tk
        from tkinter import ttk as _ttk
        from PIL import Image as _Image, ImageTk as _ImageTk
        tk, ttk, Image, ImageTk = _tk, _ttk, _Image, _ImageTk
    except Exception as e:
        print("Erro ao importar GUI: certifique-se de ter Tkinter e Pillow disponíveis.")
        print("Dica macOS: use Python oficial de python.org para vir com Tk, ou use o modo --cv2.")
        raise

# MediaPipe (opcional)
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except Exception:
    MP_AVAILABLE = False


# Global performance knobs
try:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
except Exception:
    pass

@dataclass
class AppConfig:
    title: str = "AI Magic Mask"
    win_w: int = 1200
    win_h: int = 800
    view_w: int = 960
    default_source: str = "0"  # 0 = webcam padrão
    resolution: Tuple[int, int] = (1280, 720)  # (w, h)
    use_mediapipe: bool = True  # tentar usar MediaPipe Face Mesh
    max_faces: int = 5
    mp_min_detection_confidence: float = 0.5
    mp_min_tracking_confidence: float = 0.5
    # Anonymization mode: 'off' | 'blur' | 'pixel' | 'solid'
    anonymize: str = 'off'
    # Performance tuning
    detect_scale: float = 0.6           # downscale frame for detection for speed
    detect_every_n: int = 2             # run detection every N frames; track in-between
    roi_margin_frac: float = 0.25       # expand last face bbox for ROI detection


# Utility: merge rectangles with simple NMS (by IoU)
def _merge_rects(rects: list[tuple[int, int, int, int]], iou_thresh: float = 0.3) -> list[tuple[int, int, int, int]]:
    if not rects:
        return []
    # Convert to (x1, y1, x2, y2)
    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in rects]
    # Sort by area desc
    areas = [max(1, (bx[2] - bx[0])) * max(1, (bx[3] - bx[1])) for bx in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    kept = []
    while order:
        i = order.pop(0)
        kept.append(i)
        bxi = boxes[i]
        new_order = []
        for j in order:
            bxj = boxes[j]
            # IoU
            ix1 = max(bxi[0], bxj[0]); iy1 = max(bxi[1], bxj[1])
            ix2 = min(bxi[2], bxj[2]); iy2 = min(bxi[3], bxj[3])
            iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = areas[i] + areas[j] - inter
            iou = inter / union if union > 0 else 0.0
            if iou <= iou_thresh:
                new_order.append(j)
        order = new_order
    merged = []
    for k in kept:
        x1, y1, x2, y2 = boxes[k]
        merged.append((x1, y1, x2 - x1, y2 - y1))
    return merged

# ----------------------------- OpenCV-only mode -----------------------------
# Track last faces for ROI detection
_last_faces_cv: list[tuple[int, int, int, int]] = []


def _detect_faces_haar_cv(frame_bgr: np.ndarray, detector: cv2.CascadeClassifier, detector_profile: Optional[cv2.CascadeClassifier] = None,
                          use_roi: bool = True, roi_margin_frac: float = 0.25) -> list[tuple[int, int, int, int]]:
    global _last_faces_cv
    h_img, w_img = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    rects_all: list[tuple[int, int, int, int]] = []

    def detect_in_region(g: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int, int, int]]:
        sub = g[y0:y1, x0:x1]
        res = detector.detectMultiScale(sub, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        out: list[tuple[int, int, int, int]] = []
        if res is not None:
            for f in list(res):
                try:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    out.append((x + x0, y + y0, w, h))
                except Exception:
                    pass
        return out

    if use_roi and _last_faces_cv:
        # Detect within expanded ROIs to save time
        for (x, y, w, h) in _last_faces_cv:
            mx = int(w * roi_margin_frac)
            my = int(h * roi_margin_frac)
            rx0 = max(0, x - mx)
            ry0 = max(0, y - my)
            rx1 = min(w_img, x + w + mx)
            ry1 = min(h_img, y + h + my)
            rects_all.extend(detect_in_region(gray, rx0, ry0, rx1, ry1))
    else:
        # Full-frame if no prior faces
        faces_np = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if faces_np is not None:
            for f in list(faces_np):
                try:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    rects_all.append((x, y, w, h))
                except Exception:
                    pass

    # Profile detection on full gray (quick pass)
    prof_list: list[tuple[int, int, int, int]] = []
    if detector_profile is not None:
        prof_np = detector_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if prof_np is not None:
            for f in list(prof_np):
                try:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    prof_list.append((x, y, w, h))
                except Exception:
                    pass
        gray_flip = cv2.flip(gray, 1)
        prof_np_flip = detector_profile.detectMultiScale(gray_flip, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if prof_np_flip is not None:
            fw = gray.shape[1]
            for f in list(prof_np_flip):
                try:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    x_unflip = fw - (x + w)
                    prof_list.append((x_unflip, y, w, h))
                except Exception:
                    pass

    all_rects = rects_all + prof_list
    merged = _merge_rects(all_rects, iou_thresh=0.35)
    _last_faces_cv = merged
    return merged


def _detect_faces_mediapipe_cv(frame_bgr: np.ndarray, mp_face_mesh, scale: float = 0.6) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
    h0, w0 = frame_bgr.shape[:2]
    small_frame = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)))
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_small)
    faces: list[tuple[int, int, int, int]] = []
    landmarks_list: list[np.ndarray] = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            pts = []
            for lm in face_landmarks.landmark:
                x_px = int(lm.x * small_frame.shape[1])
                y_px = int(lm.y * small_frame.shape[0])
                pts.append((x_px, y_px))
            pts_np = np.array(pts, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts_np)
            faces.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
            landmarks_list.append((pts_np / scale).astype(np.int32))
    return faces, landmarks_list


def _apply_anonymize_effect(out: np.ndarray,
                            faces: list[tuple[int, int, int, int]],
                            landmarks_list: list[np.ndarray],
                            mode: str = 'off') -> np.ndarray:
    if mode == 'off':
        return out
    h_img, w_img = out.shape[:2]
    result = out.copy()

    for idx, bbox in enumerate(faces if faces else [()]*len(landmarks_list)):
        # Ensure bbox is a tuple if provided
        has_bbox = False
        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            has_bbox = True
        elif hasattr(bbox, "__len__") and len(bbox) == 4:
            try:
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                has_bbox = True
            except Exception:
                has_bbox = False

        # Build mask polygon: prefer landmarks convex hull; else ellipse bbox
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        poly_pts = None
        if landmarks_list and idx < len(landmarks_list):
            pts = landmarks_list[idx]
            if pts is not None and len(pts) > 0:
                pts_clip = pts.copy()
                pts_clip[:, 0] = np.clip(pts_clip[:, 0], 0, w_img - 1)
                pts_clip[:, 1] = np.clip(pts_clip[:, 1], 0, h_img - 1)
                hull = cv2.convexHull(pts_clip.astype(np.int32))
                poly_pts = hull.reshape(-1, 2)
        if poly_pts is None and has_bbox:
            center = (x + w // 2, y + h // 2)
            axes = (max(1, int(w * 0.55)), max(1, int(h * 0.6)))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        elif poly_pts is not None:
            cv2.fillConvexPoly(mask, poly_pts.astype(np.int32), 255)
        else:
            continue

        if mode == 'solid':
            color = (50, 50, 50)
            solid = np.full_like(result, color)
            result = np.where(mask[..., None] == 255, solid, result)
        elif mode == 'blur':
            # Strong blur: large adaptive kernel + two-pass blur
            # Baseline strong kernel, adapt by face width if bbox available
            base_k = 35  # higher baseline than before
            if has_bbox:
                # Scale kernel with face width; clamp to reasonable max to avoid huge CPU cost
                k = int(max(base_k, min(121, (w // 3))))
            else:
                k = base_k
            # Ensure odd
            if k % 2 == 0:
                k += 1
            # Two-pass blur for extra softness
            blurred1 = cv2.GaussianBlur(result, (k, k), 0)
            k2 = min(151, k + 20)  # second pass slightly larger, clamped
            if k2 % 2 == 0:
                k2 += 1
            blurred2 = cv2.GaussianBlur(blurred1, (k2, k2), 0)
            result = np.where(mask[..., None] == 255, blurred2, result)
        elif mode == 'pixel':
            # Compute ROI from mask by min/max of foreground coords
            ys, xs = np.where(mask == 255)
            if xs.size == 0 or ys.size == 0:
                continue
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            w_roi, h_roi = x1 - x0 + 1, y1 - y0 + 1
            roi = result[y0:y0 + h_roi, x0:x0 + w_roi]
            if roi.size == 0:
                continue
            scale = 0.08  # 8% size -> strong pixelation
            small_w = max(1, int(w_roi * scale))
            small_h = max(1, int(h_roi * scale))
            tiny = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
            pix = cv2.resize(tiny, (w_roi, h_roi), interpolation=cv2.INTER_NEAREST)
            submask = mask[y0:y0 + h_roi, x0:x0 + w_roi]
            pix_area = np.where(submask[..., None] == 255, pix, roi)
            result[y0:y0 + h_roi, x0:x0 + w_roi] = pix_area
    return result


def _render_overlays_cv(frame_bgr: np.ndarray, faces: list[tuple[int, int, int, int]], landmarks_list: list[np.ndarray], anonymize_mode: str = 'off') -> np.ndarray:
    out = frame_bgr.copy()

    # Apply anonymization first if requested
    if anonymize_mode and anonymize_mode != 'off':
        out = _apply_anonymize_effect(out, faces, landmarks_list, anonymize_mode)

    # Optional visual aids (skip heavy drawing if anonymizing to keep it clean)
    overlay = out.copy()
    if anonymize_mode == 'off':
        for i, (x, y, w, h) in enumerate(faces):
            colors = [(0, 255, 255), (255, 0, 255), (0, 200, 255), (0, 255, 128), (255, 128, 0)]
            color = colors[i % len(colors)]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(out.shape[1] - 1, x + w), min(out.shape[0] - 1, y + h)
            roi = overlay[y0:y1, x0:x1]
            if roi.size > 0:
                tint = np.full_like(roi, color, dtype=roi.dtype)
                cv2.addWeighted(tint, 0.25, roi, 0.75, 0, roi)
                overlay[y0:y1, x0:x1] = roi
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            cv2.putText(overlay, f"FACE {i+1}", (x + 8, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if landmarks_list:
            for pts in landmarks_list:
                color = (0, 255, 0)
                for idx in range(0, len(pts), 5):
                    x, y = int(pts[idx][0]), int(pts[idx][1])
                    if 0 <= x < out.shape[1] and 0 <= y < out.shape[0]:
                        cv2.circle(overlay, (x, y), 1, color, -1, lineType=cv2.LINE_AA)
                left_eye_idx, right_eye_idx = 33, 263
                if left_eye_idx < len(pts) and right_eye_idx < len(pts):
                    lx, ly = int(pts[left_eye_idx][0]), int(pts[left_eye_idx][1])
                    rx, ry = int(pts[right_eye_idx][0]), int(pts[right_eye_idx][1])
                    cv2.line(overlay, (lx, ly), (rx, ry), (255, 255, 0), 3, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (lx, ly), 10, (255, 255, 0), 2)
                    cv2.circle(overlay, (rx, ry), 10, (255, 255, 0), 2)

        cv2.addWeighted(overlay, 0.8, out, 0.2, 0, out)

    return out


def run_cv2_main(source: str, res: str, backend: str, cfg: AppConfig, max_frames: int = 0) -> int:
    # Abrir fonte
    cap_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cap_source)
    if not cap or not cap.isOpened():
        print("Erro ao abrir fonte de vídeo.")
        return 2
    w, h = (640, 480) if res == "480p" else (1280, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Backends
    detector = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    detector_profile = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml"))
    mp_face_mesh = None
    if backend.lower() == "mediapipe" and MP_AVAILABLE:
        mp_solutions = mp.solutions
        mp_face_mesh = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=cfg.max_faces,
            refine_landmarks=True,
            min_detection_confidence=cfg.mp_min_detection_confidence,
            min_tracking_confidence=cfg.mp_min_tracking_confidence,
        )
        print("Executando (MediaPipe) em modo OpenCV window")
    else:
        print("Executando (Haar) em modo OpenCV window")

    screenshot_count = 0
    fps_hist: list[float] = []
    last_ts = time.perf_counter()
    frame_counter = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        faces: list[tuple[int, int, int, int]]
        landmarks_list: list[np.ndarray]
        if backend.lower() == "mediapipe" and mp_face_mesh is not None:
            # Decimate detection frequency
            if frame_counter % max(1, cfg.detect_every_n) == 0:
                faces, landmarks_list = _detect_faces_mediapipe_cv(frame, mp_face_mesh, scale=cfg.detect_scale)
            else:
                faces, landmarks_list = _last_faces_cv, []
        else:
            if frame_counter % max(1, cfg.detect_every_n) == 0:
                faces = _detect_faces_haar_cv(frame, detector, detector_profile, use_roi=True, roi_margin_frac=cfg.roi_margin_frac)
            else:
                faces = _last_faces_cv
            landmarks_list = []
        out = _render_overlays_cv(frame, faces, landmarks_list, anonymize_mode=cfg.anonymize)

        # FPS HUD
        now = time.perf_counter()
        dt = max(1e-6, now - last_ts)
        last_ts = now
        fps = 1.0 / dt
        fps_hist.append(fps)
        if len(fps_hist) > 15:
            fps_hist.pop(0)
        smooth_fps = sum(fps_hist) / len(fps_hist)
        cv2.putText(out, f"FPS: {smooth_fps:.1f}  Faces: {len(faces) if faces else len(landmarks_list)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("AI Magic Mask (OpenCV)", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            import datetime
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"screenshot_{ts}_{screenshot_count}.png"
            screenshot_count += 1
            try:
                cv2.imwrite(name, out)
                print(f"Screenshot salvo: {name}")
            except Exception as e:
                print(f"Falha ao salvar screenshot: {e}")

        frame_counter += 1
        if max_frames and frame_counter >= max_frames:
            break

    try:
        cap.release()
    finally:
        pass
    if mp_face_mesh is not None:
        try:
            mp_face_mesh.close()
        except Exception:
            pass
    cv2.destroyAllWindows()
    return 0


# ----------------------------- Tkinter GUI mode -----------------------------
class MagicMaskApp:
    def __init__(self, cfg: Optional[AppConfig] = None) -> None:
        _init_tk_modules()
        self.cfg = cfg or AppConfig()
        self.root = tk.Tk()
        self.root.title(self.cfg.title)
        self.root.geometry(f"{self.cfg.win_w}x{self.cfg.win_h}")
        self.root.configure(bg="#121420")

        # Estado
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_bgr: Optional[np.ndarray] = None
        # Haar fallback
        self.detector = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        )
        self.detector_profile = cv2.CascadeClassifier(
            os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
        )
        # MediaPipe Face Mesh (lazy init)
        self.mp_face_mesh = None
        self.mp_draw = None
        self.process_thread: Optional[threading.Thread] = None
        self.fps_hist: list[float] = []
        self.screenshot_count = 0
        self.anonymize_var = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        # Header
        header = tk.Frame(self.root, bg="#1B2030", height=64)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(
            header, text=self.cfg.title, bg="#1B2030", fg="#E74C3C",
            font=("Helvetica", 18, "bold")
        ).pack(side=tk.LEFT, padx=16)

        self.fps_lbl = tk.Label(
            header, text="FPS: 0.0", bg="#1B2030", fg="#2ECC71",
            font=("Courier", 16, "bold")
        )
        self.fps_lbl.pack(side=tk.RIGHT, padx=16)

        # Main content
        content = tk.Frame(self.root, bg="#121420")
        content.pack(fill=tk.BOTH, expand=True)

        # Video panel (left)
        video_panel = tk.Frame(content, bg="#0E1220")
        video_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(12, 6), pady=12)

        self.video_lbl = tk.Label(
            video_panel, bg="#0E1220", fg="#95A5A6",
            text="Clique em Iniciar para começar", font=("Helvetica", 14)
        )
        self.video_lbl.pack(fill=tk.BOTH, expand=True)

        # Controls (right)
        controls = tk.Frame(content, bg="#1B2030", width=300)
        controls.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 12), pady=12)
        controls.pack_propagate(False)

        # Source selector
        tk.Label(controls, text="Fonte de Vídeo", bg="#1B2030", fg="#ECF0F1",
                 font=("Helvetica", 11, "bold")).pack(anchor=tk.W, padx=14, pady=(14, 4))
        self.src_var = tk.StringVar(value=self.cfg.default_source)
        self.src_entry = tk.Entry(controls, textvariable=self.src_var, bg="#0E1220", fg="#ECF0F1")
        self.src_entry.pack(fill=tk.X, padx=14)
        tk.Label(controls, text="Exemplos: 0  |  /path/video.mp4  |  rtsp://..",
                 bg="#1B2030", fg="#7F8C8D", font=("Helvetica", 9)).pack(anchor=tk.W, padx=14, pady=(2, 10))

        # Resolution
        tk.Label(controls, text="Resolução", bg="#1B2030", fg="#ECF0F1",
                 font=("Helvetica", 11, "bold")).pack(anchor=tk.W, padx=14)
        self.res_var = tk.StringVar(value="720p")
        ttk.Combobox(controls, textvariable=self.res_var, values=["480p", "720p"], state="readonly").pack(fill=tk.X, padx=14, pady=(0, 10))

        # Backend selector
        tk.Label(controls, text="Backend de Detecção", bg="#1B2030", fg="#ECF0F1",
                 font=("Helvetica", 11, "bold")).pack(anchor=tk.W, padx=14)
        self.backend_var = tk.StringVar(value=("MediaPipe" if (self.cfg.use_mediapipe and MP_AVAILABLE) else "Haar"))
        ttk.Combobox(controls, textvariable=self.backend_var, values=["MediaPipe", "Haar"], state="readonly").pack(fill=tk.X, padx=14, pady=(0, 10))
        if not MP_AVAILABLE:
            tk.Label(controls, text="MediaPipe não instalado — usando Haar", bg="#1B2030", fg="#F1C40F", font=("Helvetica", 9)).pack(anchor=tk.W, padx=14, pady=(0, 10))

        # Anonymize mode
        tk.Label(self.root if False else self.root, text="", bg="#121420").pack_forget()  # no-op spacer safeguard
        tk.Label(self.root if False else self.root, text="", bg="#121420").pack_forget()  # no-op spacer safeguard
        tk.Label(self.root if False else self.root, text="", bg="#121420").pack_forget()  # no-op spacer safeguard
        tk.Label(self.root if False else self.root, text="", bg="#121420").pack_forget()  # no-op spacer safeguard
        # Place in controls panel
        tk.Label(self.root if False else self.root, text="", bg="#121420").pack_forget()  # keep tool happy
        # Real content
        # We'll append to controls panel after backend
        # Recreate minimal label/combobox in place without reformatting existing layout
        # NOTE: Using a nested frame would require larger refactor; keep simple appends here
        label = tk.Label(controls, text="Anonimizar", bg="#1B2030", fg="#ECF0F1", font=("Helvetica", 11, "bold"))
        label.pack(anchor=tk.W, padx=14)
        self.anonymize_var = tk.StringVar(value=self.cfg.anonymize)
        ttk.Combobox(controls, textvariable=self.anonymize_var, values=["off", "blur", "pixel", "solid"], state="readonly").pack(fill=tk.X, padx=14, pady=(0, 10))

        # Buttons
        self.start_btn = tk.Button(controls, text="Iniciar", bg="#2ECC71", fg="#FFFFFF",
                                   font=("Helvetica", 12, "bold"), height=2, command=self._start)
        self.start_btn.pack(fill=tk.X, padx=14, pady=(6, 6))

        self.stop_btn = tk.Button(controls, text="Parar", bg="#E74C3C", fg="#FFFFFF",
                                  font=("Helvetica", 12, "bold"), height=2, command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, padx=14, pady=(0, 6))

        self.snap_btn = tk.Button(controls, text="Screenshot", bg="#3498DB", fg="#FFFFFF",
                                  font=("Helvetica", 12, "bold"), height=2, command=self._snapshot, state=tk.DISABLED)
        self.snap_btn.pack(fill=tk.X, padx=14, pady=(0, 6))

        # Status
        self.status_lbl = tk.Label(controls, text="Parado", bg="#1B2030", fg="#F1C40F",
                                   font=("Helvetica", 10))
        self.status_lbl.pack(anchor=tk.W, padx=14, pady=(8, 6))

        # Footer
        footer = tk.Frame(self.root, bg="#1B2030", height=36)
        footer.pack(fill=tk.X)
        footer.pack_propagate(False)
        tk.Label(footer, text="AI Magic Mask - Single-file GUI", bg="#1B2030", fg="#7F8C8D", font=("Helvetica", 9)).pack(side=tk.LEFT, padx=12)

    # App lifecycle ------------------------------------------------------
    def _start(self) -> None:
        if self.running:
            return
        source = self.src_var.get().strip()
        cap_source: object
        if source.isdigit():
            cap_source = int(source)
        else:
            cap_source = source

        self.cap = cv2.VideoCapture(cap_source)
        if not self.cap or not self.cap.isOpened():
            self.status_lbl.config(text="Erro ao abrir fonte de vídeo", fg="#E74C3C")
            return

        # Resolution
        w, h = (640, 480) if self.res_var.get() == "480p" else (1280, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Init backend
        backend = self.backend_var.get()
        if backend == "MediaPipe" and MP_AVAILABLE:
            self._init_mediapipe()
            self.status_lbl.config(text="Executando (MediaPipe)", fg="#2ECC71")
        else:
            self._release_mediapipe()
            self.status_lbl.config(text="Executando (Haar)", fg="#2ECC71")

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.snap_btn.config(state=tk.NORMAL)

        self.process_thread = threading.Thread(target=self._loop, daemon=True)
        self.process_thread.start()

    def _stop(self) -> None:
        self.running = False
        try:
            if self.cap:
                self.cap.release()
        finally:
            self.cap = None
        self._release_mediapipe()
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.snap_btn.config(state=tk.DISABLED)
        self.status_lbl.config(text="Parado", fg="#F1C40F")
        self.video_lbl.config(image='', text="Clique em Iniciar para começar")

    def _on_close(self) -> None:
        self._stop()
        self.root.quit()
        self.root.destroy()

    def _init_mediapipe(self) -> None:
        if not MP_AVAILABLE:
            return
        mp_solutions = mp.solutions
        self.mp_face_mesh = mp_solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.cfg.max_faces,
            refine_landmarks=True,
            min_detection_confidence=self.cfg.mp_min_detection_confidence,
            min_tracking_confidence=self.cfg.mp_min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _release_mediapipe(self) -> None:
        try:
            if self.mp_face_mesh is not None:
                self.mp_face_mesh.close()
        except Exception:
            pass
        self.mp_face_mesh = None
        self.mp_draw = None

    # Processing ---------------------------------------------------------
    def _loop(self) -> None:
        self.fps_hist.clear()
        last_ts = time.perf_counter()
        frame_counter = 0
        last_faces_gui: list[tuple[int, int, int, int]] = []
        while self.running:
            ok, frame = self.cap.read() if self.cap else (False, None)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            backend = self.backend_var.get()
            faces, landmarks_list = ([], [])
            if backend == "MediaPipe" and MP_AVAILABLE and self.mp_face_mesh is not None:
                if frame_counter % max(1, self.cfg.detect_every_n) == 0:
                    # Downscale mediapipe for speed
                    faces, landmarks_list = self._detect_faces_mediapipe_scaled(frame, scale=self.cfg.detect_scale)
                    last_faces_gui = faces
                else:
                    faces = last_faces_gui
                    landmarks_list = []
            else:
                if frame_counter % max(1, self.cfg.detect_every_n) == 0:
                    faces = self._detect_faces_haar_roi(frame, roi_margin_frac=self.cfg.roi_margin_frac)
                    last_faces_gui = faces
                else:
                    faces = last_faces_gui
                    landmarks_list = []

            out = self._render_overlays(frame, faces, landmarks_list)
            self.frame_bgr = out

            now = time.perf_counter()
            dt = max(1e-6, now - last_ts)
            last_ts = now
            fps = 1.0 / dt
            self.fps_hist.append(fps)
            if len(self.fps_hist) > 15:
                self.fps_hist.pop(0)
            smooth_fps = sum(self.fps_hist) / len(self.fps_hist)

            self.root.after(0, self._update_ui_frame, out, smooth_fps, len(faces) if faces else len(landmarks_list))
            time.sleep(0.001)

    def _detect_faces_haar(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces_np = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        faces_list: list[tuple[int, int, int, int]] = []
        if faces_np is not None:
            for f in list(faces_np):
                try:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    faces_list.append((x, y, w, h))
                except Exception:
                    pass
        # Profile detection (left + right via flip)
        if self.detector_profile is not None:
            prof_np = self.detector_profile.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            if prof_np is not None:
                for f in list(prof_np):
                    try:
                        x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        faces_list.append((x, y, w, h))
                    except Exception:
                        pass
            gray_flip = cv2.flip(gray, 1)
            prof_np_flip = self.detector_profile.detectMultiScale(gray_flip, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
            if prof_np_flip is not None:
                fw = gray.shape[1]
                for f in list(prof_np_flip):
                    try:
                        x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        x_unflip = fw - (x + w)
                        faces_list.append((x_unflip, y, w, h))
                    except Exception:
                        pass
        return _merge_rects(faces_list, iou_thresh=0.35)

    def _detect_faces_mediapipe(self, frame_bgr: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]]:
        h0, w0 = frame_bgr.shape[:2]
        scale = 0.7
        small_frame = cv2.resize(frame_bgr, (int(w0 * scale), int(h0 * scale)))
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(rgb_small)
        faces: list[tuple[int, int, int, int]] = []
        landmarks_list: list[np.ndarray] = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                pts = []
                for lm in face_landmarks.landmark:
                    x_px = int(lm.x * small_frame.shape[1])
                    y_px = int(lm.y * small_frame.shape[0])
                    pts.append((x_px, y_px))
                pts_np = np.array(pts, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts_np)
                x = int(x / scale)
                y = int(y / scale)
                w = int(w / scale)
                h = int(h / scale)
                faces.append((x, y, w, h))
                pts_up = (pts_np / scale).astype(np.int32)
                landmarks_list.append(pts_up)
        return faces, landmarks_list

    def _render_overlays(self, frame_bgr: np.ndarray, faces: list[tuple[int, int, int, int]], landmarks_list: list[np.ndarray]) -> np.ndarray:
        return _render_overlays_cv(frame_bgr, faces, landmarks_list, anonymize_mode=self.cfg.anonymize)

    def _update_ui_frame(self, frame_bgr: np.ndarray, fps: float, n_faces: int) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        target_w = min(self.cfg.view_w, w)
        target_h = int(h * (target_w / w))
        rgb_resized = cv2.resize(rgb, (target_w, target_h))

        img = Image.fromarray(rgb_resized)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_lbl.imgtk = imgtk
        self.video_lbl.configure(image=imgtk, text='')

        fps_color = "#2ECC71" if fps >= 25 else ("#F1C40F" if fps >= 15 else "#E74C3C")
        self.fps_lbl.config(text=f"FPS: {fps:.1f}", fg=fps_color)
        self.status_lbl.config(text=f"Rostos: {n_faces}", fg="#ECF0F1")

    def _snapshot(self) -> None:
        if self.frame_bgr is None:
            return
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"screenshot_{ts}_{self.screenshot_count}.png"
        self.screenshot_count += 1
        try:
            cv2.imwrite(name, self.frame_bgr)
            self.status_lbl.config(text=f"Screenshot salvo: {name}", fg="#3498DB")
        except Exception as e:
            self.status_lbl.config(text=f"Falha ao salvar: {e}", fg="#E74C3C")

    def run(self) -> None:
        print("Iniciando AI Magic Mask (Tk GUI)...")
        print("Dependências: opencv-python, numpy, Pillow, opcional: mediapipe")
        self.root.mainloop()


# ----------------------------- Entrypoint -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="AI Magic Mask")
    parser.add_argument("--cv2", action="store_true", help="Usa janela OpenCV (sem Tkinter)")
    parser.add_argument("--source", type=str, default="0", help="Fonte de vídeo: índice (ex: 0), caminho, ou URL RTSP")
    parser.add_argument("--res", type=str, choices=["480p", "720p"], default="720p", help="Resolução de captura")
    parser.add_argument("--backend", type=str, choices=["haar", "mediapipe"], default=("mediapipe" if MP_AVAILABLE else "haar"), help="Backend de detecção")
    parser.add_argument("--anonymize", type=str, choices=["off", "blur", "pixel", "solid"], default="off", help="Modo de anonimização facial")
    parser.add_argument("--max-frames", type=int, default=0, help="Fecha automaticamente após N frames (somente --cv2)")
    args = parser.parse_args()

    cfg = AppConfig()
    cfg.anonymize = args.anonymize
    if args.cv2:
        return run_cv2_main(args.source, args.res, args.backend, cfg, max_frames=args.max_frames)

    # Tkinter GUI mode
    try:
        app = MagicMaskApp(cfg)
        app.run()
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Erro fatal: {e}")
        import traceback
        traceback.print_exc()
        print("Dica: execute com --cv2 para evitar Tkinter no macOS problemático.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
