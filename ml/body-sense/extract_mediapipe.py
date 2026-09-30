"""Run the app's MediaPipe models (same .task files and thresholds as
src/tracking/inference.js) over video files and store per-frame landmarks.

  python extract_mediapipe.py OUT_DIR video1.mp4 video2.mp4 ...

Per video OUT_DIR/<name>.npz: t (s), pose (F,33,5: x,y,z,visibility,presence;
NaN when no person), face (F,4,4 facial transformation matrix, NaN if none),
face_box (F,4 normalized x0,y0,x1,y1), size (width,height), fps.
"""
import os, sys, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
MODELS = os.path.join(os.path.dirname(__file__), '..', '..', 'public', 'models')
MAX_FPS = 30.0
DELEGATE_NAME = os.environ.get('MP_DELEGATE', 'GPU')

def extract(args):
    out_dir, path = args
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    DELEGATE = getattr(BaseOptions.Delegate, DELEGATE_NAME)
    name = os.path.splitext(os.path.basename(path))[0]
    target = os.path.join(out_dir, name + '.npz')
    if os.path.exists(target): return name, 'cached'
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, 'pose_full.task'), delegate=DELEGATE),
        running_mode=vision.RunningMode.VIDEO, num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5, min_tracking_confidence=0.5))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=os.path.join(MODELS, 'face_landmarker.task'), delegate=DELEGATE),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5, min_tracking_confidence=0.5, output_facial_transformation_matrixes=True))
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / MAX_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, 1280.0 / max(w, h))  # the app captures at most 1280x720
    ts, poses, faces, boxes = [], [], [], []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if index % step: index += 1; continue
        if scale < 1: frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA))
        t_ms = int(round(index / fps * 1000))
        p = pose.detect_for_video(image, t_ms); f = face.detect_for_video(image, t_ms)
        if p.pose_landmarks:
            poses.append([[l.x, l.y, l.z, l.visibility, l.presence] for l in p.pose_landmarks[0]])
        else: poses.append(np.full((33, 5), np.nan))
        if f.face_landmarks and f.facial_transformation_matrixes:
            faces.append(np.array(f.facial_transformation_matrixes[0]))
            xy = np.array([[l.x, l.y] for l in f.face_landmarks[0]])
            boxes.append([*xy.min(0), *xy.max(0)])
        else: faces.append(np.full((4, 4), np.nan)); boxes.append([np.nan] * 4)
        ts.append(index / fps); index += 1
    cap.release(); pose.close(); face.close()
    np.savez_compressed(target, t=np.array(ts, np.float32), pose=np.array(poses, np.float32),
                        face=np.array(faces, np.float32), face_box=np.array(boxes, np.float32),
                        size=np.array([round(w * scale), round(h * scale)]), fps=fps / step)
    return name, len(ts)

if __name__ == '__main__':
    out_dir, paths = sys.argv[1], sys.argv[2:]
    os.makedirs(out_dir, exist_ok=True)
    with ProcessPoolExecutor(int(os.environ.get('WORKERS', 6))) as ex:
        for name, frames in ex.map(extract, [(out_dir, p) for p in paths]):
            print(name, frames, flush=True)
