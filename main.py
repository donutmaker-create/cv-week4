import argparse
from pathlib import Path
import time
import urllib.request
import pygame

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

ROOT = Path(__file__).resolve().parent
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')
SWARAS = {
    1: ('Sa', '1_Sa.mp3'), 2: ('Re', '2_Re.mp3'),
    3: ('Ga', '3_Ga.mp3'), 4: ('Ma', '4_Ma.mp3'),
    5: ('Pa', '5_Pa.mp3'), 6: ('Dha', '6_Dha.mp3'),
    7: ('Ni', '7_Ni.mp3'), 8: ('Sa (upper)', '8_Sa_upper.mp3'),
}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def ensure_model(path):
    """Download once. A partial download never replaces the model file."""
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.download')
    print('Downloading hand model (first run only)...')
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            temporary.write_bytes(response.read())
        temporary.replace(path)
    except Exception as error:
        raise RuntimeError(
            f'Could not download model. Download {MODEL_URL} manually '
            f'and save it as {path}.') from error
    finally:
        temporary.unlink(missing_ok=True)


def count_fingers(landmarks, label):
    """Simple landmark comparisons from the guide; assumes upright palms."""
    count = sum(landmarks[tip].y < landmarks[tip - 2].y
                for tip in (8, 12, 16, 20))
    # In the mirrored image, a right palm's thumb extends towards the left.
    if label == 'Right':
        count += landmarks[4].x < landmarks[3].x
    elif label == 'Left':
        count += landmarks[4].x > landmarks[3].x
    return int(count)


def draw_landmarks(frame, landmarks):
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 0, 255), -1)


def run(args):
    sound_dir = args.sounds.resolve()
    missing = [name for _, name in SWARAS.values()
               if not (sound_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f'Put the eight provided audio clips in {sound_dir}. Missing: '
            + ', '.join(missing))
    ensure_model(args.model)
    cap = None
    landmarker = None
    try:
        pygame.mixer.init()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(args.model)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError('Cannot open webcam. Close other camera apps or try --camera 1.')
        start = time.monotonic()
        last_timestamp = -1
        candidate = stable_count = 0
        candidate_since = start
        muted = False
        window = 'Hand Gesture Swara Player'
        cv2.namedWindow(window)
        print('Keep palms upright and facing the camera. Q: quit. M: mute.')
        while True:
            success, frame = cap.read()
            if not success:
                raise RuntimeError('Webcam stopped returning frames. Check its connection.')
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now = time.monotonic()
            # VIDEO mode needs strictly increasing millisecond timestamps.
            timestamp = max(last_timestamp + 1, int((now - start) * 1000))
            last_timestamp = timestamp
            result = landmarker.detect_for_video(mp_image, timestamp)
            count = 0  # Reset every frame; sum both hands.
            for i, landmarks in enumerate(result.hand_landmarks):
                label = result.handedness[i][0].category_name
                count += count_fingers(landmarks, label)
                draw_landmarks(frame, landmarks)

            # Accept only counts held for 0.25 seconds; never block the video.
            if count != candidate:
                candidate, candidate_since = count, now
            if candidate != stable_count and now - candidate_since >= args.hold:
                stable_count = candidate
                if stable_count in SWARAS and not muted:
                    _, filename = SWARAS[stable_count]
                    pygame.mixer.music.load(str(sound_dir / filename))
                    pygame.mixer.music.play()
                else:
                    pygame.mixer.music.stop()

            swara = SWARAS.get(stable_count, ('Silence', ''))[0]
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (25, 25, 25), -1)
            lines = [f'Fingers: {count} | Stable: {stable_count}',
                     f'Swara: {swara}' + (' [MUTED]' if muted else ''),
                     'Q: quit | M: mute | Keep palms upright']
            for i, text in enumerate(lines):
                cv2.putText(frame, text, (12, 28 + i * 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('m'):
                muted = not muted
                if muted:
                    pygame.mixer.music.stop()
    finally:
        if cap is not None:
            cap.release()
        if landmarker is not None:
            landmarker.close()
        cv2.destroyAllWindows()
        pygame.mixer.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--sounds', type=Path, default=ROOT / 'sounds_swaras')
    parser.add_argument('--model', type=Path, default=ROOT / 'hand_landmarker.task')
    parser.add_argument('--hold', type=float, default=0.25,
                        help='Seconds a count must remain unchanged before playing')
    args = parser.parse_args()
    if args.hold < 0:
        parser.error('--hold must be nonnegative')
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
