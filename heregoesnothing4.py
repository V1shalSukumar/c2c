import cv2
import mediapipe as mp
import numpy as np
import time


class GestureDrawing:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        # Mediapipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Canvas setup
        self.canvas = None
        self.prev_x, self.prev_y = None, None

        # Modes
        self.drawing = False
        self.kaleidoscope_mode = False
        self.grayscale_mode = False

        # Timing
        self.last_toggle_time = 0
        self.last_gesture_time = 0

        # Finger tips indices
        self.finger_tips = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20,
        }

    def is_finger_extended(self, hand_landmarks, finger_name):
        tip_id = self.finger_tips[finger_name]
        pip_id = tip_id - 2
        tip = hand_landmarks.landmark[tip_id]
        pip = hand_landmarks.landmark[pip_id]

        if finger_name == "thumb":
            return tip.x < pip.x
        else:
            return tip.y < pip.y

    def detect_peace_sign(self, hand_landmarks):
        index_up = self.is_finger_extended(hand_landmarks, "index")
        middle_up = self.is_finger_extended(hand_landmarks, "middle")
        ring_down = not self.is_finger_extended(hand_landmarks, "ring")
        pinky_down = not self.is_finger_extended(hand_landmarks, "pinky")

        return index_up and middle_up and ring_down and pinky_down

    def detect_grayscale_gesture(self, hand_landmarks):
        """Detect if all 5 fingers are close together and hand moves away."""
        if not hand_landmarks:
            return False

        tips = [
            hand_landmarks.landmark[self.finger_tips[f]] for f in self.finger_tips
        ]

        # All fingers must be extended
        all_extended = all(
            self.is_finger_extended(hand_landmarks, f) for f in self.finger_tips
        )
        if not all_extended:
            return False

        # Cluster check
        cx = np.mean([t.x for t in tips])
        cy = np.mean([t.y for t in tips])
        cluster_dist = np.mean(
            [np.sqrt((t.x - cx) ** 2 + (t.y - cy) ** 2) for t in tips]
        )
        if cluster_dist > 0.05:
            return False

        # Wrist z movement
        wrist_z = hand_landmarks.landmark[0].z
        if not hasattr(self, "prev_wrist_z"):
            self.prev_wrist_z = wrist_z
            return False

        moving_away = wrist_z - self.prev_wrist_z > 0.02
        self.prev_wrist_z = wrist_z

        # Debounce
        if moving_away and (time.time() - self.last_gesture_time > 2):
            self.last_gesture_time = time.time()
            return True

        return False

    def draw_kaleidoscope(self, x, y, frame, color=(255, 0, 255)):
        h, w, _ = frame.shape
        cx, cy = w // 2, h // 2
        dx, dy = x - cx, y - cy

        for i in range(4):
            px = cx + (dx if i % 2 == 0 else -dx)
            py = cy + (dy if i < 2 else -dy)

            if 0 <= px < w and 0 <= py < h:
                if self.prev_x is not None and self.prev_y is not None:
                    pdx, pdy = self.prev_x - cx, self.prev_y - cy
                    ppx = cx + (pdx if i % 2 == 0 else -pdx)
                    ppy = cy + (pdy if i < 2 else -pdy)
                    cv2.line(self.canvas, (ppx, ppy), (px, py), color, 3)

    def process_hands(self, frame, results):
        h, w, _ = frame.shape

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw landmarks
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

            # Toggle kaleidoscope mode with peace sign
            if self.detect_peace_sign(hand_landmarks):
                if time.time() - self.last_toggle_time > 1:
                    self.kaleidoscope_mode = not self.kaleidoscope_mode
                    print(f"Kaleidoscope mode: {self.kaleidoscope_mode}")
                    self.last_toggle_time = time.time()

            # Toggle grayscale mode with 5-finger pull-away
            if self.detect_grayscale_gesture(hand_landmarks):
                self.grayscale_mode = not self.grayscale_mode
                print(f"Grayscale mode: {self.grayscale_mode}")

            # Index finger drawing
            index_finger = hand_landmarks.landmark[self.finger_tips["index"]]
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            fingers_extended = [
                self.is_finger_extended(hand_landmarks, f)
                for f in self.finger_tips
                if f != "thumb"
            ]

            if fingers_extended[0] and not any(fingers_extended[1:]):
                self.drawing = True
                if self.kaleidoscope_mode:
                    self.draw_kaleidoscope(x, y, frame)
                else:
                    if self.prev_x is not None and self.prev_y is not None:
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), (255, 0, 255), 3)
                self.prev_x, self.prev_y = x, y
            else:
                self.drawing = False
                self.prev_x, self.prev_y = None, None

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                self.process_hands(frame, results)

            # Combine canvas with camera feed
            mask = self.canvas > 0
            display_frame = frame.copy()
            display_frame[mask] = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)[mask]

            # Apply grayscale effect
            if self.grayscale_mode:
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # UI indicators
            if self.kaleidoscope_mode:
                cv2.putText(display_frame, "Kaleidoscope Mode", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if self.grayscale_mode:
                cv2.putText(display_frame, "Grayscale Mode", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            cv2.imshow("Gesture Drawing", display_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    GestureDrawing().run()
