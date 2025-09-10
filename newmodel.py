import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

class KineticArtCanvas:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize hand and pose detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Canvas and drawing state
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Finger tracking variables
        self.finger_positions = deque(maxlen=20)  # Store recent finger positions for smoothing
        self.finger_start_time = None
        self.is_drawing_mode = False
        self.drawing_threshold = 7.0  # seconds
        self.last_drawing_pos = None
        
        # Drawing parameters
        self.brush_size = 3
        self.drawing_color = (255, 255, 255)  # White
        self.smoothing_factor = 0.7
        
        print(f"Camera resolution: {self.width}x{self.height}")
        print("Instructions:")
        print("- Move around to see your body silhouette")
        print("- Hold your index finger steady for 7+ seconds to enter drawing mode")
        print("- In drawing mode, move your finger to draw smooth lines")
        print("- Press 'q' to quit, 'c' to clear canvas, 'd' to clear drawings only")
        print("- Press 'b' to change brush size, 'r' to change color")
        
        self.colors = [(255, 255, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
        self.color_index = 0
        
    def get_finger_tip(self, hand_landmarks):
        """Get the tip of the index finger"""
        if hand_landmarks:
            # Index finger tip is landmark 8
            finger_tip = hand_landmarks.landmark[8]
            x = int(finger_tip.x * self.width)
            y = int(finger_tip.y * self.height)
            return (x, y)
        return None
    
    def smooth_position(self, new_pos):
        """Smooth the finger position using recent history"""
        if not self.finger_positions:
            return new_pos
        
        # Average recent positions for smoothing
        recent_positions = list(self.finger_positions)[-5:]  # Use last 5 positions
        avg_x = sum(pos[0] for pos in recent_positions) / len(recent_positions)
        avg_y = sum(pos[1] for pos in recent_positions) / len(recent_positions)
        
        # Apply smoothing factor
        smooth_x = int(self.smoothing_factor * avg_x + (1 - self.smoothing_factor) * new_pos[0])
        smooth_y = int(self.smoothing_factor * avg_y + (1 - self.smoothing_factor) * new_pos[1])
        
        return (smooth_x, smooth_y)
    
    def draw_smooth_line(self, start_pos, end_pos):
        """Draw a smooth line between two points"""
        if start_pos and end_pos:
            # Draw line with anti-aliasing
            cv2.line(self.drawing_canvas, start_pos, end_pos, self.drawing_color, self.brush_size, cv2.LINE_AA)
            
            # Add some circular brush strokes for smoother appearance
            cv2.circle(self.drawing_canvas, end_pos, self.brush_size//2, self.drawing_color, -1)
    
    def process_hands(self, frame):
        """Process hand landmarks and handle drawing logic"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        
        current_time = time.time()
        finger_tip = None
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                finger_tip = self.get_finger_tip(hand_landmarks)
                break  # Use first detected hand
        
        # Handle finger tracking and drawing mode
        if finger_tip:
            self.finger_positions.append(finger_tip)
            
            # Check if finger has been steady (drawing mode activation)
            if self.finger_start_time is None:
                self.finger_start_time = current_time
            
            # Check if we should enter drawing mode
            time_held = current_time - self.finger_start_time
            if time_held >= self.drawing_threshold and not self.is_drawing_mode:
                self.is_drawing_mode = True
                self.last_drawing_pos = self.smooth_position(finger_tip)
                print("Drawing mode activated!")
            
            # If in drawing mode, draw smooth lines
            if self.is_drawing_mode:
                smooth_pos = self.smooth_position(finger_tip)
                if self.last_drawing_pos:
                    self.draw_smooth_line(self.last_drawing_pos, smooth_pos)
                self.last_drawing_pos = smooth_pos
                
                # Draw current finger position indicator
                cv2.circle(frame, smooth_pos, 10, (0, 255, 0), 2)
                cv2.circle(frame, smooth_pos, 3, (0, 255, 0), -1)
            else:
                # Show countdown for drawing mode
                if time_held > 1.0:  # Show countdown after 1 second
                    countdown = self.drawing_threshold - time_held
                    if countdown > 0:
                        cv2.putText(frame, f"Hold for: {countdown:.1f}s", 
                                  (finger_tip[0] - 50, finger_tip[1] - 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Show finger position
                cv2.circle(frame, finger_tip, 8, (255, 255, 0), 2)
        else:
            # No finger detected, reset drawing mode
            if self.is_drawing_mode:
                print("Drawing mode deactivated!")
            self.finger_start_time = None
            self.is_drawing_mode = False
            self.last_drawing_pos = None
            self.finger_positions.clear()
        
        return frame
    
    def process_body(self, frame):
        """Process body silhouette"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask > 0.5
            silhouette = np.zeros_like(frame)
            silhouette[mask] = [100, 100, 100]  # Dim gray for body
            return silhouette
        
        return np.zeros_like(frame)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # Process hands for drawing
            frame = self.process_hands(frame)
            
            # Process body silhouette (only if not in drawing mode)
            if not self.is_drawing_mode:
                silhouette = self.process_body(original_frame)
                # Add body silhouette to canvas with persistence
                self.canvas = cv2.addWeighted(self.canvas, 0.95, silhouette, 0.3, 0)
            
            # Combine all layers: canvas + drawings + live view
            display_frame = cv2.addWeighted(self.canvas, 0.7, frame, 0.3, 0)
            display_frame = cv2.addWeighted(display_frame, 0.8, self.drawing_canvas, 1.0, 0)
            
            # Add mode indicator
            mode_text = "DRAWING MODE" if self.is_drawing_mode else "BODY TRACKING"
            color = (0, 255, 0) if self.is_drawing_mode else (255, 255, 255)
            cv2.putText(display_frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add brush info when drawing
            if self.is_drawing_mode:
                cv2.putText(display_frame, f"Brush: {self.brush_size}px", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.drawing_color, 2)
            
            cv2.imshow('Kinetic Art Canvas', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear everything
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            elif key == ord('d'):
                # Clear drawings only
                self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            elif key == ord('b'):
                # Change brush size
                self.brush_size = (self.brush_size + 2) % 12 + 1
                print(f"Brush size: {self.brush_size}")
            elif key == ord('r'):
                # Change color
                self.color_index = (self.color_index + 1) % len(self.colors)
                self.drawing_color = self.colors[self.color_index]
                color_names = ["White", "Cyan", "Magenta", "Yellow", "Green", "Red"]
                print(f"Drawing color: {color_names[self.color_index]}")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

# Simpler version focusing just on finger drawing
class SimpleFingerCanvas:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing = False
        self.last_pos = None
        
    def run(self):
        print("Simple Finger Drawing - Point your index finger and move to draw")
        print("Press 'q' to quit, 'c' to clear")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get index finger tip
                    finger_tip = hand_landmarks.landmark[8]
                    x, y = int(finger_tip.x * self.width), int(finger_tip.y * self.height)
                    
                    if self.last_pos:
                        cv2.line(self.canvas, self.last_pos, (x, y), (255, 255, 255), 3)
                    
                    self.last_pos = (x, y)
                    cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)
            else:
                self.last_pos = None
            
            # Combine frame and canvas
            display = cv2.addWeighted(frame, 0.5, self.canvas, 0.5, 0)
            cv2.imshow('Simple Finger Canvas', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose version:")
    print("1. Full Kinetic Art Canvas (body tracking + finger drawing)")
    print("2. Simple Finger Drawing Canvas")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "2":
            canvas = SimpleFingerCanvas()
        else:
            canvas = KineticArtCanvas()
        
        canvas.run()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python mediapipe numpy")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and accessible")