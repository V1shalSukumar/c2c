import cv2
import numpy as np
import mediapipe as mp

class KineticCanvas:
    def __init__(self):
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Get camera dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera resolution: {self.width}x{self.height}")
        print("Press 'q' to quit, 'c' to clear canvas")
        
        # Canvas for persistent drawing
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
    def process_frame(self, frame):
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(rgb_frame)
        
        # Create mask from segmentation
        if results.segmentation_mask is not None:
            # Convert segmentation mask to binary
            mask = results.segmentation_mask > 0.5
            
            # Create white silhouette where person is detected
            silhouette = np.zeros_like(frame)
            silhouette[mask] = [255, 255, 255]  # White where person is
            
            return silhouette
        
        return np.zeros_like(frame)
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Mirror the frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)
            
            # Process frame to get silhouette
            silhouette = self.process_frame(frame)
            
            # Add current silhouette to canvas with some persistence
            # This creates a trailing effect
            self.canvas = cv2.addWeighted(self.canvas, 0.95, silhouette, 0.3, 0)
            
            # Display the kinetic canvas
            cv2.imshow('Kinetic Canvas', self.canvas)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear canvas
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            elif key == ord('r'):
                # Reset to real-time only (no persistence)
                self.canvas = silhouette.copy()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

# Alternative version using background subtraction for motion detection
class MotionKineticCanvas:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            varThreshold=50
        )
        
        # Get camera dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera resolution: {self.width}x{self.height}")
        print("Press 'q' to quit, 'c' to clear canvas")
        print("Stand still for a few seconds to calibrate background")
        
        # Canvas for drawing
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
    def process_motion(self, frame):
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to make motion areas more visible
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Create white silhouette where motion is detected
        motion_silhouette = np.zeros_like(frame)
        motion_silhouette[fg_mask > 0] = [255, 255, 255]
        
        return motion_silhouette
    
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Process motion
            motion = self.process_motion(frame)
            
            # Add to canvas with persistence
            self.canvas = cv2.addWeighted(self.canvas, 0.92, motion, 0.4, 0)
            
            # Display
            cv2.imshow('Motion Kinetic Canvas', self.canvas)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Choose version:")
    print("1. Body pose detection (requires mediapipe)")
    print("2. Motion detection (basic, no extra dependencies)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "1":
            canvas = KineticCanvas()
        else:
            canvas = MotionKineticCanvas()
        
        canvas.run()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python mediapipe numpy")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera is connected and accessible")