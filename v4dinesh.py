import cv2
import numpy as np
import mediapipe as mp
import pyaudio
import speech_recognition as sr
import threading
import time
import os
from collections import deque
from datetime import datetime
import colorsys
import math
import random

class OptimizedKineticCanvas:
    def __init__(self):
        # Initialize MediaPipe with optimized settings
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
        # Optimized hand detection settings
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.8,  # Increased for better accuracy
            min_tracking_confidence=0.7,   # Increased for smoother tracking
            model_complexity=0  # Use lighter model for speed
        )
        
        # Simplified pose detection (only when needed)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # Lightest model
            enable_segmentation=False,  # Disable segmentation for speed
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize webcam with optimized settings
        self.cap = cv2.VideoCapture(0)
        # Reduce resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Optimized canvas layers
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.body_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Optimized drawing state with smoothing
        self.finger_positions = {}  # Dict for multiple fingers
        self.smoothed_positions = {}  # Smoothed positions
        self.finger_velocities = {}
        self.drawing_states = {}  # Drawing state for each finger
        self.drawing_threshold = 1.5  # Reduced threshold
        
        # Kalman filters for smooth tracking
        self.kalman_filters = {}
        
        # Undo system (reduced size for memory)
        self.canvas_history = deque(maxlen=5)
        
        # Audio processing (simplified)
        self.audio_thread = None
        self.current_pitch = 0
        self.current_volume = 0
        self.audio_running = True
        self.audio_lock = threading.Lock()
        
        # Voice recognition (simplified)
        self.voice_thread = None
        self.recognizer = sr.Recognizer()
        self.microphone = None
        
        # Brush system
        self.brush_types = ['smooth', 'neon', 'particle']  # Reduced brush types for performance
        self.current_brush = 0
        self.brush_size_base = 3
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Frame skipping for non-essential processing
        self.frame_count = 0
        self.process_audio_every = 3  # Process audio every 3 frames
        self.process_body_every = 5   # Process body every 5 frames
        
        # Color system
        self.hue = 0.5
        
        print("Optimized Kinetic Canvas Initialized!")
        print("Controls:")
        print("- Hold finger still for 1.5s to start drawing")
        print("- Say 'Clear' to clear canvas")
        print("- Press 'c' to clear, 'b' to change brush, 's' to save, 'q' to quit")
        
        self.init_audio_simplified()
        self.init_voice_recognition_simplified()
    
    def create_kalman_filter(self):
        """Create a Kalman filter for smooth position tracking"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                           [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                          [0, 1, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        return kalman
    
    def smooth_position(self, finger_id, position):
        """Apply Kalman filtering for smooth position tracking"""
        if finger_id not in self.kalman_filters:
            self.kalman_filters[finger_id] = self.create_kalman_filter()
            self.kalman_filters[finger_id].statePre = np.array([position[0], position[1], 0, 0], dtype=np.float32)
            self.kalman_filters[finger_id].statePost = np.array([position[0], position[1], 0, 0], dtype=np.float32)
            return position
        
        # Predict and update
        self.kalman_filters[finger_id].predict()
        measurement = np.array([[position[0]], [position[1]]], dtype=np.float32)
        self.kalman_filters[finger_id].correct(measurement)
        
        # Get smoothed position
        smoothed = self.kalman_filters[finger_id].statePost
        return (int(smoothed[0]), int(smoothed[1]))
    
    def init_audio_simplified(self):
        """Initialize simplified audio processing"""
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_thread = threading.Thread(target=self.audio_processing_simplified)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        except Exception as e:
            print(f"Audio initialization failed: {e}")
            self.audio = None
    
    def audio_processing_simplified(self):
        """Simplified audio processing"""
        if not self.audio:
            return
            
        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=22050,  # Reduced sample rate
                input=True,
                frames_per_buffer=2048  # Smaller buffer
            )
            
            while self.audio_running:
                try:
                    data = np.frombuffer(stream.read(2048, exception_on_overflow=False), dtype=np.float32)
                    
                    # Quick volume calculation
                    with self.audio_lock:
                        self.current_volume = np.sqrt(np.mean(data**2)) * 50
                        
                        # Simplified pitch detection
                        if self.current_volume > 0.01:
                            pitch = self.detect_pitch_simplified(data)
                            if pitch > 0:
                                self.hue = max(0, min(1, (pitch - 100) / 400))
                                self.current_pitch = pitch
                    
                    time.sleep(0.05)  # Reduced sleep time
                except:
                    continue
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Audio processing error: {e}")
    
    def detect_pitch_simplified(self, audio_data):
        """Simplified pitch detection"""
        # Simple zero-crossing rate method
        zero_crossings = np.where(np.diff(np.signbit(audio_data)))[0]
        if len(zero_crossings) > 0:
            return len(zero_crossings) * 22050 / (2 * len(audio_data))
        return 0
    
    def init_voice_recognition_simplified(self):
        """Simplified voice recognition"""
        try:
            self.microphone = sr.Microphone()
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            
            self.voice_thread = threading.Thread(target=self.voice_recognition_simplified)
            self.voice_thread.daemon = True
            self.voice_thread.start()
        except Exception as e:
            print(f"Voice recognition failed: {e}")
            self.microphone = None
    
    def voice_recognition_simplified(self):
        """Simplified voice recognition loop"""
        if not self.microphone:
            return
            
        while self.audio_running:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=1)
                
                try:
                    command = self.recognizer.recognize_google(audio, show_all=False).lower()
                    if "clear" in command:
                        self.clear_canvas()
                        print("Voice: Clear")
                except:
                    pass
                    
            except:
                pass
    
    def get_dynamic_color(self):
        """Get color based on audio input"""
        with self.audio_lock:
            hue = self.hue
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    
    def get_dynamic_brush_size(self):
        """Get brush size based on volume"""
        with self.audio_lock:
            volume = self.current_volume
        volume_factor = min(volume / 10.0, 2.0)
        return max(2, int(self.brush_size_base * (0.5 + volume_factor)))
    
    def get_all_finger_tips(self, hand_landmarks, hand_label):
        """Get positions of all relevant finger tips"""
        fingers = {}
        if hand_landmarks:
            # Index finger (8), Middle finger (12), Ring finger (16), Pinky (20)
            finger_indices = [8, 12, 16, 20]
            finger_names = ['index', 'middle', 'ring', 'pinky']
            
            for i, finger_idx in enumerate(finger_indices):
                landmark = hand_landmarks.landmark[finger_idx]
                x = int(landmark.x * self.width)
                y = int(landmark.y * self.height)
                
                # Create unique finger ID
                finger_id = f"{hand_label}_{finger_names[i]}"
                
                # Apply smoothing
                smoothed_pos = self.smooth_position(finger_id, (x, y))
                fingers[finger_id] = smoothed_pos
        
        return fingers
    
    def draw_smooth_brush(self, start_pos, end_pos, color, size):
        """Optimized smooth brush"""
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
    
    def draw_neon_brush(self, start_pos, end_pos, color, size):
        """Optimized neon brush"""
        # Main line
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
        # Single glow layer
        glow_color = tuple(int(c * 0.5) for c in color)
        cv2.line(self.drawing_canvas, start_pos, end_pos, glow_color, size + 2, cv2.LINE_AA)
    
    def draw_particle_brush(self, start_pos, end_pos, color, size):
        """Optimized particle brush"""
        # Main line
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
        
        # Add few particles
        distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        num_particles = max(1, int(distance / 20))
        
        for _ in range(min(num_particles, 3)):  # Limit particles
            offset_x = random.randint(-size, size)
            offset_y = random.randint(-size, size)
            particle_pos = (end_pos[0] + offset_x, end_pos[1] + offset_y)
            cv2.circle(self.drawing_canvas, particle_pos, random.randint(1, 2), color, -1)
    
    def apply_brush(self, start_pos, end_pos):
        """Apply current brush effect"""
        color = self.get_dynamic_color()
        size = self.get_dynamic_brush_size()
        
        brush_name = self.brush_types[self.current_brush]
        
        if brush_name == 'smooth':
            self.draw_smooth_brush(start_pos, end_pos, color, size)
        elif brush_name == 'neon':
            self.draw_neon_brush(start_pos, end_pos, color, size)
        elif brush_name == 'particle':
            self.draw_particle_brush(start_pos, end_pos, color, size)
    
    def process_hands_optimized(self, frame):
        """Optimized hand processing"""
        # Resize frame for processing to improve speed
        small_frame = cv2.resize(frame, (320, 240))
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        hand_results = self.hands.process(rgb_small)
        current_time = time.time()
        
        # Scale factor for coordinate conversion
        scale_x = self.width / 320
        scale_y = self.height / 240
        
        active_fingers = set()
        
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()
                
                # Get all finger positions
                fingers = self.get_all_finger_tips(hand_landmarks, hand_label)
                
                # Scale positions back to full resolution
                for finger_id, pos in fingers.items():
                    scaled_pos = (int(pos[0] * scale_x), int(pos[1] * scale_y))
                    fingers[finger_id] = scaled_pos
                    active_fingers.add(finger_id)
                    
                    # Initialize finger state if new
                    if finger_id not in self.drawing_states:
                        self.drawing_states[finger_id] = {
                            'start_time': current_time,
                            'is_drawing': False,
                            'last_pos': scaled_pos
                        }
                    
                    # Update finger position history
                    self.finger_positions[finger_id] = scaled_pos
                    
                    # Check for drawing activation
                    finger_state = self.drawing_states[finger_id]
                    time_held = current_time - finger_state['start_time']
                    
                    # Calculate movement to detect stillness
                    movement = np.sqrt((scaled_pos[0] - finger_state['last_pos'][0])**2 + 
                                     (scaled_pos[1] - finger_state['last_pos'][1])**2)
                    
                    if movement > 10:  # Reset timer if finger moved significantly
                        finger_state['start_time'] = current_time
                        finger_state['last_pos'] = scaled_pos
                    
                    # Activate drawing mode
                    if time_held >= self.drawing_threshold and not finger_state['is_drawing']:
                        finger_state['is_drawing'] = True
                        if not any(state['is_drawing'] for state in self.drawing_states.values()):
                            self.save_canvas_state()
                        print(f"Drawing activated: {finger_id}")
                    
                    # Draw if in drawing mode
                    if finger_state['is_drawing'] and 'prev_draw_pos' in finger_state:
                        self.apply_brush(finger_state['prev_draw_pos'], scaled_pos)
                    
                    if finger_state['is_drawing']:
                        finger_state['prev_draw_pos'] = scaled_pos
                        # Visual indicator
                        color = self.get_dynamic_color()
                        cv2.circle(frame, scaled_pos, 8, color, 2)
                    else:
                        # Show activation progress
                        progress = min(1.0, time_held / self.drawing_threshold)
                        cv2.circle(frame, scaled_pos, int(5 + 5 * progress), (0, 255, 0), 1)
        
        # Clean up inactive fingers
        inactive_fingers = set(self.drawing_states.keys()) - active_fingers
        for finger_id in inactive_fingers:
            if self.drawing_states[finger_id]['is_drawing']:
                print(f"Drawing deactivated: {finger_id}")
            del self.drawing_states[finger_id]
            if finger_id in self.finger_positions:
                del self.finger_positions[finger_id]
            if finger_id in self.kalman_filters:
                del self.kalman_filters[finger_id]
        
        return frame
    
    def save_canvas_state(self):
        """Save current canvas state for undo"""
        self.canvas_history.append(self.drawing_canvas.copy())
    
    def clear_canvas(self):
        """Clear canvas"""
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.canvas_history.clear()
        print("Canvas cleared")
    
    def save_image(self):
        """Save current canvas"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kinetic_art_{timestamp}.jpg"
        cv2.imwrite(filename, self.drawing_canvas)
        print(f"Saved: {filename}")
    
    def calculate_fps(self):
        """Calculate current FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
    
    def draw_ui_optimized(self, frame):
        """Draw optimized UI"""
        # FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Active fingers count
        active_count = len([state for state in self.drawing_states.values() if state['is_drawing']])
        cv2.putText(frame, f"Drawing: {active_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Brush type
        cv2.putText(frame, f"Brush: {self.brush_types[self.current_brush]}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Audio info (simplified)
        with self.audio_lock:
            volume = self.current_volume
            pitch = self.current_pitch
        cv2.putText(frame, f"Vol: {volume:.1f} Hz: {pitch:.0f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Optimized main loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.frame_count += 1
                frame = cv2.flip(frame, 1)
                
                # Process hands every frame for responsiveness
                frame = self.process_hands_optimized(frame)
                
                # Combine drawing canvas with camera feed
                alpha = 0.7
                display_frame = cv2.addWeighted(frame, alpha, self.drawing_canvas, 1.0, 0)
                
                # Draw UI
                self.draw_ui_optimized(display_frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                cv2.imshow('Optimized Kinetic Canvas', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_canvas()
                elif key == ord('b'):
                    self.current_brush = (self.current_brush + 1) % len(self.brush_types)
                    print(f"Brush: {self.brush_types[self.current_brush]}")
                elif key == ord('s'):
                    self.save_image()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.audio_running = False
        
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        canvas = OptimizedKineticCanvas()
        canvas.run()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python mediapipe pyaudio SpeechRecognition numpy")
    except Exception as e:
        print(f"Error: {e}")