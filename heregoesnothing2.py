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
from tkinter import filedialog
import tkinter as tk

class AdvancedKineticCanvas:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        
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
        
        # Background image system
        self.background_image = None
        self.original_background = None
        self.background_enabled = False
        self.background_opacity = 0.8
        
        # Canvas layers
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.body_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Multi-finger drawing state
        # Each finger tracker stores: positions, velocities, start_time, drawing_mode, last_pos
        self.finger_trackers = {}
        self.drawing_threshold = 1.5  # Reduced to 1.5 seconds
        
        # Finger tip landmark indices for each finger
        self.finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        # Finger MCP (base) landmark indices for extension detection
        self.finger_mcps = {
            'thumb': 3,
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }
        
        # Undo system
        self.canvas_history = deque(maxlen=20)
        self.swipe_detection = {'left': False, 'start_x': None, 'frames': 0}
        
        # Audio processing
        self.audio_thread = None
        self.current_pitch = 0
        self.current_volume = 0
        self.audio_running = True
        
        # Voice recognition
        self.voice_thread = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Brush system
        self.brush_types = ['neon', 'fire', 'smoke', 'watercolor', 'pixelated']
        self.current_brush = 0
        self.brush_size_base = 5
        
        # Kaleidoscope mode
        self.kaleidoscope_mode = False
        self.kaleidoscope_segments = 8
        
        # Video recording
        self.video_writer = None
        self.recording = False
        self.recording_frames = []
        
        # Color system - pitch based with finger variation
        self.hue = 0.0
        
        print("Advanced Multi-Finger Kinetic Canvas with Background Support Initialized!")
        print("Features:")
        print("- Paint with multiple open fingers simultaneously")
        print("- Each finger gets a slightly different color")
        print("- Pitch controls base color, Volume controls brush size")
        print("- Say 'Clear' to clear canvas")
        print("- Swipe left to undo")
        print("- Press 'k' for kaleidoscope mode")
        print("- Press 'b' to change brush type")
        print("- Press 'i' to load background image")
        print("- Press 't' to toggle background visibility")
        print("- Press '+/-' to adjust background opacity")
        print("- Press 's' to save image")
        print("- Press 'v' to start/stop video recording")
        print("- Press 'q' to quit")
        
        self.init_audio()
        self.init_voice_recognition()
    
    def load_background_image(self):
        """Load background image using file dialog"""
        try:
            # Create a temporary tkinter root for file dialog
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            file_path = filedialog.askopenfilename(
                title="Select Background Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg *.jpeg"),
                    ("All files", "*.*")
                ]
            )
            
            root.destroy()
            
            if file_path:
                # Load and resize image to match camera resolution
                img = cv2.imread(file_path)
                if img is not None:
                    self.original_background = cv2.resize(img, (self.width, self.height))
                    self.background_image = self.original_background.copy()
                    self.background_enabled = True
                    print(f"Background loaded: {os.path.basename(file_path)}")
                    return True
                else:
                    print("Error: Could not load image file")
                    return False
            else:
                print("No image selected")
                return False
        
        except Exception as e:
            print(f"Error loading background image: {e}")
            return False
    
    def adjust_background_opacity(self, delta):
        """Adjust background image opacity"""
        if self.background_enabled and self.original_background is not None:
            self.background_opacity = max(0.0, min(1.0, self.background_opacity + delta))
            print(f"Background opacity: {self.background_opacity:.2f}")
    
    def apply_background(self, frame):
        """Apply background image to the frame"""
        if not self.background_enabled or self.background_image is None:
            return frame
        
        # Create a blended background
        background_weighted = cv2.addWeighted(
            self.background_image, self.background_opacity,
            np.zeros_like(self.background_image), 1 - self.background_opacity, 0
        )
        
        # Get pose results for segmentation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        if pose_results.segmentation_mask is not None:
            # Create person mask
            mask = pose_results.segmentation_mask > 0.5
            
            # Create 3-channel mask
            mask_3d = np.stack([mask] * 3, axis=-1)
            
            # Combine: background where no person, original frame where person is
            result = np.where(mask_3d, frame, background_weighted)
            
            return result.astype(np.uint8)
        else:
            # If no segmentation available, blend the entire frame with background
            return cv2.addWeighted(frame, 0.7, background_weighted, 0.3, 0)
    
    def init_audio(self):
        """Initialize audio processing for pitch and volume detection"""
        try:
            self.audio = pyaudio.PyAudio()
            self.audio_thread = threading.Thread(target=self.audio_processing_loop)
            self.audio_thread.daemon = True
            self.audio_thread.start()
        except Exception as e:
            print(f"Audio initialization failed: {e}")
    
    def audio_processing_loop(self):
        """Process audio for pitch and volume"""
        try:
            stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=4096
            )
            
            while self.audio_running:
                try:
                    data = np.frombuffer(stream.read(4096, exception_on_overflow=False), dtype=np.float32)
                    
                    # Calculate volume (RMS)
                    self.current_volume = np.sqrt(np.mean(data**2)) * 100
                    
                    # Calculate pitch using autocorrelation
                    pitch = self.detect_pitch(data, 44100)
                    if pitch > 0:
                        # Map pitch to hue (80Hz-800Hz -> 0-1)
                        self.hue = max(0, min(1, (pitch - 80) / 720))
                        self.current_pitch = pitch
                    
                    time.sleep(0.1)
                except Exception as e:
                    continue
            
            stream.stop_stream()
            stream.close()
            
        except Exception as e:
            print(f"Audio processing error: {e}")
    
    def detect_pitch(self, audio_data, sample_rate):
        """Simple pitch detection using autocorrelation"""
        # Apply window
        windowed = audio_data * np.hanning(len(audio_data))
        
        # Autocorrelation
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find the peak (ignore first sample)
        min_period = sample_rate // 800  # Max 800 Hz
        max_period = sample_rate // 80   # Min 80 Hz
        
        if len(autocorr) < max_period:
            return 0
        
        peak = np.argmax(autocorr[min_period:max_period]) + min_period
        
        if peak > 0 and autocorr[peak] > 0.3 * autocorr[0]:
            return sample_rate / peak
        
        return 0
    
    def init_voice_recognition(self):
        """Initialize voice command recognition"""
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            self.voice_thread = threading.Thread(target=self.voice_recognition_loop)
            self.voice_thread.daemon = True
            self.voice_thread.start()
        except Exception as e:
            print(f"Voice recognition initialization failed: {e}")
    
    def voice_recognition_loop(self):
        """Listen for voice commands"""
        while self.audio_running:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    if "clear" in command:
                        self.clear_canvas()
                        print("Voice command: Clear")
                    elif "kaleidoscope" in command:
                        self.kaleidoscope_mode = not self.kaleidoscope_mode
                        print(f"Voice command: Kaleidoscope {'ON' if self.kaleidoscope_mode else 'OFF'}")
                    elif "save" in command:
                        self.save_image()
                        print("Voice command: Save")
                    elif "background" in command:
                        self.background_enabled = not self.background_enabled
                        print(f"Voice command: Background {'ON' if self.background_enabled else 'OFF'}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
                    
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                continue
    
    def get_dynamic_color(self, finger_name):
        """Get color based on current pitch with finger variation"""
        # Base hue from pitch
        base_hue = self.hue
        
        # Add finger-specific hue offset
        finger_offsets = {
            'thumb': 0.0,
            'index': 0.1,
            'middle': 0.2,
            'ring': 0.3,
            'pinky': 0.4
        }
        
        finger_hue = (base_hue + finger_offsets.get(finger_name, 0.0)) % 1.0
        
        # Convert hue to RGB
        rgb = colorsys.hsv_to_rgb(finger_hue, 0.9, 1.0)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR for OpenCV
    
    def get_dynamic_brush_size(self):
        """Get brush size based on volume"""
        volume_factor = min(self.current_volume / 20.0, 3.0)  # Cap at 3x
        return int(self.brush_size_base * (0.5 + volume_factor))
    
    def is_finger_extended(self, hand_landmarks, finger_name):
        """Check if a finger is extended (open)"""
        if not hand_landmarks:
            return False
        
        tip_idx = self.finger_tips[finger_name]
        
        # Special case for thumb - check horizontal distance from wrist
        if finger_name == 'thumb':
            wrist = hand_landmarks.landmark[0]
            tip = hand_landmarks.landmark[tip_idx]
            mcp = hand_landmarks.landmark[self.finger_mcps[finger_name]]
            
            # Thumb is extended if tip is farther from wrist than MCP
            wrist_to_tip = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            wrist_to_mcp = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
            return wrist_to_tip > wrist_to_mcp * 1.2
        
        # For other fingers, check if tip is higher than MCP
        else:
            tip = hand_landmarks.landmark[tip_idx]
            mcp = hand_landmarks.landmark[self.finger_mcps[finger_name]]
            
            # Finger is extended if tip is significantly higher than MCP
            return tip.y < mcp.y - 0.02  # 0.02 is threshold for extension
    
    def get_extended_fingers(self, hand_landmarks):
        """Get positions of all extended fingers"""
        extended_fingers = {}
        
        if not hand_landmarks:
            return extended_fingers
        
        for finger_name, tip_idx in self.finger_tips.items():
            if self.is_finger_extended(hand_landmarks, finger_name):
                tip = hand_landmarks.landmark[tip_idx]
                x = int(tip.x * self.width)
                y = int(tip.y * self.height)
                extended_fingers[finger_name] = (x, y)
        
        return extended_fingers
    
    def calculate_velocity(self, finger_id, current_pos):
        """Calculate finger movement velocity"""
        if finger_id not in self.finger_trackers:
            return 0
        
        positions = self.finger_trackers[finger_id]['positions']
        
        if len(positions) < 2:
            return 0
        
        prev_pos = positions[-1]
        distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        
        velocities = self.finger_trackers[finger_id]['velocities']
        velocities.append(distance)
        
        return np.mean(list(velocities)) if velocities else 0
    
    def draw_neon_brush(self, start_pos, end_pos, color, size, velocity):
        """Neon glow effect brush"""
        # Main line
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
        
        # Glow effects
        glow_color = tuple(int(c * 0.6) for c in color)
        cv2.line(self.drawing_canvas, start_pos, end_pos, glow_color, size + 4, cv2.LINE_AA)
        
        glow_color2 = tuple(int(c * 0.3) for c in color)
        cv2.line(self.drawing_canvas, start_pos, end_pos, glow_color2, size + 8, cv2.LINE_AA)
    
    def draw_fire_brush(self, start_pos, end_pos, color, size, velocity):
        """Fire effect brush"""
        # Main stroke
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
        
        # Add flame particles based on velocity
        num_particles = int(velocity / 2) + 1
        for _ in range(num_particles):
            offset_x = random.randint(-size, size)
            offset_y = random.randint(-size, size)
            particle_pos = (end_pos[0] + offset_x, end_pos[1] + offset_y)
            
            # Fire colors (orange/red/yellow)
            fire_colors = [(0, 165, 255), (0, 100, 255), (0, 255, 255)]  # BGR
            fire_color = random.choice(fire_colors)
            
            cv2.circle(self.drawing_canvas, particle_pos, random.randint(1, 3), fire_color, -1)
    
    def draw_smoke_brush(self, start_pos, end_pos, color, size, velocity):
        """Smoke effect brush"""
        # Gray smoke color
        smoke_color = (128, 128, 128)
        
        # Main line with transparency effect
        overlay = self.drawing_canvas.copy()
        cv2.line(overlay, start_pos, end_pos, smoke_color, size + 2, cv2.LINE_AA)
        
        # Add wispy effects
        for i in range(3):
            offset = random.randint(-size//2, size//2)
            wispy_start = (start_pos[0] + offset, start_pos[1] + offset)
            wispy_end = (end_pos[0] + offset, end_pos[1] + offset)
            cv2.line(overlay, wispy_start, wispy_end, smoke_color, 1, cv2.LINE_AA)
        
        # Blend with main canvas
        cv2.addWeighted(self.drawing_canvas, 0.7, overlay, 0.3, 0, self.drawing_canvas)
    
    def draw_watercolor_brush(self, start_pos, end_pos, color, size, velocity):
        """Watercolor effect brush"""
        if velocity > 15:  # Fast movement - splatter
            # Create splatter effect
            num_drops = int(velocity / 5)
            for _ in range(num_drops):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(size, size * 3)
                drop_x = int(end_pos[0] + distance * math.cos(angle))
                drop_y = int(end_pos[1] + distance * math.sin(angle))
                drop_size = random.randint(1, size//2)
                
                # Vary color slightly
                varied_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in color)
                cv2.circle(self.drawing_canvas, (drop_x, drop_y), drop_size, varied_color, -1)
        else:  # Slow movement - smooth watercolor
            # Soft brush effect
            overlay = self.drawing_canvas.copy()
            cv2.line(overlay, start_pos, end_pos, color, size, cv2.LINE_AA)
            cv2.addWeighted(self.drawing_canvas, 0.8, overlay, 0.2, 0, self.drawing_canvas)
    
    def draw_pixelated_brush(self, start_pos, end_pos, color, size, velocity):
        """Pixelated/retro brush"""
        # Draw pixelated line
        pixel_size = max(2, size // 2)
        
        # Calculate line points
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = int(np.sqrt(dx*dx + dy*dy))
        
        if distance > 0:
            for i in range(0, distance, pixel_size):
                t = i / distance
                x = int(start_pos[0] + t * dx)
                y = int(start_pos[1] + t * dy)
                
                # Snap to pixel grid
                x = (x // pixel_size) * pixel_size
                y = (y // pixel_size) * pixel_size
                
                cv2.rectangle(self.drawing_canvas, 
                            (x, y), 
                            (x + pixel_size, y + pixel_size), 
                            color, -1)
    
    def apply_brush(self, start_pos, end_pos, color, velocity):
        """Apply current brush effect with specific color"""
        size = self.get_dynamic_brush_size()
        
        brush_name = self.brush_types[self.current_brush]
        
        if brush_name == 'neon':
            self.draw_neon_brush(start_pos, end_pos, color, size, velocity)
        elif brush_name == 'fire':
            self.draw_fire_brush(start_pos, end_pos, color, size, velocity)
        elif brush_name == 'smoke':
            self.draw_smoke_brush(start_pos, end_pos, color, size, velocity)
        elif brush_name == 'watercolor':
            self.draw_watercolor_brush(start_pos, end_pos, color, size, velocity)
        elif brush_name == 'pixelated':
            self.draw_pixelated_brush(start_pos, end_pos, color, size, velocity)
    
    def detect_swipe_left(self, hand_landmarks):
        """Detect left swipe gesture for undo"""
        if hand_landmarks:
            # Use wrist position for swipe detection
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * self.width)
            
            if self.swipe_detection['start_x'] is None:
                self.swipe_detection['start_x'] = wrist_x
                self.swipe_detection['frames'] = 0
            else:
                # Check if moved significantly left
                dx = wrist_x - self.swipe_detection['start_x']
                self.swipe_detection['frames'] += 1
                
                if dx < -100 and self.swipe_detection['frames'] < 30:  # Swipe left detected
                    self.undo_last_stroke()
                    self.swipe_detection['start_x'] = None
                    return True
                elif self.swipe_detection['frames'] > 30:  # Reset if too slow
                    self.swipe_detection['start_x'] = wrist_x
                    self.swipe_detection['frames'] = 0
        
        return False
    
    def undo_last_stroke(self):
        """Undo the last drawing stroke"""
        if self.canvas_history:
            self.drawing_canvas = self.canvas_history.pop().copy()
            print("Undo applied!")
    
    def save_canvas_state(self):
        """Save current canvas state for undo"""
        self.canvas_history.append(self.drawing_canvas.copy())
    
    def apply_kaleidoscope_effect(self, canvas):
        """Apply kaleidoscope effect to the canvas"""
        if not self.kaleidoscope_mode:
            return canvas
        
        center = (self.width // 2, self.height // 2)
        radius = min(self.width, self.height) // 3
        
        # Create kaleidoscope effect
        result = canvas.copy()
        
        for i in range(1, self.kaleidoscope_segments):
            angle = (360 / self.kaleidoscope_segments) * i
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated = cv2.warpAffine(canvas, M, (self.width, self.height))
            
            # Blend with result
            result = cv2.addWeighted(result, 0.8, rotated, 0.2, 0)
        
        return result
    
    def save_image(self):
        """Save current canvas as JPEG"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kinetic_art_{timestamp}.jpg"
        
        # Combine all canvases
        final_image = cv2.addWeighted(self.body_canvas, 0.3, self.drawing_canvas, 1.0, 0)
        final_image = self.apply_kaleidoscope_effect(final_image)
        
        cv2.imwrite(filename, final_image)
        print(f"Saved: {filename}")
    
    def start_video_recording(self):
        """Start video recording for timelapse"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kinetic_timelapse_{timestamp}.avi"
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 10.0, (self.width, self.height))
            self.recording = True
            print(f"Recording started: {filename}")
    
    def stop_video_recording(self):
        """Stop video recording"""
        if self.recording and self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            print("Recording stopped!")
    
    def clear_canvas(self):
        """Clear all canvases"""
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.body_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.canvas_history.clear()
        self.finger_trackers.clear()
    
    def process_hands(self, frame):
        """Process hand detection and multi-finger drawing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        
        current_time = time.time()
        current_extended_fingers = {}
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                # Check for swipe gesture (only on first hand)
                if hand_idx == 0 and self.detect_swipe_left(hand_landmarks):
                    continue
                
                # Get extended fingers for this hand
                extended_fingers = self.get_extended_fingers(hand_landmarks)
                
                # Add hand index to finger ID to distinguish between hands
                for finger_name, pos in extended_fingers.items():
                    finger_id = f"{hand_idx}_{finger_name}"
                    current_extended_fingers[finger_id] = {
                        'pos': pos,
                        'name': finger_name
                    }
        
        # Process each currently extended finger
        for finger_id, finger_data in current_extended_fingers.items():
            finger_pos = finger_data['pos']
            finger_name = finger_data['name']
            
            # Initialize tracker if new
            if finger_id not in self.finger_trackers:
                self.finger_trackers[finger_id] = {
                    'positions': deque(maxlen=20),
                    'velocities': deque(maxlen=10),
                    'start_time': current_time,
                    'drawing_mode': False,
                    'last_pos': None
                }
            
            tracker = self.finger_trackers[finger_id]
            tracker['positions'].append(finger_pos)
            
            time_held = current_time - tracker['start_time']
            
            # Activate drawing mode
            if time_held >= self.drawing_threshold and not tracker['drawing_mode']:
                tracker['drawing_mode'] = True
                tracker['last_pos'] = finger_pos
                self.save_canvas_state()
                print(f"Drawing mode activated for {finger_name}!")
            
            # Draw if in drawing mode
            if tracker['drawing_mode']:
                velocity = self.calculate_velocity(finger_id, finger_pos)
                
                if tracker['last_pos']:
                    color = self.get_dynamic_color(finger_name)
                    self.apply_brush(tracker['last_pos'], finger_pos, color, velocity)
                
                tracker['last_pos'] = finger_pos
                
                # Visual indicator for active drawing finger
                color = self.get_dynamic_color(finger_name)
                cv2.circle(frame, finger_pos, 12, color, 3)
                
                # Draw finger label
                cv2.putText(frame, finger_name[:1].upper(), 
                           (finger_pos[0] - 10, finger_pos[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                # Show activation progress
                progress = time_held / self.drawing_threshold
                circle_size = int(8 + 8 * progress)
                cv2.circle(frame, finger_pos, circle_size, (0, 255, 0), 2)
                
                # Draw finger label
                cv2.putText(frame, finger_name[:1].upper(), 
                           (finger_pos[0] - 10, finger_pos[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Clean up trackers for fingers that are no longer extended
        fingers_to_remove = []
        for finger_id, tracker in self.finger_trackers.items():
            if finger_id not in current_extended_fingers:
                if tracker['drawing_mode']:
                    print(f"Drawing mode deactivated for {finger_id.split('_')[1]}!")
                fingers_to_remove.append(finger_id)
        
        for finger_id in fingers_to_remove:
            del self.finger_trackers[finger_id]
        
        return frame
    
    def process_body(self, frame):
        """Process body silhouette"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask > 0.5
            silhouette = np.zeros_like(frame)
            silhouette[mask] = [50, 50, 50]  # Dim gray
            
            # Add to body canvas with fade
            self.body_canvas = cv2.addWeighted(self.body_canvas, 0.95, silhouette, 0.3, 0)
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                original_frame = frame.copy()
                
                # Apply background image if enabled (before processing hands)
                if self.background_enabled:
                    frame = self.apply_background(frame)
                
                # Process hands on the composited frame
                frame = self.process_hands(frame)
                self.process_body(original_frame)
                
                # Combine drawing layers with the background-composited frame
                if not self.background_enabled:
                    # If no background, use original body silhouette blending
                    display_frame = cv2.addWeighted(self.body_canvas, 0.4, frame, 0.6, 0)
                else:
                    # With background, just overlay the frame (background already applied)
                    display_frame = frame
                
                # Add drawing canvas
                display_frame = cv2.addWeighted(display_frame, 0.7, self.drawing_canvas, 1.0, 0)
                
                # Apply kaleidoscope if enabled
                if self.kaleidoscope_mode:
                    display_frame = self.apply_kaleidoscope_effect(display_frame)
                
                # Add UI elements
                self.draw_ui(display_frame)
                
                # Record frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(display_frame)
                
                cv2.imshow('Multi-Finger Kinetic Canvas with Background', display_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.clear_canvas()
                elif key == ord('k'):
                    self.kaleidoscope_mode = not self.kaleidoscope_mode
                    print(f"Kaleidoscope: {'ON' if self.kaleidoscope_mode else 'OFF'}")
                elif key == ord('b'):
                    self.current_brush = (self.current_brush + 1) % len(self.brush_types)
                    print(f"Brush: {self.brush_types[self.current_brush]}")
                elif key == ord('i'):
                    self.load_background_image()
                elif key == ord('t'):
                    if self.background_image is not None:
                        self.background_enabled = not self.background_enabled
                        print(f"Background: {'ON' if self.background_enabled else 'OFF'}")
                    else:
                        print("No background image loaded. Press 'i' to load one.")
                elif key == ord('=') or key == ord('+'):  # Increase opacity
                    self.adjust_background_opacity(0.1)
                elif key == ord('-') or key == ord('_'):  # Decrease opacity
                    self.adjust_background_opacity(-0.1)
                elif key == ord('s'):
                    self.save_image()
                elif key == ord('v'):
                    if self.recording:
                        self.stop_video_recording()
                    else:
                        self.start_video_recording()
        
        finally:
            self.cleanup()
    
    def draw_ui(self, frame):
        """Draw UI information"""
        y_offset = 30
        
        # Active fingers count
        active_count = sum(1 for tracker in self.finger_trackers.values() if tracker['drawing_mode'])
        cv2.putText(frame, f"Active Fingers: {active_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if active_count > 0 else (255, 255, 255), 2)
        
        # Audio info
        y_offset += 30
        cv2.putText(frame, f"Pitch: {self.current_pitch:.0f}Hz", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset += 25
        cv2.putText(frame, f"Volume: {self.current_volume:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Brush info
        y_offset += 25
        brush_name = self.brush_types[self.current_brush]
        cv2.putText(frame, f"Brush: {brush_name}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Background info
        y_offset += 25
        if self.background_image is not None:
            bg_status = f"BG: {'ON' if self.background_enabled else 'OFF'} ({self.background_opacity:.1f})"
            cv2.putText(frame, bg_status, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if self.background_enabled else (128, 128, 128), 1)
        else:
            cv2.putText(frame, "BG: None loaded", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (self.width - 50, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 80, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Kaleidoscope indicator
        if self.kaleidoscope_mode:
            cv2.putText(frame, "KALEIDOSCOPE", (self.width - 200, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Instructions
        instructions = [
            "Extend fingers to paint",
            "Hold 1.5s to activate",
            "Each finger = different color",
            "Press 'i' for background image"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = self.height - 100 + (i * 20)
            cv2.putText(frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def cleanup(self):
        """Clean up resources"""
        self.audio_running = False
        
        if self.video_writer:
            self.video_writer.release()
        
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Installing required packages if not present...")
    print("pip install opencv-python mediapipe pyaudio SpeechRecognition numpy tk")
    
    try:
        canvas = AdvancedKineticCanvas()
        canvas.run()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python mediapipe pyaudio SpeechRecognition numpy tk")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera and microphone are connected")