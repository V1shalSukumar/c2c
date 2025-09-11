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

class VirtualIcon:
    """Represents a virtual on-screen icon that can be activated by touch."""
    def __init__(self, x, y, width, height, icon_type, label=""):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.icon_type = icon_type
        self.label = label
        self.is_active = False
        self.touch_timer = 0
        self.activation_time = 0.5  # 500ms to activate
        
    def contains_point(self, point_x, point_y):
        """Check if a point is inside this icon"""
        if self.icon_type == "record":
            # For a circular icon, check distance from the center
            center_x = self.x + self.width // 2
            center_y = self.y + self.height // 2
            radius = self.width // 2 # Use half the icon width as radius
            distance = np.sqrt((point_x - center_x)**2 + (point_y - center_y)**2)
            return distance <= radius
        else:
            # For all other icons, check rectangular area
            return (self.x <= point_x <= self.x + self.width and 
                    self.y <= point_y <= self.y + self.height)
    
    def draw(self, frame, is_touching=False, touch_progress=0.0):
        """Draw the icon on the frame"""
        if is_touching:
            color = (0, 255, 255)  # Yellow when touched
            thickness = 3
        else:
            color = (255, 255, 255)  # White normally
            thickness = 2
        
        # Draw icon background with transparency effect
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     (50, 50, 50), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     color, thickness)
        
        center_x = self.x + self.width // 2
        center_y = self.y + self.height // 2
        
        if self.icon_type == "brush":
            cv2.circle(frame, (center_x, center_y - 10), 8, color, 2)
            cv2.line(frame, (center_x, center_y + 2), 
                    (center_x, center_y + 15), color, 3)
        
        elif self.icon_type == "background":
            cv2.rectangle(frame, (center_x - 12, center_y - 8), 
                         (center_x + 12, center_y + 8), color, 2)
            cv2.circle(frame, (center_x - 6, center_y - 3), 3, color, -1)
            cv2.line(frame, (center_x - 8, center_y + 5), 
                    (center_x + 8, center_y - 2), color, 2)
        
        elif self.icon_type == "opacity_up":
            cv2.line(frame, (center_x - 8, center_y), 
                    (center_x + 8, center_y), color, 3)
            cv2.line(frame, (center_x, center_y - 8), 
                    (center_x, center_y + 8), color, 3)
        
        elif self.icon_type == "opacity_down":
            cv2.line(frame, (center_x - 8, center_y), 
                    (center_x + 8, center_y), color, 3)
        
        elif self.icon_type == "kaleidoscope":
            points = []
            for i in range(8):
                angle = i * math.pi / 4
                x = int(center_x + 10 * math.cos(angle))
                y = int(center_y + 10 * math.sin(angle))
                points.append((x, y))
            
            for i in range(0, len(points), 2):
                cv2.line(frame, (center_x, center_y), points[i], color, 2)
        
        elif self.icon_type == "clear":
            cv2.rectangle(frame, (center_x - 8, center_y - 6), 
                         (center_x + 8, center_y + 8), color, 2)
            cv2.line(frame, (center_x - 6, center_y - 10), 
                    (center_x + 6, center_y - 10), color, 2)
            cv2.line(frame, (center_x - 2, center_y - 2), 
                    (center_x - 2, center_y + 4), color, 1)
            cv2.line(frame, (center_x + 2, center_y - 2), 
                    (center_x + 2, center_y + 4), color, 1)
        
        elif self.icon_type == "save":
            cv2.rectangle(frame, (center_x - 10, center_y - 8), 
                         (center_x + 10, center_y + 8), color, 2)
            cv2.rectangle(frame, (center_x - 8, center_y - 8), 
                         (center_x + 4, center_y - 4), color, 1)
            cv2.rectangle(frame, (center_x - 6, center_y - 2), 
                         (center_x + 6, center_y + 6), color, -1)
        
        elif self.icon_type == "record":
            if is_touching:
                cv2.circle(frame, (center_x, center_y), 10, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (center_x, center_y), 10, color, 2)
        
        if is_touching and touch_progress > 0:
            progress_width = int(self.width * touch_progress)
            cv2.rectangle(frame, (self.x, self.y + self.height + 2), 
                         (self.x + progress_width, self.y + self.height + 6), 
                         (0, 255, 0), -1)
        
        if self.label:
            text_size = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            text_x = self.x + (self.width - text_size[0]) // 2
            text_y = self.y + self.height + 15 # Adjusted text position for clarity
            cv2.putText(frame, self.label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

class AdvancedKineticCanvas:
    def __init__(self):
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
        
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.background_image = None
        self.original_background = None
        self.background_enabled = False
        self.background_opacity = 0.8
        
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.body_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        self.finger_trackers = {}
        self.drawing_threshold = 1.0  # Reduced activation time for better responsiveness
        
        self.finger_tips = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        self.finger_mcps = {
            'thumb': 3,
            'index': 5,
            'middle': 9,
            'ring': 13,
            'pinky': 17
        }
        
        self.icons = []
        self.setup_virtual_icons()
        self.icon_touch_tracker = {}
        
        self.canvas_history = deque(maxlen=20)
        self.swipe_detection = {'left': False, 'start_x': None, 'frames': 0}
        
        self.audio_thread = None
        self.current_pitch = 0
        self.current_volume = 0
        self.audio_running = True
        
        self.voice_thread = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        self.brush_types = ['neon', 'fire', 'smoke', 'watercolor', 'pixelated']
        self.current_brush = 0
        self.brush_size_base = 5
        
        self.kaleidoscope_mode = False
        self.kaleidoscope_segments = 8
        
        self.video_writer = None
        self.recording = False
        self.recording_frames = []
        
        self.hue = 0.0
        
        print("Advanced Multi-Finger Kinetic Canvas with Virtual Touch Icons Initialized!")
        print("Features:")
        print("- Paint with multiple open fingers simultaneously")
        print("- Touch virtual icons to control features")
        print("- Voice commands: 'Clear', 'Save', 'Kaleidoscope'")
        print("- Swipe left to undo")
        print("- Press 'q' to quit")
        
        self.init_audio()
        self.init_voice_recognition()
    
    def setup_virtual_icons(self):
        """Setup virtual touchable icons"""
        icon_size = 50
        margin = 10
        text_spacing = 25 # Increased vertical spacing for icon labels
        start_x = self.width - icon_size - margin
        
        icons_config = [
            ("brush", "Brush", 0),
            ("background", "BG Image", 1),
            ("opacity_up", "Opacity +", 2),
            ("opacity_down", "Opacity -", 3),
            ("kaleidoscope", "Kaleid", 4),
            ("clear", "Clear", 5),
            ("save", "Save", 6),
            ("record", "Record", 7)
        ]
        
        for icon_type, label, index in icons_config:
            y_pos = margin + index * (icon_size + text_spacing)
            icon = VirtualIcon(start_x, y_pos, icon_size, icon_size, icon_type, label)
            self.icons.append(icon)
    
    def check_icon_touches(self, extended_fingers_all_hands):
        """Check if any extended fingers are touching icons"""
        current_time = time.time()
        
        all_finger_positions = [finger['pos'] for finger in extended_fingers_all_hands.values()]
        
        for icon in self.icons:
            is_being_touched = False
            for finger_pos in all_finger_positions:
                if icon.contains_point(finger_pos[0], finger_pos[1]):
                    is_being_touched = True
                    break
            
            icon_id = f"{icon.icon_type}_{icon.x}_{icon.y}"
            
            if is_being_touched:
                if icon_id not in self.icon_touch_tracker:
                    self.icon_touch_tracker[icon_id] = current_time
                
                touch_duration = current_time - self.icon_touch_tracker[icon_id]
                
                if touch_duration >= icon.activation_time:
                    if not icon.is_active:
                        self.activate_icon(icon)
                        icon.is_active = True
            else:
                if icon_id in self.icon_touch_tracker:
                    del self.icon_touch_tracker[icon_id]
                icon.is_active = False
    
    def activate_icon(self, icon):
        """Activate an icon's function"""
        print(f"Icon activated: {icon.icon_type}")
        
        if icon.icon_type == "brush":
            self.current_brush = (self.current_brush + 1) % len(self.brush_types)
            print(f"Brush changed to: {self.brush_types[self.current_brush]}")
        
        elif icon.icon_type == "background":
            self.load_background_image()
        
        elif icon.icon_type == "opacity_up":
            self.adjust_background_opacity(0.1)
        
        elif icon.icon_type == "opacity_down":
            self.adjust_background_opacity(-0.1)
        
        elif icon.icon_type == "kaleidoscope":
            self.kaleidoscope_mode = not self.kaleidoscope_mode
            print(f"Kaleidoscope: {'ON' if self.kaleidoscope_mode else 'OFF'}")
        
        elif icon.icon_type == "clear":
            self.clear_canvas()
            print("Canvas cleared!")
        
        elif icon.icon_type == "save":
            self.save_image()
        
        elif icon.icon_type == "record":
            if self.recording:
                self.stop_video_recording()
            else:
                self.start_video_recording()
    
    def draw_icons(self, frame):
        """Draw all virtual icons with touch feedback"""
        current_time = time.time()
        
        for icon in self.icons:
            icon_id = f"{icon.icon_type}_{icon.x}_{icon.y}"
            is_touching = icon_id in self.icon_touch_tracker
            
            touch_progress = 0.0
            if is_touching:
                touch_duration = current_time - self.icon_touch_tracker[icon_id]
                touch_progress = min(1.0, touch_duration / icon.activation_time)
            
            icon.draw(frame, is_touching, touch_progress)
    
    def load_background_image(self):
        """Load background image using file dialog"""
        try:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select Background Image",
                filetypes=[
                    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            if file_path:
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
        
        background_weighted = cv2.addWeighted(
            self.background_image, self.background_opacity,
            np.zeros_like(self.background_image), 1 - self.background_opacity, 0
        )
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = self.pose.process(rgb_frame)
        
        if pose_results.segmentation_mask is not None:
            mask = pose_results.segmentation_mask > 0.5
            mask_3d = np.stack([mask] * 3, axis=-1)
            result = np.where(mask_3d, frame, background_weighted)
            return result.astype(np.uint8)
        else:
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
                    self.current_volume = np.sqrt(np.mean(data**2)) * 100
                    pitch = self.detect_pitch(data, 44100)
                    if pitch > 0:
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
        windowed = audio_data * np.hanning(len(audio_data))
        autocorr = np.correlate(windowed, windowed, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        min_period = sample_rate // 800
        max_period = sample_rate // 80
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
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=2)
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    if "clear" in command:
                        self.clear_canvas()
                        print("Voice command: Clear")
                    elif "kaleidoscope" in command:
                        self.kaleidoscope_mode = not self.kaleidoscope_mode
                        print(f"Voice command: Kaleidoscope {'ON' if self.kaleidscope_mode else 'OFF'}")
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
        base_hue = self.hue
        finger_offsets = {'thumb': 0.0, 'index': 0.1, 'middle': 0.2, 'ring': 0.3, 'pinky': 0.4}
        finger_hue = (base_hue + finger_offsets.get(finger_name, 0.0)) % 1.0
        # Increased saturation to 1.0 for more vibrant colors
        rgb = colorsys.hsv_to_rgb(finger_hue, 1.0, 1.0) 
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
    
    def get_dynamic_brush_size(self):
        """Get brush size based on volume"""
        volume_factor = min(self.current_volume / 20.0, 3.0)
        return int(self.brush_size_base * (0.5 + volume_factor))
    
    def is_finger_extended(self, hand_landmarks, finger_name):
        """Check if a finger is extended (open)"""
        if not hand_landmarks:
            return False
        tip_idx = self.finger_tips[finger_name]
        if finger_name == 'thumb':
            wrist = hand_landmarks.landmark[0]
            tip = hand_landmarks.landmark[tip_idx]
            mcp = hand_landmarks.landmark[self.finger_mcps[finger_name]]
            wrist_to_tip = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
            wrist_to_mcp = np.sqrt((mcp.x - wrist.x)**2 + (mcp.y - wrist.y)**2)
            return wrist_to_tip > wrist_to_mcp * 1.2
        else:
            tip = hand_landmarks.landmark[tip_idx]
            mcp = hand_landmarks.landmark[self.finger_mcps[finger_name]]
            return tip.y < mcp.y - 0.02
    
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
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
        glow_color = tuple(int(c * 0.6) for c in color)
        cv2.line(self.drawing_canvas, start_pos, end_pos, glow_color, size + 4, cv2.LINE_AA)
        glow_color2 = tuple(int(c * 0.3) for c in color)
        cv2.line(self.drawing_canvas, start_pos, end_pos, glow_color2, size + 8, cv2.LINE_AA)
    
    def draw_fire_brush(self, start_pos, end_pos, color, size, velocity):
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
        num_particles = int(velocity / 2) + 1
        for _ in range(num_particles):
            offset_x = random.randint(-size, size)
            offset_y = random.randint(-size, size)
            particle_pos = (end_pos[0] + offset_x, end_pos[1] + offset_y)
            fire_colors = [(0, 165, 255), (0, 100, 255), (0, 255, 255)]
            fire_color = random.choice(fire_colors)
            cv2.circle(self.drawing_canvas, particle_pos, random.randint(1, 3), fire_color, -1)
    
    def draw_smoke_brush(self, start_pos, end_pos, color, size, velocity):
        smoke_color = (128, 128, 128)
        overlay = self.drawing_canvas.copy()
        cv2.line(overlay, start_pos, end_pos, smoke_color, size + 2, cv2.LINE_AA)
        for i in range(3):
            offset = random.randint(-size//2, size//2)
            wispy_start = (start_pos[0] + offset, start_pos[1] + offset)
            wispy_end = (end_pos[0] + offset, end_pos[1] + offset)
            cv2.line(overlay, wispy_start, wispy_end, smoke_color, 1, cv2.LINE_AA)
        cv2.addWeighted(self.drawing_canvas, 0.7, overlay, 0.3, 0, self.drawing_canvas)
    
    def draw_watercolor_brush(self, start_pos, end_pos, color, size, velocity):
        if velocity > 15:
            num_drops = int(velocity / 5)
            for _ in range(num_drops):
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(size, size * 3)
                drop_x = int(end_pos[0] + distance * math.cos(angle))
                drop_y = int(end_pos[1] + distance * math.sin(angle))
                drop_size = random.randint(1, size//2)
                varied_color = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in color)
                cv2.circle(self.drawing_canvas, (drop_x, drop_y), drop_size, varied_color, -1)
        else:
            overlay = self.drawing_canvas.copy()
            cv2.line(overlay, start_pos, end_pos, color, size, cv2.LINE_AA)
            cv2.addWeighted(self.drawing_canvas, 0.8, overlay, 0.2, 0, self.drawing_canvas)
    
    def draw_pixelated_brush(self, start_pos, end_pos, color, size, velocity):
        pixel_size = max(2, size // 2)
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = int(np.sqrt(dx*dx + dy*dy))
        if distance > 0:
            for i in range(0, distance, pixel_size):
                t = i / distance
                x = int(start_pos[0] + t * dx)
                y = int(start_pos[1] + t * dy)
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
        result = canvas.copy()
        for i in range(1, self.kaleidoscope_segments):
            angle = (360 / self.kaleidoscope_segments) * i
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated = cv2.warpAffine(canvas, M, (self.width, self.height))
            result = cv2.addWeighted(result, 0.8, rotated, 0.2, 0)
        return result
    
    def draw_kaleidoscope_brush(self, start_pos, end_pos, color, velocity):
        """Draw a brush stroke and its kaleidoscope reflections"""
        center_x, center_y = self.width // 2, self.height // 2
        vec_x_start, vec_y_start = start_pos[0] - center_x, start_pos[1] - center_y
        vec_x_end, vec_y_end = end_pos[0] - center_x, end_pos[1] - center_y
        
        for i in range(self.kaleidoscope_segments):
            angle = (360 / self.kaleidoscope_segments) * i
            rad_angle = math.radians(angle)
            cos_a, sin_a = math.cos(rad_angle), math.sin(rad_angle)
            rot_x_start = vec_x_start * cos_a - vec_y_start * sin_a
            rot_y_start = vec_x_start * sin_a + vec_y_start * cos_a
            rotated_start_pos = (int(rot_x_start + center_x), int(rot_y_start + center_y))
            rot_x_end = vec_x_end * cos_a - vec_y_end * sin_a
            rot_y_end = vec_x_end * sin_a + vec_y_end * cos_a
            rotated_end_pos = (int(rot_x_end + center_x), int(rot_y_end + center_y))
            self.apply_brush(rotated_start_pos, rotated_end_pos, color, velocity)
            mirror_start_pos = (2 * center_x - rotated_start_pos[0], rotated_start_pos[1])
            mirror_end_pos = (2 * center_x - rotated_end_pos[0], rotated_end_pos[1])
            self.apply_brush(mirror_start_pos, mirror_end_pos, color, velocity)
    
    def save_image(self):
        """Save current canvas as JPEG"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"kinetic_art_{timestamp}.jpg"
        final_image = self.drawing_canvas.copy()
        if self.background_enabled and self.background_image is not None:
            combined = cv2.addWeighted(self.background_image, 1 - 0.2, final_image, 0.2, 0)
            final_image = combined
        cv2.imwrite(filename, final_image)
        print(f"Saved: {filename}")
    
    def start_video_recording(self):
        """Start video recording for timelapse"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kinetic_timelapse_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (self.width, self.height))
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
        self.canvas_history.clear()
        self.finger_trackers.clear()
    
    def process_hands(self, frame, hand_results):
        """Process hand detection and multi-finger drawing"""
        current_time = time.time()
        current_extended_fingers = {}
        
        if hand_results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                extended_fingers = self.get_extended_fingers(hand_landmarks)
                for finger_name, pos in extended_fingers.items():
                    finger_id = f"{hand_idx}_{finger_name}"
                    current_extended_fingers[finger_id] = {'pos': pos, 'name': finger_name}
                if hand_idx == 0:
                    self.detect_swipe_left(hand_landmarks)

        for finger_id, finger_data in current_extended_fingers.items():
            finger_pos = finger_data['pos']
            finger_name = finger_data['name']
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
            if time_held >= self.drawing_threshold and not tracker['drawing_mode']:
                tracker['drawing_mode'] = True
                tracker['last_pos'] = finger_pos
                self.save_canvas_state()
                print(f"Drawing mode activated for {finger_name}!")
            if tracker['drawing_mode']:
                velocity = self.calculate_velocity(finger_id, finger_pos)
                if tracker['last_pos']:
                    color = self.get_dynamic_color(finger_name)
                    if self.kaleidoscope_mode:
                        self.draw_kaleidoscope_brush(tracker['last_pos'], finger_pos, color, velocity)
                    else:
                        self.apply_brush(tracker['last_pos'], finger_pos, color, velocity)
                tracker['last_pos'] = finger_pos
                color = self.get_dynamic_color(finger_name)
                cv2.circle(frame, finger_pos, 12, color, 3)
                cv2.putText(frame, finger_name[:1].upper(), 
                           (finger_pos[0] - 10, finger_pos[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                progress = time_held / self.drawing_threshold
                circle_size = int(8 + 8 * progress)
                cv2.circle(frame, finger_pos, circle_size, (0, 255, 0), 2)
                cv2.putText(frame, finger_name[:1].upper(), 
                           (finger_pos[0] - 10, finger_pos[1] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        fingers_to_remove = []
        for finger_id, tracker in self.finger_trackers.items():
            if finger_id not in current_extended_fingers:
                if tracker['drawing_mode']:
                    print(f"Drawing mode deactivated for {finger_id.split('_')[1]}!")
                fingers_to_remove.append(finger_id)
        
        for finger_id in fingers_to_remove:
            del self.finger_trackers[finger_id]
        
        return current_extended_fingers
    
    def detect_swipe_left(self, hand_landmarks):
        """Detect left swipe gesture for undo"""
        if hand_landmarks:
            wrist = hand_landmarks.landmark[0]
            wrist_x = int(wrist.x * self.width)
            if self.swipe_detection['start_x'] is None:
                self.swipe_detection['start_x'] = wrist_x
                self.swipe_detection['frames'] = 0
            else:
                dx = wrist_x - self.swipe_detection['start_x']
                self.swipe_detection['frames'] += 1
                if dx < -100 and self.swipe_detection['frames'] < 30:
                    self.undo_last_stroke()
                    self.swipe_detection['start_x'] = None
                    return True
                elif self.swipe_detection['frames'] > 30:
                    self.swipe_detection['start_x'] = wrist_x
                    self.swipe_detection['frames'] = 0
        return False
    
    def draw_ui(self, frame):
        """Draw UI information"""
        y_offset = 30
        active_count = sum(1 for tracker in self.finger_trackers.values() if tracker['drawing_mode'])
        cv2.putText(frame, f"Active Fingers: {active_count}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if active_count > 0 else (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Pitch: {self.current_pitch:.0f}Hz", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        cv2.putText(frame, f"Volume: {self.current_volume:.1f}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        brush_name = self.brush_types[self.current_brush]
        cv2.putText(frame, f"Brush: {brush_name}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
        if self.background_image is not None:
            bg_status = f"BG: {'ON' if self.background_enabled else 'OFF'} ({self.background_opacity:.1f})"
            cv2.putText(frame, bg_status, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if self.background_enabled else (128, 128, 128), 1)
        else:
            cv2.putText(frame, "BG: None loaded", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        if self.recording:
            cv2.circle(frame, (self.width - 50, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 80, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if self.kaleidoscope_mode:
            cv2.putText(frame, "KALEIDOSCOPE", (self.width - 200, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    def cleanup(self):
        self.audio_running = False
        if self.video_writer:
            self.video_writer.release()
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        self.cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                processed_frame = frame.copy()
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                hand_results = self.hands.process(rgb_frame)
                
                extended_fingers_all_hands = self.process_hands(processed_frame, hand_results)
                
                final_display_frame = self.drawing_canvas.copy()
                if self.background_enabled and self.background_image is not None:
                    final_display_frame = cv2.addWeighted(final_display_frame, 1, self.background_image, 1, 0)
                
                final_display_frame = self.apply_kaleidoscope_effect(final_display_frame)
                
                blended_frame = cv2.addWeighted(processed_frame, 0.7, final_display_frame, 0.3, 0)

                self.check_icon_touches(extended_fingers_all_hands)
                
                self.draw_icons(blended_frame)
                
                self.draw_ui(blended_frame)
                
                if self.recording and self.video_writer:
                    self.video_writer.write(blended_frame)
                
                cv2.imshow('Multi-Finger Kinetic Canvas with Background', blended_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            self.cleanup()

if __name__ == "__main__":
    try:
        canvas = AdvancedKineticCanvas()
        canvas.run()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python mediapipe pyaudio SpeechRecognition numpy tk")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera and microphone are connected")
