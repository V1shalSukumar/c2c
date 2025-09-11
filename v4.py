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

class MultiFingerKineticCanvas:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Enable 2 hands
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
        
        # Canvas layers
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.body_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Multi-finger drawing state
        self.finger_drawing_states = {}  # Track each finger's drawing state
        self.finger_stop_timers = {}     # Track stopping times for each finger
        self.finger_positions_history = {}  # Position history for each finger
        self.stop_threshold = 5.0        # 5 seconds to stop drawing
        self.stop_distance_threshold = 30  # pixels
        
        # Gesture detection
        self.snap_detection = {'last_snap': 0, 'snap_threshold': 1.0}
        self.five_finger_swipe = {'detecting': False, 'start_positions': [], 'start_time': 0}
        self.thumbs_up_detection = {'left': False, 'right': False, 'last_detection': 0}
        
        # Hand tracking
        self.current_hands = {}  # Track current hand positions and states
        
        # Canvas history for undo
        self.canvas_history = deque(maxlen=20)
        
        # Audio and voice (from previous implementation)
        self.audio_thread = None
        self.current_pitch = 0
        self.current_volume = 0
        self.audio_running = True
        
        self.voice_thread = None
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Brush system
        self.brush_types = ['neon', 'fire', 'smoke', 'watercolor', 'pixelated']
        self.current_brush = 0
        self.brush_size_base = 5
        
        # Kaleidoscope mode
        self.kaleidoscope_mode = False
        
        # Video recording
        self.video_writer = None
        self.recording = False
        
        # Color system
        self.hue = 0.0
        
        # Finger colors for multi-finger drawing
        self.finger_colors = [
            (255, 100, 100),  # Thumb - Light Blue
            (100, 255, 100),  # Index - Light Green  
            (100, 100, 255),  # Middle - Light Red
            (255, 255, 100),  # Ring - Light Cyan
            (255, 100, 255)   # Pinky - Light Magenta
        ]
        
        print("Multi-Finger Kinetic Canvas Initialized!")
        print("Gesture Controls:")
        print("- All 5 fingers: Individual drawing with different colors")
        print("- 5-finger swipe: Clear entire screen")
        print("- Snap fingers: Take screenshot")
        print("- Thumbs up (both hands): Mark completion")
        print("- Stop at same position 5 seconds: Stop drawing that finger")
        print("- Voice commands: 'Clear', 'Save', 'Kaleidoscope'")
        
        self.init_audio()
        self.init_voice_recognition()
    
    def init_audio(self):
        """Initialize audio processing"""
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
                except Exception:
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
            print(f"Voice recognition failed: {e}")
    
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
                        print("Voice: Clear")
                    elif "kaleidoscope" in command:
                        self.kaleidoscope_mode = not self.kaleidoscope_mode
                        print(f"Voice: Kaleidoscope {'ON' if self.kaleidoscope_mode else 'OFF'}")
                    elif "save" in command:
                        self.save_image()
                        print("Voice: Save")
                except (sr.UnknownValueError, sr.RequestError):
                    pass
                    
            except sr.WaitTimeoutError:
                pass
            except Exception:
                continue
    
    def get_finger_positions(self, hand_landmarks):
        """Get all 5 finger tip positions"""
        if not hand_landmarks:
            return []
        
        # MediaPipe finger tip landmark indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        positions = []
        
        for tip_id in finger_tips:
            landmark = hand_landmarks.landmark[tip_id]
            x = int(landmark.x * self.width)
            y = int(landmark.y * self.height)
            positions.append((x, y))
        
        return positions
    
    def detect_finger_snap(self, hand_landmarks):
        """Detect finger snap gesture"""
        if not hand_landmarks:
            return False
        
        # Get thumb tip and middle finger tip
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        # Calculate distance
        thumb_pos = (thumb_tip.x * self.width, thumb_tip.y * self.height)
        middle_pos = (middle_tip.x * self.width, middle_tip.y * self.height)
        
        distance = np.sqrt((thumb_pos[0] - middle_pos[0])**2 + (thumb_pos[1] - middle_pos[1])**2)
        
        current_time = time.time()
        
        # If fingers are very close and enough time has passed since last snap
        if distance < 30 and (current_time - self.snap_detection['last_snap']) > self.snap_detection['snap_threshold']:
            self.snap_detection['last_snap'] = current_time
            return True
        
        return False
    
    def detect_five_finger_swipe(self, all_finger_positions):
        """Detect 5-finger swipe gesture for screen clear"""
        current_time = time.time()
        
        if len(all_finger_positions) >= 5:
            if not self.five_finger_swipe['detecting']:
                # Start detecting swipe
                self.five_finger_swipe['detecting'] = True
                self.five_finger_swipe['start_positions'] = all_finger_positions[:5]
                self.five_finger_swipe['start_time'] = current_time
            else:
                # Check if all fingers moved significantly in same direction
                if (current_time - self.five_finger_swipe['start_time']) < 2.0:  # Within 2 seconds
                    movements = []
                    for i, (start_pos, current_pos) in enumerate(zip(self.five_finger_swipe['start_positions'], all_finger_positions[:5])):
                        dx = current_pos[0] - start_pos[0]
                        dy = current_pos[1] - start_pos[1]
                        movements.append((dx, dy))
                    
                    # Check if all fingers moved in similar direction (horizontal swipe)
                    avg_dx = np.mean([mov[0] for mov in movements])
                    avg_dy = np.mean([mov[1] for mov in movements])
                    
                    if abs(avg_dx) > 200 and abs(avg_dy) < 100:  # Horizontal swipe detected
                        self.five_finger_swipe['detecting'] = False
                        return True
                else:
                    # Reset if too slow
                    self.five_finger_swipe['detecting'] = False
        else:
            # Reset if not enough fingers
            self.five_finger_swipe['detecting'] = False
        
        return False
    
    def detect_thumbs_up(self, hand_landmarks, hand_label):
        """Detect thumbs up gesture"""
        if not hand_landmarks:
            return False
        
        # Get key landmarks
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        # Check if thumb is extended upward
        thumb_up = thumb_tip.y < thumb_mcp.y - 0.05
        
        # Check if other fingers are folded (tips below knuckles)
        index_folded = index_tip.y > hand_landmarks.landmark[6].y
        middle_folded = middle_tip.y > hand_landmarks.landmark[10].y
        ring_folded = ring_tip.y > hand_landmarks.landmark[14].y
        pinky_folded = pinky_tip.y > hand_landmarks.landmark[18].y
        
        is_thumbs_up = thumb_up and index_folded and middle_folded and ring_folded and pinky_folded
        
        # Update detection state
        self.thumbs_up_detection[hand_label] = is_thumbs_up
        
        return is_thumbs_up
    
    def check_finger_stopped(self, finger_id, position):
        """Check if finger has stopped at same position for 5 seconds"""
        current_time = time.time()
        
        # Initialize tracking for this finger if needed
        if finger_id not in self.finger_stop_timers:
            self.finger_stop_timers[finger_id] = {'start_time': current_time, 'position': position}
            self.finger_positions_history[finger_id] = deque(maxlen=30)  # 30 frames of history
            return False
        
        # Add current position to history
        self.finger_positions_history[finger_id].append(position)
        
        # Check if finger has been in roughly same position
        if len(self.finger_positions_history[finger_id]) >= 10:
            recent_positions = list(self.finger_positions_history[finger_id])[-10:]
            
            # Calculate variance in positions
            x_positions = [pos[0] for pos in recent_positions]
            y_positions = [pos[1] for pos in recent_positions]
            
            x_var = np.var(x_positions)
            y_var = np.var(y_positions)
            
            # If finger is relatively still
            if x_var < self.stop_distance_threshold and y_var < self.stop_distance_threshold:
                # Check if enough time has passed
                if (current_time - self.finger_stop_timers[finger_id]['start_time']) >= self.stop_threshold:
                    return True
            else:
                # Reset timer if finger moved significantly
                self.finger_stop_timers[finger_id] = {'start_time': current_time, 'position': position}
        
        return False
    
    def get_dynamic_color(self, finger_index=0):
        """Get color based on pitch and finger"""
        # Base color from audio
        rgb = colorsys.hsv_to_rgb(self.hue, 0.9, 1.0)
        audio_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        
        # Blend with finger-specific color
        finger_color = self.finger_colors[finger_index % len(self.finger_colors)]
        
        # Weighted blend: 70% audio, 30% finger-specific
        blended_color = tuple(
            int(0.7 * audio_color[i] + 0.3 * finger_color[i])
            for i in range(3)
        )
        
        return blended_color
    
    def get_dynamic_brush_size(self):
        """Get brush size based on volume"""
        volume_factor = min(self.current_volume / 20.0, 3.0)
        return int(self.brush_size_base * (0.5 + volume_factor))
    
    def apply_brush_effect(self, start_pos, end_pos, color, size, velocity, finger_index):
        """Apply brush effect with finger-specific variations"""
        brush_name = self.brush_types[self.current_brush]
        
        # Modify effect based on finger
        if finger_index == 0:  # Thumb - thicker strokes
            size = int(size * 1.5)
        elif finger_index == 1:  # Index - normal
            pass
        elif finger_index == 2:  # Middle - sharper effects
            if brush_name == 'fire':
                velocity *= 1.5
        elif finger_index == 3:  # Ring - softer effects
            if brush_name == 'watercolor':
                velocity *= 0.7
        elif finger_index == 4:  # Pinky - finer details
            size = max(1, int(size * 0.7))
        
        # Apply the brush effect
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
    
    def draw_neon_brush(self, start_pos, end_pos, color, size, velocity):
        """Neon effect brush"""
        cv2.line(self.drawing_canvas, start_pos, end_pos, color, size, cv2.LINE_AA)
        glow_color = tuple(int(c * 0.6) for c in color)
        cv2.line(self.drawing_canvas, start_pos, end_pos, glow_color, size + 4, cv2.LINE_AA)
        glow_color2 = tuple(int(c * 0.3) for c in color)
        cv2.line(self.drawing_canvas, start_pos, end_pos, glow_color2, size + 8, cv2.LINE_AA)
    
    def draw_fire_brush(self, start_pos, end_pos, color, size, velocity):
        """Fire effect brush"""
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
        """Smoke effect brush"""
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
        """Watercolor effect brush"""
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
        """Pixelated brush"""
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
    
    def calculate_velocity(self, current_pos, finger_id):
        """Calculate finger movement velocity"""
        if finger_id not in self.finger_positions_history:
            self.finger_positions_history[finger_id] = deque(maxlen=5)
        
        history = self.finger_positions_history[finger_id]
        if len(history) > 0:
            prev_pos = history[-1]
            distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
            return distance
        
        return 0
    
    def save_canvas_state(self):
        """Save canvas state for undo"""
        self.canvas_history.append(self.drawing_canvas.copy())
    
    def clear_canvas(self):
        """Clear all drawing"""
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.body_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.canvas_history.clear()
        self.finger_drawing_states.clear()
        self.finger_stop_timers.clear()
        self.finger_positions_history.clear()
        print("Canvas cleared!")
    
    def save_image(self):
        """Save current canvas as JPEG"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"multifinger_art_{timestamp}.jpg"
        
        final_image = cv2.addWeighted(self.body_canvas, 0.3, self.drawing_canvas, 1.0, 0)
        if self.kaleidoscope_mode:
            final_image = self.apply_kaleidoscope_effect(final_image)
        
        cv2.imwrite(filename, final_image)
        print(f"Saved: {filename}")
    
    def apply_kaleidoscope_effect(self, canvas):
        """Apply kaleidoscope effect"""
        if not self.kaleidoscope_mode:
            return canvas
        
        center = (self.width // 2, self.height // 2)
        result = canvas.copy()
        
        for i in range(1, 8):
            angle = 45 * i
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated = cv2.warpAffine(canvas, M, (self.width, self.height))
            result = cv2.addWeighted(result, 0.8, rotated, 0.2, 0)
        
        return result
    
    def process_hands(self, frame):
        """Process hand detection and multi-finger drawing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        
        all_finger_positions = []
        snap_detected = False
        current_time = time.time()
        
        if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
            for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()  # 'left' or 'right'
                
                # Get all finger positions for this hand
                finger_positions = self.get_finger_positions(hand_landmarks)
                all_finger_positions.extend(finger_positions)
                
                # Check for snap gesture
                if self.detect_finger_snap(hand_landmarks):
                    snap_detected = True
                
                # Check for thumbs up
                if self.detect_thumbs_up(hand_landmarks, hand_label):
                    cv2.putText(frame, f"THUMBS UP - {hand_label.upper()}", 
                               (10, 100 if hand_label == 'left' else 130), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Process each finger for drawing
                for finger_idx, finger_pos in enumerate(finger_positions):
                    finger_id = f"{hand_label}_{finger_idx}"
                    
                    # Check if finger stopped at position
                    if self.check_finger_stopped(finger_id, finger_pos):
                        if finger_id in self.finger_drawing_states:
                            del self.finger_drawing_states[finger_id]
                            print(f"Stopped drawing: {hand_label} {['thumb','index','middle','ring','pinky'][finger_idx]}")
                    else:
                        # Continue or start drawing
                        if finger_id in self.finger_drawing_states:
                            # Continue drawing
                            last_pos = self.finger_drawing_states[finger_id]['last_pos']
                            velocity = self.calculate_velocity(finger_pos, finger_id)
                            
                            color = self.get_dynamic_color(finger_idx)
                            size = self.get_dynamic_brush_size()
                            
                            self.apply_brush_effect(last_pos, finger_pos, color, size, velocity, finger_idx)
                            
                            self.finger_drawing_states[finger_id]['last_pos'] = finger_pos
                        else:
                            # Start drawing
                            self.finger_drawing_states[finger_id] = {
                                'last_pos': finger_pos,
                                'start_time': current_time
                            }
                            self.save_canvas_state()
                    
                    # Visual feedback
                    color = self.get_dynamic_color(finger_idx)
                    if finger_id in self.finger_drawing_states:
                        cv2.circle(frame, finger_pos, 8, color, -1)  # Solid circle when drawing
                    else:
                        cv2.circle(frame, finger_pos, 8, color, 2)   # Hollow circle when not drawing
                    
                    # Show finger label
                    finger_names = ['T', 'I', 'M', 'R', 'P']
                    cv2.putText(frame, finger_names[finger_idx], 
                               (finger_pos[0] - 5, finger_pos[1] - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Handle gestures
        if snap_detected:
            self.save_image()
            print("Snap detected - Screenshot saved!")
        
        if self.detect_five_finger_swipe(all_finger_positions):
            self.clear_canvas()
            print("5-finger swipe detected - Canvas cleared!")
        
        # Check for both thumbs up (completion gesture)
        if (self.thumbs_up_detection['left'] and self.thumbs_up_detection['right'] and 
            (current_time - self.thumbs_up_detection.get('last_detection', 0)) > 2.0):
            self.thumbs_up_detection['last_detection'] = current_time
            self.save_image()
            print("Both thumbs up - Artwork completed and saved!")
        
        return frame
    
    def process_body(self, frame):
        """Process body silhouette"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.segmentation_mask is not None:
            mask = results.segmentation_mask > 0.5
            silhouette = np.zeros_like(frame)
            silhouette[mask] = [30, 30, 30]  # Very dim gray
            
            self.body_canvas = cv2.addWeighted(self.body_canvas, 0.98, silhouette, 0.2, 0)
    
    def draw_ui(self, frame):
        """Draw UI information"""
        y_offset = 30
        
        # Active fingers count
        active_fingers = len(self.finger_drawing_states)
        cv2.putText(frame, f"Active Fingers: {active_fingers}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Audio info
        y_offset += 30
        cv2.putText(frame, f"Pitch: {self.current_pitch:.0f}Hz Vol: {self.current_volume:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Brush info
        y_offset += 25
        cv2.putText(frame, f"Brush: {self.brush_types[self.current_brush]}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Gesture instructions
        y_offset += 25
        cv2.putText(frame, "5-finger swipe: Clear | Snap: Save | Both thumbs up: Complete", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (self.width - 50, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 80, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Kaleidoscope indicator
        if self.kaleidoscope_mode:
            cv2.putText(frame, "KALEIDOSCOPE", (self.width - 200, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Show finger stop timers
        y_offset = self.height - 100
        for finger_id, timer_info in self.finger_stop_timers.items():
            elapsed = time.time() - timer_info['start_time']
            if elapsed > 1.0:  # Only show if finger has been still for more than 1 second
                remaining = max(0, self.stop_threshold - elapsed)
                if remaining > 0:
                    cv2.putText(frame, f"{finger_id}: {remaining:.1f}s to stop", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    y_offset += 20
    
    def start_video_recording(self):
        """Start video recording for timelapse"""
        if not self.recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"multifinger_timelapse_{timestamp}.avi"
            
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
    
    def run(self):
        """Main application loop"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                original_frame = frame.copy()
                
                # Process hands and body
                frame = self.process_hands(frame)
                self.process_body(original_frame)
                
                # Combine all layers
                display_frame = cv2.addWeighted(self.body_canvas, 0.3, frame, 0.7, 0)
                display_frame = cv2.addWeighted(display_frame, 0.7, self.drawing_canvas, 1.0, 0)
                
                # Apply kaleidoscope if enabled
                if self.kaleidoscope_mode:
                    display_frame = self.apply_kaleidoscope_effect(display_frame)
                
                # Add UI elements
                self.draw_ui(display_frame)
                
                # Record frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(display_frame)
                
                cv2.imshow('Multi-Finger Kinetic Canvas', display_frame)
                
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
                elif key == ord('s'):
                    self.save_image()
                elif key == ord('v'):
                    if self.recording:
                        self.stop_video_recording()
                    else:
                        self.start_video_recording()
                elif key == ord('u'):
                    # Manual undo
                    if self.canvas_history:
                        self.drawing_canvas = self.canvas_history.pop().copy()
                        print("Manual undo applied!")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.audio_running = False
        
        if self.video_writer:
            self.video_writer.release()
        
        if hasattr(self, 'audio') and self.audio:
            self.audio.terminate()
        
        self.cap.release()
        cv2.destroyAllWindows()


# Simplified version for testing without audio
class SimpleMultiFingerCanvas:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.finger_drawing_states = {}
        self.finger_stop_timers = {}
        self.finger_positions_history = {}
        self.stop_threshold = 5.0
        self.stop_distance_threshold = 30
        
        self.snap_detection = {'last_snap': 0, 'snap_threshold': 1.0}
        self.five_finger_swipe = {'detecting': False, 'start_positions': [], 'start_time': 0}
        self.thumbs_up_detection = {'left': False, 'right': False, 'last_detection': 0}
        
        self.finger_colors = [
            (255, 100, 100),  # Thumb
            (100, 255, 100),  # Index  
            (100, 100, 255),  # Middle
            (255, 255, 100),  # Ring
            (255, 100, 255)   # Pinky
        ]
        
        print("Simple Multi-Finger Canvas - All gesture controls active!")
    
    def get_finger_positions(self, hand_landmarks):
        """Get all 5 finger tip positions"""
        finger_tips = [4, 8, 12, 16, 20]
        positions = []
        
        for tip_id in finger_tips:
            landmark = hand_landmarks.landmark[tip_id]
            x = int(landmark.x * self.width)
            y = int(landmark.y * self.height)
            positions.append((x, y))
        
        return positions
    
    def detect_finger_snap(self, hand_landmarks):
        """Detect snap gesture"""
        thumb_tip = hand_landmarks.landmark[4]
        middle_tip = hand_landmarks.landmark[12]
        
        thumb_pos = (thumb_tip.x * self.width, thumb_tip.y * self.height)
        middle_pos = (middle_tip.x * self.width, middle_tip.y * self.height)
        
        distance = np.sqrt((thumb_pos[0] - middle_pos[0])**2 + (thumb_pos[1] - middle_pos[1])**2)
        current_time = time.time()
        
        if distance < 30 and (current_time - self.snap_detection['last_snap']) > self.snap_detection['snap_threshold']:
            self.snap_detection['last_snap'] = current_time
            return True
        
        return False
    
    def detect_five_finger_swipe(self, all_finger_positions):
        """Detect 5-finger swipe"""
        current_time = time.time()
        
        if len(all_finger_positions) >= 5:
            if not self.five_finger_swipe['detecting']:
                self.five_finger_swipe['detecting'] = True
                self.five_finger_swipe['start_positions'] = all_finger_positions[:5]
                self.five_finger_swipe['start_time'] = current_time
            else:
                if (current_time - self.five_finger_swipe['start_time']) < 2.0:
                    movements = []
                    for start_pos, current_pos in zip(self.five_finger_swipe['start_positions'], all_finger_positions[:5]):
                        dx = current_pos[0] - start_pos[0]
                        movements.append(dx)
                    
                    avg_dx = np.mean(movements)
                    
                    if abs(avg_dx) > 200:  # Swipe detected
                        self.five_finger_swipe['detecting'] = False
                        return True
                else:
                    self.five_finger_swipe['detecting'] = False
        else:
            self.five_finger_swipe['detecting'] = False
        
        return False
    
    def detect_thumbs_up(self, hand_landmarks, hand_label):
        """Detect thumbs up"""
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        index_tip = hand_landmarks.landmark[8]
        
        thumb_up = thumb_tip.y < thumb_mcp.y - 0.05
        index_folded = index_tip.y > hand_landmarks.landmark[6].y
        
        is_thumbs_up = thumb_up and index_folded
        self.thumbs_up_detection[hand_label] = is_thumbs_up
        
        return is_thumbs_up
    
    def check_finger_stopped(self, finger_id, position):
        """Check if finger stopped for 5 seconds"""
        current_time = time.time()
        
        if finger_id not in self.finger_stop_timers:
            self.finger_stop_timers[finger_id] = {'start_time': current_time, 'position': position}
            self.finger_positions_history[finger_id] = deque(maxlen=30)
            return False
        
        self.finger_positions_history[finger_id].append(position)
        
        if len(self.finger_positions_history[finger_id]) >= 10:
            recent_positions = list(self.finger_positions_history[finger_id])[-10:]
            x_var = np.var([pos[0] for pos in recent_positions])
            y_var = np.var([pos[1] for pos in recent_positions])
            
            if x_var < self.stop_distance_threshold and y_var < self.stop_distance_threshold:
                if (current_time - self.finger_stop_timers[finger_id]['start_time']) >= self.stop_threshold:
                    return True
            else:
                self.finger_stop_timers[finger_id] = {'start_time': current_time, 'position': position}
        
        return False
    
    def save_image(self):
        """Save canvas"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"simple_multifinger_{timestamp}.jpg"
        cv2.imwrite(filename, self.drawing_canvas)
        print(f"Saved: {filename}")
    
    def clear_canvas(self):
        """Clear canvas"""
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.finger_drawing_states.clear()
        self.finger_stop_timers.clear()
        self.finger_positions_history.clear()
        print("Canvas cleared!")
    
    def run(self):
        """Main loop"""
        print("Gesture Controls:")
        print("- All 5 fingers draw with different colors")
        print("- Stop finger at same position for 5 seconds to stop drawing")
        print("- Snap fingers to save image")
        print("- 5-finger swipe to clear canvas")
        print("- Both thumbs up to save and mark completion")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb_frame)
            
            all_finger_positions = []
            snap_detected = False
            current_time = time.time()
            
            if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                for hand_landmarks, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                    hand_label = handedness.classification[0].label.lower()
                    
                    finger_positions = self.get_finger_positions(hand_landmarks)
                    all_finger_positions.extend(finger_positions)
                    
                    if self.detect_finger_snap(hand_landmarks):
                        snap_detected = True
                    
                    if self.detect_thumbs_up(hand_landmarks, hand_label):
                        cv2.putText(frame, f"THUMBS UP - {hand_label.upper()}", 
                                   (10, 100 if hand_label == 'left' else 130), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Process each finger
                    for finger_idx, finger_pos in enumerate(finger_positions):
                        finger_id = f"{hand_label}_{finger_idx}"
                        color = self.finger_colors[finger_idx]
                        
                        if self.check_finger_stopped(finger_id, finger_pos):
                            if finger_id in self.finger_drawing_states:
                                del self.finger_drawing_states[finger_id]
                                print(f"Stopped: {hand_label} {['thumb','index','middle','ring','pinky'][finger_idx]}")
                        else:
                            if finger_id in self.finger_drawing_states:
                                # Draw line
                                last_pos = self.finger_drawing_states[finger_id]
                                cv2.line(self.drawing_canvas, last_pos, finger_pos, color, 3, cv2.LINE_AA)
                                self.finger_drawing_states[finger_id] = finger_pos
                            else:
                                # Start drawing
                                self.finger_drawing_states[finger_id] = finger_pos
                        
                        # Visual feedback
                        if finger_id in self.finger_drawing_states:
                            cv2.circle(frame, finger_pos, 8, color, -1)
                        else:
                            cv2.circle(frame, finger_pos, 8, color, 2)
                        
                        # Finger labels
                        cv2.putText(frame, ['T','I','M','R','P'][finger_idx], 
                                   (finger_pos[0] - 5, finger_pos[1] - 15), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Handle gestures
            if snap_detected:
                self.save_image()
                print("Snap - Screenshot saved!")
            
            if self.detect_five_finger_swipe(all_finger_positions):
                self.clear_canvas()
                print("5-finger swipe - Canvas cleared!")
            
            if (self.thumbs_up_detection['left'] and self.thumbs_up_detection['right'] and 
                (current_time - self.thumbs_up_detection.get('last_detection', 0)) > 2.0):
                self.thumbs_up_detection['last_detection'] = current_time
                self.save_image()
                print("Both thumbs up - Completed!")
            
            # Combine and display
            display = cv2.addWeighted(frame, 0.5, self.drawing_canvas, 0.8, 0)
            
            # UI
            cv2.putText(display, f"Active: {len(self.finger_drawing_states)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Simple Multi-Finger Canvas', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clear_canvas()
            elif key == ord('s'):
                self.save_image()
        
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Choose version:")
    print("1. Full Multi-Finger Canvas (with audio, voice, advanced brushes)")
    print("2. Simple Multi-Finger Canvas (gesture controls only)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    try:
        if choice == "2":
            canvas = SimpleMultiFingerCanvas()
        else:
            canvas = MultiFingerKineticCanvas()
        
        canvas.run()
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install opencv-python mediapipe pyaudio SpeechRecognition numpy")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure camera and microphone are connected")