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
        
        # Canvas layers
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.drawing_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.body_canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Drawing state
        self.finger_positions = deque(maxlen=20)
        self.finger_velocities = deque(maxlen=10)
        self.finger_start_time = None
        self.is_drawing_mode = False
        self.drawing_threshold = 2.0  # Reduced to 2 seconds
        self.last_drawing_pos = None
        
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
        
        # Color system - pitch based
        self.hue = 0.0
        
        print("Advanced Kinetic Canvas Initialized!")
        print("Features:")
        print("- Pitch controls color, Volume controls brush size")
        print("- Say 'Clear' to clear canvas")
        print("- Swipe left to undo")
        print("- Press 'k' for kaleidoscope mode")
        print("- Press 'b' to change brush type")
        print("- Press 's' to save image")
        print("- Press 'v' to start/stop video recording")
        print("- Press 'q' to quit")
        
        self.init_audio()
        self.init_voice_recognition()
    
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
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
                    
            except sr.WaitTimeoutError:
                pass
            except Exception as e:
                continue
    
    def get_dynamic_color(self):
        """Get color based on current pitch"""
        # Convert hue to RGB
        rgb = colorsys.hsv_to_rgb(self.hue, 0.9, 1.0)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR for OpenCV
    
    def get_dynamic_brush_size(self):
        """Get brush size based on volume"""
        volume_factor = min(self.current_volume / 20.0, 3.0)  # Cap at 3x
        return int(self.brush_size_base * (0.5 + volume_factor))
    
    def calculate_velocity(self, current_pos):
        """Calculate finger movement velocity"""
        if len(self.finger_positions) < 2:
            return 0
        
        prev_pos = self.finger_positions[-1]
        distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        self.finger_velocities.append(distance)
        
        return np.mean(list(self.finger_velocities)) if self.finger_velocities else 0
    
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
    
    def apply_brush(self, start_pos, end_pos, velocity):
        """Apply current brush effect"""
        color = self.get_dynamic_color()
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
    
    def get_finger_tip(self, hand_landmarks):
        """Get index finger tip position"""
        if hand_landmarks:
            finger_tip = hand_landmarks.landmark[8]
            x = int(finger_tip.x * self.width)
            y = int(finger_tip.y * self.height)
            return (x, y)
        return None
    
    def process_hands(self, frame):
        """Process hand detection and drawing"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        
        current_time = time.time()
        finger_tip = None
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Check for swipe gesture
                if self.detect_swipe_left(hand_landmarks):
                    break
                
                finger_tip = self.get_finger_tip(hand_landmarks)
                break
        
        # Handle drawing mode
        if finger_tip:
            self.finger_positions.append(finger_tip)
            
            if self.finger_start_time is None:
                self.finger_start_time = current_time
            
            time_held = current_time - self.finger_start_time
            
            if time_held >= self.drawing_threshold and not self.is_drawing_mode:
                self.is_drawing_mode = True
                self.last_drawing_pos = finger_tip
                self.save_canvas_state()
                print("Drawing mode activated!")
            
            if self.is_drawing_mode:
                velocity = self.calculate_velocity(finger_tip)
                
                if self.last_drawing_pos:
                    self.apply_brush(self.last_drawing_pos, finger_tip, velocity)
                
                self.last_drawing_pos = finger_tip
                
                # Visual indicator
                color = self.get_dynamic_color()
                cv2.circle(frame, finger_tip, 15, color, 3)
            else:
                # Show activation progress
                progress = time_held / self.drawing_threshold
                cv2.circle(frame, finger_tip, int(10 + 10 * progress), (0, 255, 0), 2)
        else:
            if self.is_drawing_mode:
                print("Drawing mode deactivated!")
            self.finger_start_time = None
            self.is_drawing_mode = False
            self.last_drawing_pos = None
            self.finger_positions.clear()
            self.finger_velocities.clear()
        
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
                
                # Process hands and body
                frame = self.process_hands(frame)
                self.process_body(original_frame)
                
                # Combine all layers
                display_frame = cv2.addWeighted(self.body_canvas, 0.4, frame, 0.6, 0)
                display_frame = cv2.addWeighted(display_frame, 0.7, self.drawing_canvas, 1.0, 0)
                
                # Apply kaleidoscope if enabled
                if self.kaleidoscope_mode:
                    display_frame = self.apply_kaleidoscope_effect(display_frame)
                
                # Add UI elements
                self.draw_ui(display_frame)
                
                # Record frame if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(display_frame)
                
                cv2.imshow('Advanced Kinetic Canvas', display_frame)
                
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
        
        finally:
            self.cleanup()
    
    def draw_ui(self, frame):
        """Draw UI information"""
        y_offset = 30
        
        # Mode indicator
        mode_text = "DRAWING" if self.is_drawing_mode else "TRACKING"
        color = (0, 255, 0) if self.is_drawing_mode else (255, 255, 255)
        cv2.putText(frame, mode_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
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
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (self.width - 50, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (self.width - 80, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Kaleidoscope indicator
        if self.kaleidoscope_mode:
            cv2.putText(frame, "KALEIDOSCOPE", (self.width - 200, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
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
    print("pip install opencv-python mediapipe pyaudio SpeechRecognition numpy")
    
    try:
        canvas = AdvancedKineticCanvas()
        canvas.run()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install opencv-python mediapipe pyaudio SpeechRecognition numpy")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your camera and microphone are connected")