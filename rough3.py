import cv2
import numpy as np
import pygame
import mediapipe as mp
from threading import Thread
import time
import random
import math

class KineticCanvas:
    def __init__(self, width=1920, height=1080):
        # Initialize pygame
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
        pygame.display.set_caption("Kinetic Canvas - Interactive Wall")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Initialize MediaPipe for person detection and pose estimation
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5
        )
        
        # Background and art layers
        self.background_image = None
        self.art_canvas = pygame.Surface((width, height), pygame.SRCALPHA)
        self.person_layer = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Movement tracking
        self.prev_landmarks = None
        self.movement_trail = []
        self.last_movement_time = time.time()
        
        # Art generation parameters
        self.brush_size = 20
        self.colors = [
            (255, 100, 100, 180),  # Red with transparency
            (100, 255, 100, 180),  # Green
            (100, 100, 255, 180),  # Blue
            (255, 255, 100, 180),  # Yellow
            (255, 100, 255, 180),  # Magenta
            (100, 255, 255, 180),  # Cyan
        ]
        self.current_color = random.choice(self.colors)
        
        # State management
        self.person_present = False
        self.interaction_mode = False  # True when person is standing still for art creation
        self.stillness_timer = 0
        self.stillness_threshold = 2.0  # seconds to activate art mode
        
        self.clock = pygame.time.Clock()
        self.running = True
        
    def load_background(self, image_path):
        """Load and scale background image"""
        try:
            # Load with pygame for better integration
            self.background_image = pygame.image.load(image_path)
            self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))
        except:
            # Create a default garden-like background
            self.create_default_background()
    
    def create_default_background(self):
        """Create a simple garden background"""
        self.background_image = pygame.Surface((self.width, self.height))
        # Sky gradient
        for y in range(self.height // 2):
            color = (135, 206, 235 - int(y * 0.1))  # Sky blue gradient
            pygame.draw.line(self.background_image, color, (0, y), (self.width, y))
        
        # Ground
        ground_color = (34, 139, 34)  # Forest green
        pygame.draw.rect(self.background_image, ground_color, 
                        (0, self.height // 2, self.width, self.height // 2))
        
        # Simple trees
        for i in range(5):
            x = (self.width // 6) * (i + 1)
            y = self.height // 2
            # Trunk
            pygame.draw.rect(self.background_image, (101, 67, 33), 
                           (x - 10, y - 100, 20, 100))
            # Leaves
            pygame.draw.circle(self.background_image, (0, 100, 0), 
                             (x, y - 80), 40)
    
    def process_camera_frame(self):
        """Process camera input and detect person"""
        ret, frame = self.cap.read()
        if not ret:
            return None, None
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        person_mask = None
        landmarks = None
        
        if results.pose_landmarks and results.segmentation_mask is not None:
            self.person_present = True
            landmarks = results.pose_landmarks.landmark
            
            # Create person mask
            mask = results.segmentation_mask
            mask = (mask > 0.5).astype(np.uint8) * 255
            
            # Scale mask to screen size
            mask_resized = cv2.resize(mask, (self.width, self.height))
            person_mask = mask_resized
            
            # Scale frame to screen size
            frame_resized = cv2.resize(frame, (self.width, self.height))
            
        else:
            self.person_present = False
            
        return frame_resized if self.person_present else None, person_mask, landmarks
    
    def detect_movement(self, landmarks):
        """Detect significant movement from pose landmarks"""
        if not landmarks or not self.prev_landmarks:
            self.prev_landmarks = landmarks
            return False, (0, 0)
            
        # Calculate movement of key points (hands, head)
        movement_threshold = 0.05
        significant_movement = False
        avg_movement = [0, 0]
        
        key_points = [11, 12, 15, 16]  # Shoulders and hands
        
        for i in key_points:
            if i < len(landmarks) and i < len(self.prev_landmarks):
                curr = landmarks[i]
                prev = self.prev_landmarks[i]
                
                dx = abs(curr.x - prev.x)
                dy = abs(curr.y - prev.y)
                
                if dx > movement_threshold or dy > movement_threshold:
                    significant_movement = True
                    
                avg_movement[0] += curr.x * self.width
                avg_movement[1] += curr.y * self.height
        
        avg_movement[0] /= len(key_points)
        avg_movement[1] /= len(key_points)
        
        self.prev_landmarks = landmarks
        return significant_movement, avg_movement
    
    def create_art_from_movement(self, position, movement_intensity):
        """Generate art based on movement"""
        x, y = int(position[0]), int(position[1])
        
        # Vary brush size based on movement intensity
        dynamic_brush_size = int(self.brush_size * (1 + movement_intensity))
        
        # Add movement trail
        if len(self.movement_trail) > 0:
            prev_x, prev_y = self.movement_trail[-1]
            
            # Draw flowing line
            self.draw_flowing_line(
                (prev_x, prev_y), (x, y), 
                dynamic_brush_size, self.current_color
            )
        
        # Add current position to trail
        self.movement_trail.append((x, y))
        
        # Limit trail length
        if len(self.movement_trail) > 50:
            self.movement_trail.pop(0)
        
        # Occasionally change color based on movement
        if random.random() < 0.1:
            self.current_color = random.choice(self.colors)
    
    def draw_flowing_line(self, start, end, thickness, color):
        """Draw a flowing artistic line"""
        # Create organic, flowing brush strokes
        steps = int(math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2) / 5)
        
        for i in range(steps):
            t = i / max(steps - 1, 1)
            
            # Interpolate position
            x = int(start[0] + (end[0] - start[0]) * t)
            y = int(start[1] + (end[1] - start[1]) * t)
            
            # Add some randomness for organic feel
            offset_x = random.randint(-thickness//4, thickness//4)
            offset_y = random.randint(-thickness//4, thickness//4)
            
            # Vary thickness along the stroke
            current_thickness = int(thickness * (0.5 + 0.5 * math.sin(t * math.pi)))
            
            pygame.draw.circle(self.art_canvas, color, 
                             (x + offset_x, y + offset_y), 
                             max(current_thickness, 1))
    
    def update_interaction_state(self, has_movement):
        """Update interaction state based on movement"""
        current_time = time.time()
        
        if not has_movement:
            # Person is still
            if current_time - self.last_movement_time > self.stillness_threshold:
                self.interaction_mode = True
            else:
                self.stillness_timer += 1
        else:
            # Person is moving
            self.last_movement_time = current_time
            self.interaction_mode = True  # Always in interaction mode when moving
            self.stillness_timer = 0
    
    def render_person_with_background(self, frame, mask):
        """Composite person onto background"""
        self.person_layer.fill((0, 0, 0, 0))  # Clear layer
        
        if frame is not None and mask is not None:
            # Convert frame to pygame surface
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            # Create mask surface
            mask_surface = pygame.surfarray.make_surface(mask)
            
            # Apply mask to frame
            for x in range(self.width):
                for y in range(self.height):
                    if x < frame_surface.get_width() and y < frame_surface.get_height():
                        if mask[y, x] > 127:  # If pixel is part of person
                            color = frame_surface.get_at((x, y))
                            self.person_layer.set_at((x, y), color)
    
    def fade_art_canvas(self):
        """Gradually fade the art canvas when no one is present"""
        if not self.person_present:
            # Create fade effect
            fade_surface = pygame.Surface((self.width, self.height))
            fade_surface.fill((0, 0, 0))
            fade_surface.set_alpha(5)  # Very slow fade
            self.art_canvas.blit(fade_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
    
    def run(self):
        """Main game loop"""
        # Create default background if none loaded
        if self.background_image is None:
            self.create_default_background()
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_c:  # Clear art
                        self.art_canvas.fill((0, 0, 0, 0))
            
            # Process camera
            frame, mask, landmarks = self.process_camera_frame()
            
            # Detect movement and create art
            if self.person_present and landmarks:
                has_movement, position = self.detect_movement(landmarks)
                self.update_interaction_state(has_movement)
                
                if self.interaction_mode and has_movement:
                    # Calculate movement intensity
                    movement_intensity = min(1.0, len(self.movement_trail) / 20.0)
                    self.create_art_from_movement(position, movement_intensity)
            else:
                # Clear movement trail when no person detected
                self.movement_trail.clear()
            
            # Render layers
            self.screen.blit(self.background_image, (0, 0))  # Background
            
            if self.person_present:
                self.render_person_with_background(frame, mask)
                self.screen.blit(self.person_layer, (0, 0))  # Person layer
            
            self.screen.blit(self.art_canvas, (0, 0))  # Art layer
            
            # Fade art when no one is present
            self.fade_art_canvas()
            
            # Display status
            if self.person_present:
                status_color = (0, 255, 0) if self.interaction_mode else (255, 255, 0)
                status_text = "Creating Art!" if self.interaction_mode else "Stand Still to Create Art"
            else:
                status_color = (255, 0, 0)
                status_text = "Step into the scene!"
            
            # Simple text rendering (you might want to use a font)
            font = pygame.font.Font(None, 36)
            text_surface = font.render(status_text, True, status_color)
            self.screen.blit(text_surface, (50, 50))
            
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        # Cleanup
        self.cap.release()
        pygame.quit()

# Example usage
if __name__ == "__main__":
    canvas = KineticCanvas()
    
    # Optionally load a background image
    # canvas.load_background("garden_background.jpg")
    
    canvas.run()

#hi new change made