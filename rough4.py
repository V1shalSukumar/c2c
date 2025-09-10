import cv2
import numpy as np
import pygame
from threading import Thread
import time
import random
import math
from scipy import ndimage

class KineticCanvasOpenCV:
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
        
        # Background subtractor for person detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, 
            varThreshold=50,
            history=500
        )
        
        # Optical flow for movement tracking
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Background and art layers
        self.background_image = None
        self.art_canvas = pygame.Surface((width, height), pygame.SRCALPHA)
        self.person_layer = pygame.Surface((width, height), pygame.SRCALPHA)
        
        # Movement tracking
        self.prev_gray = None
        self.tracking_points = np.array([], dtype=np.float32).reshape(0, 1, 2)
        self.movement_trail = []
        self.last_movement_time = time.time()
        
        # Person detection
        self.person_present = False
        self.min_person_area = 5000  # Minimum area to consider as person
        self.person_contour = None
        self.person_center = None
        
        # Art generation parameters
        self.brush_size = 25
        self.colors = [
            (255, 80, 80, 200),    # Vibrant red
            (80, 255, 80, 200),    # Vibrant green
            (80, 80, 255, 200),    # Vibrant blue
            (255, 255, 80, 200),   # Vibrant yellow
            (255, 80, 255, 200),   # Vibrant magenta
            (80, 255, 255, 200),   # Vibrant cyan
            (255, 150, 0, 200),    # Orange
            (150, 0, 255, 200),    # Purple
        ]
        self.current_color = random.choice(self.colors)
        
        # State management
        self.interaction_mode = False
        self.stillness_timer = 0
        self.stillness_threshold = 1.5  # seconds to activate art mode
        self.movement_sensitivity = 15.0
        
        # Calibration
        self.calibration_frames = 30
        self.current_calibration = 0
        self.calibrated = False
        
        self.clock = pygame.time.Clock()
        self.running = True
        
    def load_background(self, image_path):
        """Load and scale background image"""
        try:
            self.background_image = pygame.image.load(image_path)
            self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))
        except:
            self.create_default_background()
    
    def create_default_background(self):
        """Create a beautiful gradient garden background"""
        self.background_image = pygame.Surface((self.width, self.height))
        
        # Sky gradient (top to middle)
        for y in range(self.height // 2):
            ratio = y / (self.height // 2)
            r = int(135 + (255 - 135) * (1 - ratio))
            g = int(206 + (200 - 206) * ratio)
            b = int(235 + (150 - 235) * ratio)
            color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
            pygame.draw.line(self.background_image, color, (0, y), (self.width, y))
        
        # Ground gradient (middle to bottom)
        for y in range(self.height // 2, self.height):
            ratio = (y - self.height // 2) / (self.height // 2)
            r = int(34 + (20 - 34) * ratio)
            g = int(139 + (80 - 139) * ratio)
            b = int(34 + (20 - 34) * ratio)
            color = (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
            pygame.draw.line(self.background_image, color, (0, y), (self.width, y))
        
        # Add decorative elements
        self.add_decorative_elements()
    
    def add_decorative_elements(self):
        """Add trees, flowers, and other garden elements"""
        # Trees
        tree_positions = [(200, self.height//2), (400, self.height//2), 
                         (self.width-300, self.height//2), (self.width-100, self.height//2)]
        
        for x, y in tree_positions:
            # Trunk
            trunk_width = random.randint(15, 25)
            trunk_height = random.randint(80, 120)
            trunk_color = (101, 67, 33)
            pygame.draw.rect(self.background_image, trunk_color, 
                           (x - trunk_width//2, y - trunk_height, trunk_width, trunk_height))
            
            # Canopy
            canopy_size = random.randint(50, 80)
            canopy_colors = [(0, 100, 0), (34, 139, 34), (0, 128, 0)]
            canopy_color = random.choice(canopy_colors)
            pygame.draw.circle(self.background_image, canopy_color, 
                             (x, y - trunk_height + 20), canopy_size)
            
            # Add some texture to canopy
            for _ in range(10):
                offset_x = random.randint(-canopy_size//2, canopy_size//2)
                offset_y = random.randint(-canopy_size//2, canopy_size//2)
                small_size = random.randint(10, 20)
                dark_green = (max(0, canopy_color[0] - 30), 
                             max(0, canopy_color[1] - 30), 
                             max(0, canopy_color[2] - 30))
                pygame.draw.circle(self.background_image, dark_green,
                                 (x + offset_x, y - trunk_height + 20 + offset_y), small_size)
        
        # Flowers scattered on ground
        for _ in range(20):
            fx = random.randint(50, self.width - 50)
            fy = random.randint(self.height//2 + 50, self.height - 50)
            flower_colors = [(255, 100, 100), (255, 255, 100), (255, 0, 255), (100, 100, 255)]
            flower_color = random.choice(flower_colors)
            pygame.draw.circle(self.background_image, flower_color, (fx, fy), 8)
            pygame.draw.circle(self.background_image, (255, 255, 0), (fx, fy), 3)  # Center
    
    def calibrate_background(self, frame):
        """Calibrate background subtractor"""
        if self.current_calibration < self.calibration_frames:
            self.bg_subtractor.apply(frame)
            self.current_calibration += 1
            return False
        else:
            self.calibrated = True
            return True
    
    def detect_person_opencv(self, frame):
        """Detect person using OpenCV background subtraction"""
        if not self.calibrated:
            return None, None, None
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        person_contour = None
        person_mask = None
        person_center = None
        
        if contours:
            # Find largest contour (assuming it's the person)
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > self.min_person_area:
                self.person_present = True
                person_contour = largest_contour
                
                # Create person mask
                person_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.fillPoly(person_mask, [largest_contour], 255)
                
                # Smooth the mask
                person_mask = cv2.GaussianBlur(person_mask, (21, 21), 0)
                
                # Calculate center of mass
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    person_center = (cx, cy)
                
                # Scale mask to screen size
                person_mask = cv2.resize(person_mask, (self.width, self.height))
            else:
                self.person_present = False
        else:
            self.person_present = False
        
        return person_contour, person_mask, person_center
    
    def track_movement_optical_flow(self, frame):
        """Track movement using Lucas-Kanade optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        movement_detected = False
        movement_vectors = []
        
        if self.prev_gray is not None and len(self.tracking_points) > 0:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.tracking_points, None, **self.lk_params
            )
            
            # Select good points
            good_new = new_points[status == 1]
            good_old = self.tracking_points[status == 1]
            
            # Calculate movement vectors
            if len(good_new) > 0 and len(good_old) > 0:
                for new, old in zip(good_new, good_old):
                    dx = new[0] - old[0]
                    dy = new[1] - old[1]
                    magnitude = np.sqrt(dx*dx + dy*dy)
                    
                    if magnitude > self.movement_sensitivity:
                        movement_detected = True
                        movement_vectors.append((dx, dy, magnitude))
                
                self.tracking_points = good_new.reshape(-1, 1, 2)
        
        # Detect new features to track if we have too few
        if len(self.tracking_points) < 20 and self.person_present:
            mask = np.zeros_like(gray)
            if self.person_contour is not None:
                cv2.fillPoly(mask, [self.person_contour], 255)
                
            new_corners = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if new_corners is not None:
                self.tracking_points = np.vstack([self.tracking_points, new_corners])
        
        self.prev_gray = gray.copy()
        
        return movement_detected, movement_vectors
    
    def create_art_from_movement(self, movement_vectors, person_center):
        """Generate art based on movement vectors"""
        if not movement_vectors or not person_center:
            return
        
        # Scale person center to screen coordinates
        screen_x = int(person_center[0] * self.width / 640)  # Assuming 640 is camera width
        screen_y = int(person_center[1] * self.height / 480)  # Assuming 480 is camera height
        
        # Calculate average movement
        avg_magnitude = sum(v[2] for v in movement_vectors) / len(movement_vectors)
        avg_dx = sum(v[0] for v in movement_vectors) / len(movement_vectors)
        avg_dy = sum(v[1] for v in movement_vectors) / len(movement_vectors)
        
        # Scale movement to screen coordinates
        scaled_dx = avg_dx * self.width / 640
        scaled_dy = avg_dy * self.height / 480
        
        # Create art based on movement type
        if avg_magnitude > 40:  # Large movement - burst effect
            self.create_burst_effect(screen_x, screen_y, avg_magnitude)
        elif avg_magnitude > 20:  # Medium movement - flowing lines
            self.create_flowing_lines(screen_x, screen_y, scaled_dx, scaled_dy, avg_magnitude)
        else:  # Small movement - particles
            self.create_particle_effect(screen_x, screen_y, avg_magnitude)
        
        # Add to movement trail
        self.movement_trail.append((screen_x, screen_y))
        if len(self.movement_trail) > 30:
            self.movement_trail.pop(0)
        
        # Occasionally change color
        if random.random() < 0.05:
            self.current_color = random.choice(self.colors)
    
    def create_burst_effect(self, x, y, magnitude):
        """Create explosive burst effect"""
        intensity = min(magnitude / 100.0, 1.0)
        num_rays = int(8 + intensity * 12)
        
        for i in range(num_rays):
            angle = (2 * math.pi * i) / num_rays
            length = int(30 + intensity * 70)
            
            end_x = x + int(math.cos(angle) * length)
            end_y = y + int(math.sin(angle) * length)
            
            # Draw gradient ray
            steps = max(length // 5, 1)
            for step in range(steps):
                t = step / steps
                curr_x = int(x + (end_x - x) * t)
                curr_y = int(y + (end_y - y) * t)
                
                alpha = int(self.current_color[3] * (1 - t))
                color = (*self.current_color[:3], alpha)
                radius = int(self.brush_size * (1 - t * 0.7))
                
                if 0 <= curr_x < self.width and 0 <= curr_y < self.height and radius > 0:
                    pygame.draw.circle(self.art_canvas, color, (curr_x, curr_y), radius)
    
    def create_flowing_lines(self, x, y, dx, dy, magnitude):
        """Create flowing artistic lines"""
        if len(self.movement_trail) > 1:
            prev_x, prev_y = self.movement_trail[-1]
            
            # Create organic flowing line
            steps = max(int(math.sqrt(dx*dx + dy*dy) / 8), 1)
            
            for i in range(steps):
                t = i / max(steps - 1, 1)
                
                # Bezier-like curve
                curve_x = int(prev_x + (x - prev_x) * t)
                curve_y = int(prev_y + (y - prev_y) * t)
                
                # Add organic variation
                wave = math.sin(t * math.pi * 2) * magnitude * 0.2
                curve_x += int(wave * math.cos(math.atan2(dy, dx) + math.pi/2))
                curve_y += int(wave * math.sin(math.atan2(dy, dx) + math.pi/2))
                
                # Varying thickness
                thickness = int(self.brush_size * (0.5 + 0.5 * math.sin(t * math.pi)))
                
                if 0 <= curve_x < self.width and 0 <= curve_y < self.height and thickness > 0:
                    pygame.draw.circle(self.art_canvas, self.current_color, 
                                     (curve_x, curve_y), thickness)
    
    def create_particle_effect(self, x, y, magnitude):
        """Create particle-like effects for subtle movements"""
        num_particles = int(5 + magnitude)
        
        for _ in range(num_particles):
            offset_x = random.randint(-30, 30)
            offset_y = random.randint(-30, 30)
            
            particle_x = x + offset_x
            particle_y = y + offset_y
            
            if 0 <= particle_x < self.width and 0 <= particle_y < self.height:
                particle_size = random.randint(3, int(self.brush_size // 3))
                particle_color = (*self.current_color[:3], 
                                random.randint(100, self.current_color[3]))
                
                pygame.draw.circle(self.art_canvas, particle_color, 
                                 (particle_x, particle_y), particle_size)
    
    def render_person_with_background(self, frame, mask):
        """Composite person onto background using mask"""
        self.person_layer.fill((0, 0, 0, 0))  # Clear layer
        
        if frame is not None and mask is not None:
            # Scale frame to screen size
            frame_resized = cv2.resize(frame, (self.width, self.height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Apply person mask
            for y in range(self.height):
                for x in range(self.width):
                    if mask[y, x] > 50:  # Threshold for mask
                        color = frame_rgb[y, x]
                        alpha = min(255, int(mask[y, x]))
                        self.person_layer.set_at((x, y), (*color, alpha))
    
    def fade_art_canvas(self):
        """Gradually fade the art canvas"""
        if not self.person_present:
            fade_surface = pygame.Surface((self.width, self.height))
            fade_surface.fill((0, 0, 0))
            fade_surface.set_alpha(3)  # Slow fade
            self.art_canvas.blit(fade_surface, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)
    
    def run(self):
        """Main application loop"""
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
                    elif event.key == pygame.K_r:  # Recalibrate
                        self.current_calibration = 0
                        self.calibrated = False
                        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                            detectShadows=True, varThreshold=50, history=500
                        )
            
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Calibration phase
            if not self.calibrated:
                self.calibrate_background(frame)
                # Show calibration status
                font = pygame.font.Font(None, 48)
                cal_text = f"Calibrating... {self.current_calibration}/{self.calibration_frames}"
                text_surface = font.render(cal_text, True, (255, 255, 255))
                self.screen.fill((0, 0, 0))
                text_rect = text_surface.get_rect(center=(self.width//2, self.height//2))
                self.screen.blit(text_surface, text_rect)
                pygame.display.flip()
                continue
            
            # Detect person
            person_contour, person_mask, person_center = self.detect_person_opencv(frame)
            self.person_contour = person_contour
            self.person_center = person_center
            
            # Track movement and create art
            if self.person_present:
                movement_detected, movement_vectors = self.track_movement_optical_flow(frame)
                
                if movement_detected and movement_vectors:
                    self.interaction_mode = True
                    self.create_art_from_movement(movement_vectors, person_center)
                    self.last_movement_time = time.time()
                else:
                    # Check for stillness
                    if time.time() - self.last_movement_time > self.stillness_threshold:
                        self.interaction_mode = False
            else:
                self.movement_trail.clear()
                self.tracking_points = np.array([], dtype=np.float32).reshape(0, 1, 2)
            
            # Render all layers
            self.screen.blit(self.background_image, (0, 0))  # Background
            
            if self.person_present and person_mask is not None:
                self.render_person_with_background(frame, person_mask)
                self.screen.blit(self.person_layer, (0, 0))  # Person layer
            
            self.screen.blit(self.art_canvas, (0, 0))  # Art layer
            
            # Fade art when no interaction
            self.fade_art_canvas()
            
            # Status display
            font = pygame.font.Font(None, 36)
            if not self.calibrated:
                status_text = "Calibrating background..."
                status_color = (255, 255, 0)
            elif self.person_present:
                if self.interaction_mode:
                    status_text = "Creating Art with Movement!"
                    status_color = (0, 255, 0)
                else:
                    status_text = "Move to create art!"
                    status_color = (255, 255, 0)
            else:
                status_text = "Step into the magical garden!"
                status_color = (255, 100, 100)
            
            text_surface = font.render(status_text, True, status_color)
            self.screen.blit(text_surface, (50, 50))
            
            # Show controls
            controls_font = pygame.font.Font(None, 24)
            controls = ["ESC: Exit", "C: Clear Art", "R: Recalibrate Background"]
            for i, control in enumerate(controls):
                control_surface = controls_font.render(control, True, (200, 200, 200))
                self.screen.blit(control_surface, (50, self.height - 100 + i * 25))
            
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
        
        # Cleanup
        self.cap.release()
        pygame.quit()

# Example usage
if __name__ == "__main__":
    canvas = KineticCanvasOpenCV()
    
    # Optionally load a background image
    # canvas.load_background("your_garden_background.jpg")
    
    canvas.run()