import cv2
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw

# --- Global Variables ---
drawing_color = (255, 255, 255)  # will update from sound
last_position = None
volume_level = 0

# --- Audio Callback ---
def audio_callback(indata, frames, time, status):
    global volume_level, drawing_color
    # Compute magnitude of sound
    volume_level = np.linalg.norm(indata)  

    # Map volume to color spectrum (R, G, B)
    # louder = brighter, softer = darker
    r = int(min(255, volume_level * 500))
    g = int(min(255, 255 - r))   # inverse relationship
    b = int(min(255, (volume_level * 800) % 255))
    drawing_color = (r, g, b)

# Start microphone stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=1024)
stream.start()

# --- Main Program ---
def main():
    global last_position, drawing_color
    
    # Initialize the video capture
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    # Create the art canvas
    canvas_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    art_canvas = Image.new('RGB', canvas_size, 'black')
    draw = ImageDraw.Draw(art_canvas)

    # --- Main Loop ---
    while True:
        ret, current_frame = camera.read()
        if not ret or current_frame is None:
            break

        # Flip horizontally to fix inversion
        current_frame = cv2.flip(current_frame, 1)

        # Hand segmentation
        ycrcb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb_frame, lower_skin, upper_skin)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 100)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        current_position = None
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > 5000:
                M = cv2.moments(max_contour)
                if M['m00'] != 0:
                    cX = int(M['m10'] / M['m00'])
                    cY = int(M['m01'] / M['m00'])
                    current_position = (cX, cY)

        # Draw on canvas using sound-based color
        if current_position is not None:
            dot_radius = 8
            x, y = current_position
            draw.ellipse([x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius], fill=drawing_color)

        # Display both feeds
        cv2.imshow('Live Feed', current_frame)
        cv2.imshow('Kinetic Canvas', np.array(art_canvas))

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    stream.stop()

if __name__ == "__main__":
    main()
