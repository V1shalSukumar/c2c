import cv2
import numpy as np
import sounddevice as sd
from scipy.fftpack import fft
from PIL import Image, ImageDraw
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Indices for finger tips in MediaPipe (Thumb, Index, Middle, Ring, Pinky)
finger_tip_ids = [4, 8, 12, 16, 20]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for idx in finger_tip_ids:
                h, w, _ = frame.shape
                x = int(hand_landmarks.landmark[idx].x * w)
                y = int(hand_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 15, (0,255,0), cv2.FILLED) # Draw fingers only
    cv2.imshow('Finger Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# --- Global Variables ---
drawing_color = (255, 255, 255)
last_position = None
volume_level = 0
pitch_value = 0

# --- Audio Callback ---
def audio_callback(indata, frames, time, status):
    global volume_level, pitch_value, drawing_color

    # Flatten audio input
    audio_data = indata[:, 0]
    
    # Volume = RMS amplitude
    volume_level = np.linalg.norm(audio_data)

    # FFT for pitch detection
    fft_data = np.abs(fft(audio_data))
    freqs = np.fft.fftfreq(len(fft_data), 1/44100)
    peak_idx = np.argmax(fft_data[:len(fft_data)//2])
    pitch_value = abs(freqs[peak_idx])

    # Map volume to color (higher volume = brighter color)
    r = int(min(255, volume_level * 800) % 255)
    g = int(min(255, volume_level * 500) % 255)
    b = int(min(255, 255 - r))
    drawing_color = (r, g, b)

# Start microphone stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=44100, blocksize=1024)
stream.start()

# --- Main Program ---
def main():
    global last_position, drawing_color, volume_level, pitch_value
    
    # Initialize video capture
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    # Create art canvas
    canvas_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)), int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    art_canvas = Image.new('RGB', canvas_size, 'black')
    draw = ImageDraw.Draw(art_canvas)

    while True:
        ret, current_frame = camera.read()
        if not ret or current_frame is None:
            break

        # Flip horizontally (mirror correction)
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

        # Draw with sound-reactive brush
        if current_position is not None:
            # Pitch → Brush Size (low pitch = small, high pitch = big)
            brush_size = int(min(max(pitch_value / 50, 5), 80))  
            x, y = current_position
            draw.ellipse([x - brush_size, y - brush_size, x + brush_size, y + brush_size], fill=drawing_color)

        # Show feeds
        cv2.imshow('Live Feed', current_frame)
        cv2.imshow('Kinetic Canvas', np.array(art_canvas))

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    stream.stop()

if __name__ == "__main__":
    main()
