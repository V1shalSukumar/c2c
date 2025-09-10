print("we cookin")
import pygame
import sounddevice as sd
import numpy as np

# Settings
WIDTH, HEIGHT = 800, 600
FPS = 60

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Kinetic Canvas - Sound Reactive")
clock = pygame.time.Clock()

# Audio settings
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024

volume_level = 0

def audio_callback(indata, frames, time, status):
    global volume_level
    volume_level = np.linalg.norm(indata)  # magnitude of sound

# Start microphone stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE)
stream.start()

running = True
while running:
    screen.fill((0, 0, 0))  # clear background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Map volume to circle size
    size = int(min(max(volume_level * 500, 10), 300))  # scaled size
    color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))

    # Draw circles reacting to sound
    pygame.draw.circle(screen, color, (np.random.randint(0, WIDTH), np.random.randint(0, HEIGHT)), size)

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
stream.stop()
