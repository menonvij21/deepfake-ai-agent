import os

# === CONFIGURATION ===
IMAGE_WATCH_FOLDER = "watched_images"
VIDEO_WATCH_FOLDER = "watched_videos"
MODEL_PATH = "outputs/checkpoints/mobilenetv2.pt"
LOG_PATH = "outputs/logs/agent_log.txt"

# Create log file if not exists
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
open(LOG_PATH, 'a').close()
