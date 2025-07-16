import os
import time
from ai_agent.predict_utils import predict_image, predict_video
from ai_agent.config_agent import IMAGE_WATCH_FOLDER, VIDEO_WATCH_FOLDER, LOG_PATH

def log_result(path, result):
    with open(LOG_PATH, "a") as f:
        f.write(f"[{time.ctime()}] {path} => {result}\n")

def watch_folder():
    seen = set()

    print("🔍 Watching folders for new files...")
    while True:
        # Watch images
        for fname in os.listdir(IMAGE_WATCH_FOLDER):
            path = os.path.join(IMAGE_WATCH_FOLDER, fname)
            if path not in seen and fname.lower().endswith((".jpg", ".jpeg", ".png")):
                print(f"🖼️ New image: {fname}")
                result = predict_image(path)
                print(f"🧠 Prediction: {result}")
                log_result(fname, result)
                seen.add(path)

        # Watch videos
        for fname in os.listdir(VIDEO_WATCH_FOLDER):
            path = os.path.join(VIDEO_WATCH_FOLDER, fname)
            if path not in seen and fname.lower().endswith((".mp4", ".avi", ".mov")):
                print(f"🎞️ New video: {fname}")
                result = predict_video(path)
                print(f"🧠 Prediction: {result}")
                log_result(fname, result)
                seen.add(path)

        time.sleep(3)

if __name__ == "__main__":
    watch_folder()
