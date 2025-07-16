# Deepfake Detection AI Agent

Hi there! 👋  
This is my personal project — a **Deepfake Detection AI Agent** built to detect manipulated content in **images** and **videos** using deep learning. Whether you're curious about AI ethics, digital forensics, or just looking for a solid end-to-end project built with PyTorch, this is something you'll want to check out!

## 🌟 What It Does

This AI agent can:
- 🔍 Detect deepfakes in both images and videos.
- 💻 Run in real-time with a clean web interface (built using Streamlit).
- ⚡ Use GPU acceleration for faster predictions (CUDA 12 + PyTorch).
- 📝 Log predictions with time, result, and file name.
- ✅ Handle both manual uploads and (soon) automatic folder watching.

I’ve trained the model using **MobileNetV2**, chosen for its lightweight architecture and great performance on real-world tasks. The goal was to make something practical, fast, and easy to use — even on mid-range systems.

---
## 🧰 Tech Used

- **Python 3.10**
- **PyTorch** (for model training + inference)
- **OpenCV** (for image/video processing)
- **Streamlit** (for interactive web UI)
- **CUDA 12** (for GPU support)

## 💻 Run It on Your Machine

Here’s how you can get it running locally:


# Step 1: Clone the repo
git clone https://github.com/your-username/deepfake-ai-agent.git
cd deepfake-ai-agent

# Step 2: (Optional but recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Step 3: Install all required libraries
pip install -r requirements.txt

# Step 4: Run the app
streamlit run app.py

#About Me
I'm Vijayesh Menon, a passionate AI learner focused on building ethical, real-world AI solutions.
This project is part of my journey to master AI agents, Computer Vision, and Generative AI.

THANK YOU.....





