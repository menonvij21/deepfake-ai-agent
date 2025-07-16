
# 🤖 Deepfake Detection AI Agent

A complete AI-powered tool to **detect deepfake images and videos**, featuring real-time predictions, model evaluation, folder-watching automation, and a sleek **Streamlit** interface.  
Built using **PyTorch**, **OpenCV**, and **MobileNetV2**, this project is optimized for GPU (CUDA) and designed for professionals and learners alike.

---

## 🚀 Features

- ✅ **Real-time Deepfake Detection** (image/video/webcam)
- 🧠 Trained with **MobileNetV2** using PyTorch (GPU enabled)
- 🔎 Auto-predict on new images/videos via folder watching
- 📊 Model Evaluation & Metrics Dashboard
- 🖼️ Streamlit Web App with upload and preview functionality
- 📁 Organized Logs and Output Management
- 💡 Designed as an AI Agent with automation features

---

## 🗂️ Project Structure

```

deepfake-ai-agent/
│
├── ai\_agent/                 # Agent automation logic
├── models/                  # Model loading and architecture
├── outputs/
│   ├── checkpoint/          # Trained MobileNetV2 model (.pt)
│   └── logs/                # Evaluation logs
├── watched\_images/          # Folder for auto-detection
├── utils/                   # Helper utilities
├── app.py                   # Streamlit UI
├── config.py                # Configs and paths
├── evaluate.py              # Evaluate model accuracy
├── image file prediction.py # Quick standalone image testing
├── main.py                  # Agent runner
├── model.py                 # Model structure
├── predict\_image.py         # Image prediction logic
├── predict\_video.py         # Video prediction logic
├── requirements.txt         # Dependencies
└── README.md                # This file

````

---

## ⚙️ Installation

### Step 1: Clone the Repo

```
git clone https://github.com/yourusername/deepfake-ai-agent.git
cd deepfake-ai-agent
````

### Step 2: Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate     # For Windows
source venv/bin/activate  # For Linux/Mac
```

### Step 3: Install Requirements

```
pip install -r requirements.txt
```

---

## 💻 Run the App

```
streamlit run app.py
```

* 🔍 For folder watching & auto-detection:

```
python main.py
```

* 📊 For evaluating the model:

```
python evaluate.py
```

---

## 🧠 Model Info

* **Architecture:** MobileNetV2
* **Trained On:** Deepfake image dataset
* **Accuracy:** \~93% (custom dataset)
* **Framework:** PyTorch (GPU-accelerated)

---

## 📷 Sample Use-Cases

* Social media content verification
* Deepfake threat monitoring
* Educational demos for AI security
* Lightweight CV model deployment

---

## 🧰 Tools & Libraries

* **PyTorch 2.2.0**
* **TorchVision 0.17.0**
* **OpenCV**
* **Streamlit**
* **Scikit-Learn**
* **Pillow**
* **Matplotlib**
* **NumPy**

---

## 📬 About Me

I'm **Vijayesh Menon**, a passionate AI learner focused on building **ethical, real-world AI solutions**.
This project is part of my journey to master **AI agents**, **Computer Vision**, and **Generative AI**.

Let’s connect:

* 💼 [LinkedIn](https://www.linkedin.com/in/vijayesh-menon-)
* 💌  [GMAIL] [menonvijayesh@gmail.com])
---

## 🙌 Contribute

Want to help improve this AI Agent?
Feel free to fork, enhance, or collaborate!
PRs and ideas are always welcome. 💡

```







