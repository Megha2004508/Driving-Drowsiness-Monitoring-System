💤 Yawning and Drowsiness Detection System

This project detects yawning and eye closure (drowsiness) in real-time using a webcam feed. It utilizes MediaPipe Face Mesh for facial landmark tracking and pyttsx3 for voice alerts, helping prevent fatigue-related accidents or sleepiness during tasks like driving or studying.

🧠 Features

✅ Real-time face tracking using MediaPipe

😪 Yawn detection via Mouth Aspect Ratio (MAR)

💤 Drowsiness detection using Eye Aspect Ratio (EAR)

🔊 Voice alerts through pyttsx3 when drowsiness or yawning is detected

🖼️ Live display of MAR and EAR values on camera feed

⚙️ Tech Stack

Language: Python

Libraries:

opencv-python — for video capture and visualization

mediapipe — for face mesh landmark detection

scipy — for Euclidean distance computation

pyttsx3 — for offline text-to-speech alerts

📦 Installation

Clone the repository

git clone https://github.com/your-username/yawn-drowsiness-detector.git
cd yawn-drowsiness-detector


Install dependencies

pip install opencv-python mediapipe scipy pyttsx3


Run the program

python main.py

🎯 How It Works
| Parameter                     | Description                                                       | Threshold |
| ----------------------------- | ----------------------------------------------------------------- | --------- |
| **MAR (Mouth Aspect Ratio)**  | Measures vertical vs. horizontal mouth distance to detect yawning | `> 0.8`   |
| **EAR (Eye Aspect Ratio)**    | Measures openness of eyes to detect drowsiness                    | `< 0.2`   |
| **EYE_CLOSED_TIME_THRESHOLD** | Duration (seconds) eyes remain closed before alert                | `2s`      |


When the user yawns repeatedly or keeps eyes closed too long, a voice alert like

“Yawning detected. Please take a break!”
or
“Drowsiness detected. Please wake up!”
is played automatically.

🧩 Controls
Key	Action
v	Test voice alert manually
q	Quit the application
🖥️ Output Preview

Real-time video feed with overlaid MAR and EAR values

“Yawning Detected” or “Drowsiness Detected” text displayed on screen

Voice alerts triggered automatically

⚠️ Notes

Ensure good lighting and proper camera positioning for best results.

Thresholds (YAWN_THRESHOLD, EYE_THRESHOLD) can be fine-tuned based on face distance and lighting conditions.

🚀 Future Enhancements

Add GUI for settings (thresholds, alert volume, etc.)

Integrate with Raspberry Pi for portable fatigue detection

Include facial orientation tracking for distraction monitoring
