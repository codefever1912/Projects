# PitchVision - AI Sports Analytics

PitchVision is a professional-grade sports analytics dashboard that uses Computer Vision to extract tactical insights from match footage. It visualizes player movement, team formations (Tactical Net), and ball trajectory in a broadcast-style interface.

## Vibe-Code Alert

This project has been mostly vibe coded using Google AI studio with Gemini 3 preview. Do take it with a grain of salt and will be continued to be done that way. This was a fun project to get me introduced to vibe coding, and advise anyone using this to keep that in mind


## Architecture: Hybrid Cloud-Local



This project uses a unique hybrid architecture to leverage high-performance GPUs without local hardware requirements:



1.  **Frontend (Local):** React + Vite application running on localhost. Handles the UI, video playback, and overlay rendering.

2.  **Middleware (Local):** Node.js Express server. Manages file uploads and coordinates communication.

3.  **AI Engine (Cloud):** Google Colab (Tesla T4 GPU). Runs the heavy YOLOv8x inference and OpenCV rendering, tunneling data back to the local machine via Ngrok.



## Prerequisites



*   Node.js (v16 or higher)

*   Python 3.9+ (For local fallback only)

*   Google Account (For Colab GPU access)

*   Ngrok Account (Free tier is sufficient)



## Installation



1.  **Clone the repository:**

&nbsp;   git clone <repository-url>

&nbsp;   cd PitchVision



2.  **Install Backend Dependencies:**

&nbsp;   npm install



3.  **Install Frontend Dependencies:**

&nbsp;   cd client

&nbsp;   npm install

&nbsp;   cd ..



## How to Run (The Setup)



Because this project offloads AI processing to the cloud, follow this specific startup sequence:



### Step 1: Start the AI Engine (Google Colab)

1.  Open the provided [Colab notebook](https://colab.research.google.com/drive/1363R4Tko-uPrElTyn47sf145VjO2CqPl?usp=sharing) 

2.  Set Runtime to **T4 GPU**.

3.  Insert your Ngrok Auth Token in the script.

4.  Run the cell.

5.  **Copy the Ngrok URL** generated in the output (e.g., `https://xxxx-xx-xx.ngrok-free.app`).



### Step 2: Configure Local Frontend

1.  Open `client/src/App.jsx`.

2.  Paste the Ngrok URL into the configuration constant:

&nbsp;   const SOCKET_URL = 'https://your-ngrok-url.ngrok-free.app';



### Step 3: Launch Local Application

1.  Open a terminal in the root folder.

2.  Start the Development Server:

&nbsp;   cd client

&nbsp;   npm run dev

3.  Open your browser to the local URL (usually `http://localhost:5173`).



**Important:** Before using the app, open your Ngrok URL in a separate browser tab and click "Visit Site" to bypass the security warning. This only needs to be done once per session.



## Features



*   **1080p Player Tracking:** Uses YOLOv8x for high-fidelity detection.

*   **Tactical Net:** Real-time visualization of passing options and team shape.

*   **Team Classification:** Automatic jersey color detection using K-Means clustering.

*   **Export Render:** Server-side rendering of analytics overlaid on the original footage for export.

*   **Physics Engine:** Heuristic speed calculation and smoothing.



## Directory Structure



*   `/client`: React Frontend code.

*   `/uploads`: Temporary storage for raw video files (Local).

*   `/outputs`: Storage for processed JSON telemetry and rendered videos.

*   `server.js`: Local Node.js orchestration.

*   `vision.py`: Local fallback for CV logic (Production uses Colab version).

