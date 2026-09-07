const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const multer = require('multer');

const app = express();
const server = http.createServer(app);

// Enable CORS for Localhost
app.use(cors({ origin: "*" }));
app.use(express.json());

// Serve the uploads folder as static files
// This allows the Frontend to access video/json directly like: http://localhost:3001/uploads/match.mp4
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

const io = new Server(server, { cors: { origin: "*" } });

const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) fs.mkdirSync(uploadDir);

// Multer Storage
const storage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, uploadDir),
    filename: (req, file, cb) => cb(null, 'match.mp4') // Always overwrite match.mp4
});
const upload = multer({ storage: storage });

let pythonProcess = null;

app.post('/upload', upload.single('video'), (req, res) => {
    const videoPath = path.join(uploadDir, 'match.mp4');
    const jsonPath = path.join(uploadDir, 'match_data.json');

    if (pythonProcess) pythonProcess.kill();

    console.log(">> Starting Local Analysis...");
    
    // Spawn Python Process
    pythonProcess = spawn('python', ['vision.py', videoPath, jsonPath]);

    pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        
        // Progress Updates
        if (output.includes('PROGRESS:')) {
            const progressVal = output.split('PROGRESS:')[1].trim();
            io.emit('analysis-progress', parseInt(progressVal));
        }
        
        // Completion
        if (output.includes('DONE')) {
            console.log(">> Analysis Complete. Notifying Frontend.");
            io.emit('analysis-complete', { jsonUrl: '/uploads/match_data.json' });
        }
    });

    pythonProcess.stderr.on('data', (data) => console.error(`ERR: ${data}`));
    res.json({ message: "Analysis Started" });
});

// Demo Route (Optional for testing)
app.get('/demo', (req, res) => {
    // Basic placeholder if you just want to test connection
    res.json({message: "Demo mode disabled for Local CPU version"}); 
});

server.listen(3001, () => console.log("Local Server running on http://localhost:3001"));