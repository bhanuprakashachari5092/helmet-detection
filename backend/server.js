const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased limit for parsing large image/video base-64 payloads

const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.get('/', (req, res) => {
  res.send('Helmet Detection API is running...');
});

// Endpoint for Python script to send detection events
app.post('/api/detection', (req, res) => {
  const { status, confidence, timestamp } = req.body;
  const detection = {
    id: Date.now().toString(),
    status,
    confidence,
    timestamp: timestamp || new Date().toLocaleTimeString()
  };
  
  io.emit('detection', detection);
  console.log(`[Detection] ${status} (${(confidence * 100).toFixed(1)}%)`);
  res.status(200).json({ message: 'Detection received' });
});

// File Upload endpoint for direct Express -> Python piping, or we can just proxy. 
// For simplicity, we just pass the base64 via socket.
app.post('/api/upload', (req, res) => {
   const { image } = req.body;
   if(image) {
      // Broadcast this large image frame to Python
      io.emit('process_this_frame', image);
      res.json({ success: true, message: "File processing started." });
   } else {
      res.status(400).json({ error: "Missing image data" });
   }
});

// Socket connection
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);

  // When frontend sends a live frame via camera
  socket.on('frontend_frame', (data) => {
    // Send it to Python AI process
    socket.broadcast.emit('process_this_frame', data);
  });

  // When Python returns the drawn frame
  socket.on('frame', (data) => {
    // Broadcast back to frontend to display the LIVE annotated footage
    socket.broadcast.emit('frame', data);
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected', socket.id);
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
