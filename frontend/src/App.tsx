import { useState, useEffect, useRef } from 'react';
import { io } from 'socket.io-client';
import { ShieldCheck, ShieldAlert, Camera, Activity, UploadCloud, Video, Bike, Shield, Zap, Maximize, RefreshCw } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';
const socket = io(BACKEND_URL);

interface Detection {
  id: string;
  timestamp: string;
  status: 'Helmet' | 'No Helmet';
  confidence: number;
}

// Official Animated Helmet Icon 
const ScanningHelmet = () => (
  <div className="relative p-3 bg-[#3b82f6]/10 rounded-2xl border border-[#3b82f6]/30 overflow-hidden isolate flex items-center justify-center shadow-[0_0_20px_rgba(59,130,246,0.2)]">
    <motion.div
      className="absolute top-0 left-0 right-0 h-1 bg-[#10b981] shadow-[0_0_10px_#10b981]"
      animate={{ y: [-10, 50, -10] }}
      transition={{ duration: 2.5, repeat: Infinity, ease: "easeInOut" }}
      style={{ zIndex: 5 }}
    />
    <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="relative z-10 drop-shadow-[0_0_8px_rgba(59,130,246,0.8)]">
      <path d="M2 18a1 1 0 0 0 1 1h18a1 1 0 0 0 1-1v-2a1 1 0 0 0-1-1H3a1 1 0 0 0-1 1v2z"></path>
      <path d="M12 2C7.58172 2 4 5.58172 4 10v5h16v-5c0-4.41828-3.58172-8-8-8z"></path>
    </svg>
  </div>
);

const HelmetCrashLoader = () => {
  const [loadingText, setLoadingText] = useState("Analyzing Uploaded Media...");

  useEffect(() => {
    const texts = [
      "Running Physics Engine...",
      "Detecting Impact Scenarios...",
      "Verifying Helmet Protection..."
    ];
    let i = 0;
    const interval = setInterval(() => {
      setLoadingText(texts[i % texts.length]);
      i++;
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="crash-loader-container">
      <div className="relative w-64 h-40 flex items-center justify-center mb-8">
        
        {/* Wall/Obstacle */}
        <motion.div 
          className="absolute right-8 w-2 h-20 bg-danger rounded-full"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.5 }}
        />

        {/* Bike riding fast and crashing */}
        <motion.div
           initial={{ x: -100, rotate: 0 }}
           animate={{ 
             x: [ -100, 40, 40 ],
             rotate: [ 0, 0, -15 ],
           }}
           transition={{ 
             duration: 1.5, 
             times: [0, 0.4, 1],
             ease: "easeOut",
             repeat: Infinity,
             repeatDelay: 3
           }}
           className="absolute z-10"
        >
          <Bike size={48} className="text-white" />
        </motion.div>

        {/* Impact Sparks */}
        <motion.div
          className="absolute right-10"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: [0, 1, 0], scale: [0.5, 1.5, 0.5] }}
          transition={{
            duration: 0.5, delay: 0.6, repeat: Infinity, repeatDelay: 4
          }}
        >
          <Zap size={32} className="text-yellow-400" />
        </motion.div>

        {/* Shield (Helmet) Appears to save */}
        <motion.div
          className="absolute right-4"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ 
             opacity: [0, 1, 1, 0],  scale: [0, 1.2, 1, 0] 
          }}
          transition={{ 
             duration: 2, delay: 0.8, repeat: Infinity, repeatDelay: 2.5
          }}
        >
          <Shield size={64} className="text-success drop-shadow-[0_0_15px_rgba(16,185,129,0.5)]" />
        </motion.div>

      </div>
      <h3 className="text-lg font-bold text-primary animate-pulse">{loadingText}</h3>
      <div className="w-48 h-1 bg-white/10 mt-4 rounded-full overflow-hidden">
        <motion.div 
          className="h-full bg-primary"
          initial={{ width: "0%" }}
          animate={{ width: "100%" }}
          transition={{ duration: 4.5, ease: "linear" }}
        />
      </div>
    </div>
  );
};

function App() {
  const [stream, setStream] = useState<string | null>(null);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [stats, setStats] = useState({ helmet: 28, noHelmet: 4 });
  const [isConnected, setIsConnected] = useState(false);
  
  const [activeTab, setActiveTab] = useState<'live' | 'upload'>('live');
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedResult, setUploadedResult] = useState<string | null>(null);

  // Hidden video and canvas refs to capture local WebCam
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const uploadVideoRef = useRef<HTMLVideoElement>(null);
  const [isVideoUpload, setIsVideoUpload] = useState(false);
  const [uploadedVideoURL, setUploadedVideoURL] = useState<string | null>(null);
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('environment');

  useEffect(() => {
    socket.on('connect', () => setIsConnected(true));
    socket.on('disconnect', () => setIsConnected(false));
    
    // Receive predicted frame from python backend
    socket.on('frame', (data) => {
      console.log("Socket: Received frame from AI");
      const base64Data = `data:image/jpeg;base64,${data}`;
      // Output differently based on mode
      if (activeTab === 'live') {
          setStream(base64Data);
      } else {
          // It's the upload processing result
          setUploadedResult(base64Data);
      }
    });

    socket.on('detection', (data: Detection) => {
      setDetections(prev => [data, ...prev].slice(0, 5));
      if (data.status === 'Helmet') {
        setStats(prev => ({ ...prev, helmet: prev.helmet + 1 }));
      } else {
        setStats(prev => ({ ...prev, noHelmet: prev.noHelmet + 1 }));
      }
    });

    return () => {
      socket.off('connect');
      socket.off('disconnect');
      socket.off('frame');
      socket.off('detection');
    };
  }, [activeTab]);

  // Activate Laptop/Mobile WebCam
  const startCamera = async (currentFacingMode = facingMode) => {
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      try {
        const localStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: currentFacingMode } 
        });
        if (videoRef.current) {
          videoRef.current.srcObject = localStream;
          setIsCameraActive(true);
        }
      } catch (err) {
        console.error("Camera access denied or unavailable", err);
        alert("Please permit camera access for Live Surveillance.");
      }
    }
  };

  const flipCamera = () => {
    stopCamera();
    const newMode = facingMode === 'user' ? 'environment' : 'user';
    setFacingMode(newMode);
    setTimeout(() => {
       startCamera(newMode);
    }, 200);
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      setIsCameraActive(false);
      setStream(null);
    }
  };

  // Loop to emit frames to backend
  useEffect(() => {
    let intervalId: any;
    if (activeTab === 'live' && isCameraActive && videoRef.current && canvasRef.current) {
      intervalId = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video && canvas && video.readyState === 4) {
          const context = canvas.getContext('2d');
          if (context) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            // Compress heavily for Socket transit to ensure fast real-time frame rate
            const imageData = canvas.toDataURL('image/jpeg', 0.5); 
            socket.emit('frontend_frame', imageData);
          }
        }
      }, 500); // 2 frames per second for stability. Drop to 200 for ~5fps.
    }
    return () => clearInterval(intervalId);
  }, [activeTab, isCameraActive]);

  // Loop to emit frames for Uploaded Video
  useEffect(() => {
    let intervalId: any;
    if (activeTab === 'upload' && isVideoUpload && uploadedVideoURL && uploadVideoRef.current && canvasRef.current && !isUploading) {
      intervalId = setInterval(() => {
        const video = uploadVideoRef.current;
        const canvas = canvasRef.current;
        if (video && canvas && video.readyState >= 2 && !video.paused && !video.ended) {
          const context = canvas.getContext('2d');
          if (context) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const imageData = canvas.toDataURL('image/jpeg', 0.5); 
            socket.emit('frontend_frame', imageData);
          }
        }
      }, 200); // Process 5 frames per second for smoother video playback
    }
    return () => clearInterval(intervalId);
  }, [activeTab, isVideoUpload, uploadedVideoURL, isUploading]);

  // Upload Logic handling
  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragging(true);
    } else if (e.type === "dragleave") {
      setIsDragging(false);
    }
  };

  const processFile = (file: File) => {
    if (!file) return;
    setUploadedResult(null);
    setIsUploading(true);
    setIsVideoUpload(file.type.startsWith('video/'));

    if (file.type.startsWith('video/')) {
       const url = URL.createObjectURL(file);
       setUploadedVideoURL(url);
       setTimeout(() => setIsUploading(false), 2500); // Wait for loader animation
    } else {
       const reader = new FileReader();
       reader.onload = (event) => {
           const base64Data = event.target?.result as string;
           
           // Emit multiple times during the loader phase to ensure AI picks it up
           const intervalId = setInterval(() => {
              console.log("Socket: Emitting frame to AI...");
              socket.emit('frontend_frame', base64Data);
           }, 500);

           setTimeout(() => {
              clearInterval(intervalId);
              setIsUploading(false);
           }, 3000); 
       };
       reader.readAsDataURL(file);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processFile(e.dataTransfer.files[0]);
    }
  };
  
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFile(e.target.files[0]);
    }
  };

  const toggleFullScreen = (id: string) => {
    const el = document.getElementById(id);
    if (!document.fullscreenElement) {
      el?.requestFullscreen().catch(err => {
        console.error(`Error attempting to enable fullscreen: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  };

  return (
    <>
      {/* Hidden local canvas for WebCam capture routing */}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <div className="premium-bg"></div>
      <div className="grid-overlay"></div>
      
      <div className="main-container">
        {/* Header Area */}
        <header className="flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left" style={{ marginBottom: '60px' }}>
          <div className="flex items-center gap-5 justify-center md:justify-start">
            <ScanningHelmet />
            <div>
              <h1 className="text-4xl font-extrabold tracking-tight gradient-blue mb-1">HELMET GUARD AI</h1>
              <p className="text-text-muted text-sm uppercase tracking-[3px] font-bold">Powered by OpenCV & YOLOv8n Automation</p>
            </div>
          </div>
          
          <div className="live-badge">
            <div className="dot"></div>
            {isConnected ? 'SYSTEM CONNECTED' : 'AWAITING CONNECTION'}
          </div>
        </header>

        {/* Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
          {[
            { label: 'Safety Index', value: `${Math.round((stats.helmet / (stats.helmet + stats.noHelmet)) * 100)}%`, color: 'text-primary', icon: Activity },
            { label: 'Verified Helmets', value: stats.helmet, color: 'text-success', icon: ShieldCheck },
            { label: 'Safety Violations', value: stats.noHelmet, color: 'text-danger', icon: ShieldAlert }
          ].map((item, idx) => (
            <motion.div 
              key={idx}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: idx * 0.1 }}
              className="card flex flex-col justify-between"
            >
              <div className="flex justify-between items-center mb-6">
                <span className="stat-title">{item.label}</span>
                <item.icon size={20} className={item.color} />
              </div>
              <div className={`stat-value ${item.color}`}>{item.value}</div>
            </motion.div>
          ))}
        </div>

        {/* Main Interface Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10" style={{ marginTop: '40px' }}>
          
          {/* Main Visuals Area */}
          <div className="lg:col-span-8 flex flex-col" style={{ gap: '24px' }}>
            
            {/* Context Tabs */}
            <div className="flex gap-4 p-2 bg-white/[0.03] border border-white/5 rounded-2xl w-fit backdrop-blur-sm" style={{ marginBottom: '10px' }}>
              <button 
                className={`tab-btn flex items-center gap-2 ${activeTab === 'live' ? 'active' : ''}`}
                onClick={() => { setActiveTab('live'); }}
              >
                <Video size={18} /> Live Surveillance
              </button>
              <button 
                className={`tab-btn flex items-center gap-2 ${activeTab === 'upload' ? 'active' : ''}`}
                onClick={() => { setActiveTab('upload'); stopCamera(); }}
              >
                <UploadCloud size={18} /> Deep Upload Analysis
              </button>
            </div>

            {/* Video / Upload Interface */}
            <div className="card !p-0 overflow-hidden relative min-h-[400px] flex">
              {activeTab === 'live' ? (
                <div id="video-container" className="video-frame w-full h-full flex-1 relative bg-black/50 group">
                  <video ref={videoRef} autoPlay playsInline muted className={`w-full h-full object-cover ${!isCameraActive ? 'hidden' : ''}`} />
                  
                  {stream && (
                    <img src={stream} alt="Surveillance Output" className="absolute inset-0 w-full h-full object-cover pointer-events-none" />
                  )}

                  {(!isCameraActive && !stream) && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-text-muted gap-4">
                      <Camera size={48} className="opacity-20 text-primary" />
                      <button onClick={() => startCamera()} className="mt-4 px-6 py-2 bg-[#3b82f6]/20 border border-[#3b82f6] text-[#3b82f6] rounded-xl font-bold animate-pulse hover:bg-[#3b82f6]/40">
                        Initialize WebCam Access
                      </button>
                    </div>
                  )}

                  {isCameraActive && (
                     <div className="absolute top-4 right-4 flex gap-2">
                       <button onClick={flipCamera} className="md:hidden bg-black/50 text-white border border-white/20 p-2 rounded-xl hover:bg-white/20 transition-colors backdrop-blur-sm" title="Flip Camera">
                         <RefreshCw size={18} />
                       </button>
                       <button onClick={() => toggleFullScreen('video-container')} className="bg-black/50 text-white border border-white/20 p-2 rounded-xl hover:bg-white/20 transition-colors backdrop-blur-sm" title="Full Screen">
                         <Maximize size={18} />
                       </button>
                       <button onClick={stopCamera} className="bg-danger/20 text-danger border border-danger/50 px-3 py-1 rounded-xl text-xs font-bold backdrop-blur-sm flex items-center">
                          Stop Camera
                       </button>
                     </div>
                  )}
                </div>
              ) : (
                <div id="upload-container" className="w-full relative flex flex-col min-h-[400px] bg-black/50 group">
                  {isUploading ? (
                    <div className="p-6 h-full flex flex-col items-center justify-center">
                       <HelmetCrashLoader />
                    </div>
                  ) : (uploadedResult || uploadedVideoURL) ? (
                    <div className="w-full h-full relative flex-1">
                        {/* Hidden original video player used for extracting frames */}
                        {isVideoUpload && uploadedVideoURL && (
                          <video 
                             ref={uploadVideoRef} 
                             src={uploadedVideoURL} 
                             autoPlay 
                             loop 
                             muted 
                             className="hidden" 
                          />
                        )}
                        
                        {/* The Annotated Output Image emitted by backend */}
                        {uploadedResult && (
                           <div className="absolute inset-0 w-full h-full">
                              <img src={uploadedResult} alt="Analysis Result" className="w-full h-full object-contain bg-black" />
                              <div className="absolute bottom-6 left-6 flex items-center gap-2 bg-success/20 text-success border border-success/40 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest backdrop-blur-md animate-pulse">
                                 <Zap size={12} /> AI Analysis Active
                              </div>
                           </div>
                        )}
                        
                        {/* Placeholder before first frame arrives for video */}
                        {isVideoUpload && !uploadedResult && (
                           <div className="absolute inset-0 flex flex-col items-center justify-center text-primary font-bold gap-4 bg-black/80">
                              <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin"></div>
                              <p className="animate-pulse">Initializing Neural Detection Stream...</p>
                           </div>
                        )}

                        <div className="absolute top-4 right-4 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                          <button onClick={() => toggleFullScreen('upload-container')} className="bg-black/50 text-white border border-white/20 p-2 rounded-xl hover:bg-white/20 transition-colors backdrop-blur-sm">
                            <Maximize size={18} />
                          </button>
                          <button onClick={() => {
                             setUploadedResult(null); 
                             setUploadedVideoURL(null); 
                             setIsVideoUpload(false);
                          }} className="bg-danger/20 text-danger border border-danger/50 px-4 py-2 rounded-xl text-xs font-bold hover:bg-danger/30 backdrop-blur-sm">Clear Output</button>
                        </div>
                    </div>
                  ) : (
                    <div className="p-6 h-full flex flex-col">
                      <div 
                        className={`upload-box w-full flex-1 ${isDragging ? 'drag-active' : ''}`}
                        onDragEnter={handleDrag}
                        onDragLeave={handleDrag}
                        onDragOver={handleDrag}
                        onDrop={handleDrop}
                        onClick={() => document.getElementById('file-upload')?.click()}
                      >
                        <motion.div animate={{ y: [0, -10, 0] }} transition={{ repeat: Infinity, duration: 3 }}>
                          <UploadCloud size={64} className="text-[#3b82f6] mb-4" />
                        </motion.div>
                        <h3 className="text-2xl font-extrabold tracking-tight">Drop Image or Video for Analysis</h3>
                        <p className="text-md text-text-muted text-center max-w-sm mt-2">Upload any media and our AI will process the physics, injury chance & helmet protection level.</p>
                        
                        <input id="file-upload" type="file" className="hidden" accept="image/*,video/*" onChange={handleFileSelect} />
                        
                        <button className="mt-6 px-6 py-3 bg-primary/20 text-primary border border-primary/30 rounded-xl font-bold hover:bg-primary/30 transition-colors pointer-events-none">
                          Browse Files
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Logs Panel */}
          <div className="lg:col-span-4 block">
            <div className="card h-full flex flex-col min-h-[500px]">
              <div className="flex justify-between items-center mb-8">
                <h3 className="text-sm font-bold text-[#3b82f6] tracking-widest uppercase flex items-center gap-2">
                  <Activity size={18} /> Event Logs
                </h3>
              </div>
              <div className="flex-1 scroll-area overflow-y-auto space-y-4 pr-2">

                <AnimatePresence initial={false}>
                  {detections.length === 0 ? (
                    <div className="h-full flex items-center justify-center opacity-30 text-sm italic py-20 flex-col gap-4 text-center">
                      <ShieldAlert size={32} />
                      <p>Neural logs will stream here <br/>upon successful connection</p>
                    </div>
                  ) : (
                    detections.map((det) => (
                      <motion.div
                        key={det.id}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="p-4 bg-white/[0.03] border border-white/5 rounded-2xl flex justify-between items-center hover:bg-white/[0.05] transition-colors"
                      >
                        <div>
                          <p className={`text-xs font-black uppercase tracking-wider ${det.status === 'Helmet' ? 'text-success' : 'text-danger'}`}>
                            {det.status}
                          </p>
                          <p className="text-[10px] text-text-muted mt-2">{det.timestamp}</p>
                        </div>
                        <div className="text-xs font-mono text-primary font-bold">{(det.confidence * 100).toFixed(1)}%</div>
                      </motion.div>
                    ))
                  )}
                </AnimatePresence>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;
