import React, { useState, useEffect, useRef } from 'react';
import io from 'socket.io-client';
import { 
  Play, Pause, Upload, Activity, Volume2, VolumeX, 
  Zap, Trophy, BarChart3, Download, Settings, RefreshCw, Layers
} from 'lucide-react';

// ⚠️ PASTE NGROK URL HERE
const SOCKET_URL = 'https://brainlessly-tracklaying-annie.ngrok-free.dev'; 

const App = () => {
  const [appState, setAppState] = useState('IDLE'); 
  const [connectionStatus, setConnectionStatus] = useState('DISCONNECTED');
  const [progress, setProgress] = useState(0);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [logs, setLogs] = useState(["> SYSTEM ONLINE"]); 
  
  const [matchData, setMatchData] = useState([]); 
  const [videoBlobUrl, setVideoBlobUrl] = useState(null);
  const [downloadUrl, setDownloadUrl] = useState(null);
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  const [teamConfig, setTeamConfig] = useState({
    0: { name: 'Home', color: '#3b82f6' }, 
    1: { name: 'Away', color: '#ef4444' }  
  });
  
  const currentFrameData = useRef({ p: [], b: null });
  const containerRef = useRef(null);
  const videoRef = useRef(null);
  const animationRef = useRef(null);
  const lastFrameIndex = useRef(0);
  const [overlayStyle, setOverlayStyle] = useState({});
  const fileInputRef = useRef(null);

  const addLog = (msg) => setLogs(prev => [...prev.slice(-2), msg]);
  const formatTime = (t) => { const m = Math.floor(t / 60); const s = Math.floor(t % 60); return `${m}:${s < 10 ? '0' : ''}${s}`; };

  const updateOverlayDimensions = () => {
    if (!videoRef.current || !containerRef.current) return;
    const { videoWidth: vw, videoHeight: vh } = videoRef.current;
    const { clientWidth: cw, clientHeight: ch } = containerRef.current;
    if (vw === 0) return;
    const videoRatio = vw / vh;
    const containerRatio = cw / ch;
    let finalW, finalH, finalLeft, finalTop;
    if (containerRatio > videoRatio) {
      finalH = ch; finalW = finalH * videoRatio; finalTop = 0; finalLeft = (cw - finalW) / 2;
    } else {
      finalW = cw; finalH = finalW / videoRatio; finalLeft = 0; finalTop = (ch - finalH) / 2;
    }
    setOverlayStyle({ width: `${finalW}px`, height: `${finalH}px`, left: `${finalLeft}px`, top: `${finalTop}px`, position: 'absolute' });
  };

  useEffect(() => { window.addEventListener('resize', updateOverlayDimensions); return () => window.removeEventListener('resize', updateOverlayDimensions); }, []);

  useEffect(() => {
    const socket = io(SOCKET_URL, { extraHeaders: { "ngrok-skip-browser-warning": "69420" }, reconnectionAttempts: 5 });
    socket.on('connect', () => setConnectionStatus('ONLINE'));
    socket.on('disconnect', () => setConnectionStatus('OFFLINE'));
    socket.on('analysis-progress', (pct) => { setAppState('PROCESSING'); setProgress(pct); });

    socket.on('analysis-complete', async (data) => {
      try {
        setDownloadUrl(`${SOCKET_URL}${data.downloadUrl}`);
        const jsonRes = await fetch(`${SOCKET_URL}${data.jsonUrl}`, { headers: { "ngrok-skip-browser-warning": "69420" } });
        const json = await jsonRes.json();
        setMatchData(json);

        setAppState('DOWNLOADING');
        const vidRes = await fetch(`${SOCKET_URL}${data.videoUrl}`, { headers: { "ngrok-skip-browser-warning": "69420" } });
        const reader = vidRes.body.getReader();
        const contentLength = +vidRes.headers.get('Content-Length');
        let receivedLength = 0;
        let chunks = [];
        while(true) {
            const {done, value} = await reader.read();
            if (done) break;
            chunks.push(value);
            receivedLength += value.length;
            if (contentLength) setDownloadProgress(Math.round((receivedLength / contentLength) * 100));
        }
        setVideoBlobUrl(URL.createObjectURL(new Blob(chunks)));
        setAppState('READY');
      } catch (err) { setAppState('IDLE'); }
    });
    return () => socket.close();
  }, []);

  const syncLoop = () => {
    if (videoRef.current && matchData.length > 0 && !videoRef.current.paused && videoRef.current.readyState >= 2) {
      const t = videoRef.current.currentTime + 0.15; 
      setCurrentTime(videoRef.current.currentTime);
      let startIndex = lastFrameIndex.current;
      if (matchData[startIndex] && matchData[startIndex].t > t) startIndex = 0;
      let frame = null;
      for (let i = startIndex; i < Math.min(matchData.length, startIndex + 60); i++) {
        if (Math.abs(matchData[i].t - t) < 0.1) { 
            frame = matchData[i];
            lastFrameIndex.current = i;
            break;
        }
      }
      if (frame) currentFrameData.current = frame; 
    }
    animationRef.current = requestAnimationFrame(syncLoop);
  };

  useEffect(() => { if (appState === 'READY') animationRef.current = requestAnimationFrame(syncLoop); return () => cancelAnimationFrame(animationRef.current); }, [appState]);

  const togglePlay = () => { if (videoRef.current.paused) { videoRef.current.play(); setIsPlaying(true); } else { videoRef.current.pause(); setIsPlaying(false); } };
  const handleSeek = (e) => { const t = parseFloat(e.target.value); videoRef.current.currentTime = t; setCurrentTime(t); lastFrameIndex.current = 0; };
  const handleUpload = async (e) => { const file = e.target.files[0]; if (!file) return; setAppState('PROCESSING'); const fd = new FormData(); fd.append('video', file); try { await fetch(`${SOCKET_URL}/upload`, { method: 'POST', body: fd }); } catch(err) { setAppState('IDLE'); } };
  const handleDemo = async () => { if (connectionStatus !== 'ONLINE') return alert("Connect GPU first"); setAppState('PROCESSING'); try { await fetch(`${SOCKET_URL}/demo`, { headers: { "ngrok-skip-browser-warning": "69420" } }); } catch(err) { setAppState('IDLE'); } };
  const downloadVideo = async () => { if(!downloadUrl) return; try { const res = await fetch(downloadUrl, { headers: { "ngrok-skip-browser-warning": "69420" } }); const blob = await res.blob(); const url = window.URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = "PitchVision_Analysis.mp4"; document.body.appendChild(a); a.click(); } catch (e) { alert("Download failed"); } };

  const getTacticalNet = (players) => {
      const lines = [];
      const teams = { 0: [], 1: [] };
      players.forEach(p => teams[p[5]].push(p));
      [0, 1].forEach(teamId => {
          const roster = teams[teamId];
          if (roster.length < 2) return;
          roster.forEach((p1, i) => {
              const distances = roster.map((p2, j) => {
                  if (i === j) return { dist: Infinity, p2 };
                  const dist = Math.sqrt(Math.pow(p1[1] - p2[1], 2) + Math.pow(p1[2] - p2[2], 2));
                  return { dist, p2 };
              });
              distances.sort((a, b) => a.dist - b.dist);
              const nearest = distances.slice(0, 2);
              nearest.forEach(n => {
                  if (n.dist < 0.3) {
                      lines.push({
                          x1: p1[1] * 100, y1: p1[2] * 100,
                          x2: n.p2[1] * 100, y2: n.p2[2] * 100,
                          team: teamId
                      });
                  }
              });
          });
      });
      return lines;
  };

  return (
    <div className="bg-slate-50 min-h-screen text-slate-900 font-sans selection:bg-blue-100">
      <nav className="bg-white border-b border-slate-200 fixed w-full top-0 z-50 h-16">
        <div className="max-w-[1400px] mx-auto px-6 h-full flex justify-between items-center relative">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center shadow-md shadow-blue-600/20"><Trophy size={16} className="text-white" /></div>
            <h1 className="text-lg font-bold tracking-tight text-slate-900">PitchVision</h1>
          </div>
          <div className="flex items-center gap-4">
             {appState === 'READY' && <button onClick={() => window.location.reload()} className="p-2 text-slate-400 hover:text-slate-600 transition" title="New Analysis"><RefreshCw size={18} /></button>}
             <div className="hidden sm:flex items-center gap-2 text-xs font-bold px-3 py-1.5 bg-slate-100 rounded-full border border-slate-200 text-slate-600"><div className={`w-2 h-2 rounded-full ${connectionStatus === 'ONLINE' ? 'bg-emerald-500' : 'bg-amber-500'}`}></div>{connectionStatus === 'ONLINE' ? 'Engine Ready' : 'Offline'}</div>
          </div>
        </div>
      </nav>

      <main className="max-w-[1400px] mx-auto p-6 pt-24 h-[calc(100vh)] flex flex-col gap-6">
        <div className="flex flex-col lg:flex-row gap-6 h-full min-h-[500px]">
            <div className="flex-1 bg-black rounded-2xl overflow-hidden shadow-2xl relative group flex flex-col justify-center border border-slate-900">
                <div ref={containerRef} className="relative w-full h-full">
                    {appState !== 'READY' && (
                        <div className="absolute inset-0 z-30 bg-slate-900 flex flex-col items-center justify-center">
                            {appState === 'IDLE' ? (
                                <div className="text-center space-y-6">
                                    <h2 className="text-2xl font-bold text-white">Upload Match Footage</h2>
                                    <div className="flex gap-4">
                                        <button onClick={handleDemo} className="px-6 py-3 bg-white text-slate-900 font-bold rounded-lg hover:bg-slate-100 transition">Load Demo</button>
                                        <label className="px-6 py-3 bg-blue-600 text-white font-bold rounded-lg cursor-pointer hover:bg-blue-500 transition shadow-lg shadow-blue-600/20">Upload .MP4<input type="file" accept="video/mp4" onChange={handleUpload} className="hidden" /></label>
                                    </div>
                                </div>
                            ) : (
                                <div className="w-64 space-y-4">
                                    <div className="flex justify-between text-xs font-bold text-indigo-400"><span>{appState === 'DOWNLOADING' ? 'BUFFERING' : 'PROCESSING'}</span><span>{appState === 'DOWNLOADING' ? downloadProgress : progress}%</span></div>
                                    <div className="h-1 bg-slate-800 rounded-full overflow-hidden"><div className="h-full bg-indigo-500 transition-all duration-300" style={{width: `${appState === 'DOWNLOADING' ? downloadProgress : progress}%`}}></div></div>
                                    <p className="text-center text-xs text-slate-500 font-mono">{logs[logs.length-1]}</p>
                                </div>
                            )}
                        </div>
                    )}

                    <video ref={videoRef} src={videoBlobUrl} className="w-full h-full object-contain" onLoadedMetadata={(e) => { updateOverlayDimensions(); setDuration(e.target.duration); }} onPlay={() => setIsPlaying(true)} onPause={() => setIsPlaying(false)} onClick={togglePlay} playsInline />

                    {appState === 'READY' && (
                        <div className="pointer-events-none overflow-hidden" style={overlayStyle}>
                            <svg className="absolute inset-0 w-full h-full">
                                {getTacticalNet(currentFrameData.current.p).map((line, i) => (
                                    <line key={i} x1={`${line.x1}%`} y1={`${line.y1}%`} x2={`${line.x2}%`} y2={`${line.y2}%`} 
                                        stroke={teamConfig[line.team].color} strokeWidth="2.5" strokeOpacity="0.4" />
                                ))}
                            </svg>
                            {currentFrameData.current.p.map(([id, x, y, w, h, team]) => (
                                <div key={id} className="absolute transition-all duration-100 ease-linear will-change-transform" style={{ left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%` }}>
                                    <div className="absolute bottom-0 left-0 w-full h-full opacity-90">
                                        <div className="absolute bottom-[2%] left-0 w-[20%] h-[10%] border-b-[3px] border-l-[3px]" style={{ borderColor: teamConfig[team].color }}></div>
                                        <div className="absolute bottom-[2%] right-0 w-[20%] h-[10%] border-b-[3px] border-r-[3px]" style={{ borderColor: teamConfig[team].color }}></div>
                                    </div>
                                    <div className="absolute -top-3 left-1/2 -translate-x-1/2 text-[8px] font-bold text-white px-1 bg-black/40 rounded-sm">{id}</div>
                                </div>
                            ))}
                            {currentFrameData.current.b && (
                                <div className="absolute" style={{ left: `${currentFrameData.current.b[0] * 100}%`, top: `${currentFrameData.current.b[1] * 100}%`, width: `${currentFrameData.current.b[2] * 100}%`, height: `${currentFrameData.current.b[3] * 100}%`, transition: 'all 0.1s linear' }}>
                                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[150%] h-[150%] border-2 border-yellow-400 rounded-full animate-ping opacity-60"></div>
                                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full bg-yellow-400 rounded-full shadow-md"></div>
                                </div>
                            )}
                        </div>
                    )}

                    {appState === 'READY' && (
                    <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/90 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200">
                        <input type="range" min="0" max={duration} step="0.1" value={currentTime} onChange={handleSeek} className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer mb-2" />
                        <div className="flex justify-between items-center text-white">
                            <button onClick={togglePlay} className="hover:text-blue-400">{isPlaying ? <Pause size={20} fill="currentColor"/> : <Play size={20} fill="currentColor"/>}</button>
                            <span className="text-xs font-mono">{formatTime(currentTime)} / {formatTime(duration)}</span>
                            <div className="flex gap-2">
                                <button onClick={downloadVideo} className="text-xs font-bold bg-white/10 px-2 py-1 rounded hover:bg-white/20">Export</button>
                                <button onClick={() => { videoRef.current.muted = !videoRef.current.muted; setIsMuted(!isMuted); }}>{isMuted ? <VolumeX size={18} /> : <Volume2 size={18} />}</button>
                            </div>
                        </div>
                    </div>
                    )}
                </div>
            </div>

            <div className="w-full lg:w-80 bg-white rounded-2xl shadow-xl border border-slate-200 p-6 flex flex-col gap-6">
                <div className="flex items-center gap-2 pb-4 border-b border-slate-100"><Settings size={18} className="text-slate-400" /><h3 className="font-bold text-slate-800">Tactical Setup</h3></div>
                <div className="space-y-4">
                    <div><label className="text-xs font-bold text-slate-400 uppercase">Home Team</label><div className="flex gap-2 mt-1"><input type="color" value={teamConfig[0].color} onChange={(e) => setTeamConfig({...teamConfig, 0: {...teamConfig[0], color: e.target.value}})} className="w-8 h-8 rounded cursor-pointer border-0" /><input type="text" value={teamConfig[0].name} onChange={(e) => setTeamConfig({...teamConfig, 0: {...teamConfig[0], name: e.target.value}})} className="flex-1 bg-slate-50 border border-slate-200 rounded px-2 text-sm font-medium" /></div></div>
                    <div><label className="text-xs font-bold text-slate-400 uppercase">Away Team</label><div className="flex gap-2 mt-1"><input type="color" value={teamConfig[1].color} onChange={(e) => setTeamConfig({...teamConfig, 1: {...teamConfig[1], color: e.target.value}})} className="w-8 h-8 rounded cursor-pointer border-0" /><input type="text" value={teamConfig[1].name} onChange={(e) => setTeamConfig({...teamConfig, 1: {...teamConfig[1], name: e.target.value}})} className="flex-1 bg-slate-50 border border-slate-200 rounded px-2 text-sm font-medium" /></div></div>
                </div>
                <div className="pt-4 border-t border-slate-100 flex-1"><div className="flex items-center gap-2 mb-4"><Layers size={18} className="text-slate-400" /><h3 className="font-bold text-slate-800">Formation Web</h3></div><p className="text-xs text-slate-500 leading-relaxed">The live overlay visualizes the <strong>Passing Network</strong> by connecting nearest teammates. This reveals defensive shape and attacking clusters in real-time.</p></div>
            </div>
        </div>
      </main>
    </div>
  );
};

export default App;