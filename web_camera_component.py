"""
Live Web Camera & AI Eye Tracking Component for Streamlit
Enables real-time 30 FPS continuous video streaming directly in the browser
via WebRTC / HTML5 getUserMedia and MediaPipe FaceMesh.
Works seamlessly on Streamlit Community Cloud and Localhost without any "Take Photo" snapshots.
"""

def get_live_web_camera_html(sensitivity=4):
    # sensitivity 2-15 maps to EAR threshold between 0.16 and 0.28
    ear_thresh = 0.16 + (sensitivity - 2) * 0.012

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: #0d1117;
  color: #e6edf3;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  overflow: hidden;
}}
#cam-card {{
  position: relative;
  width: 100%;
  height: 400px;
  background: #111622;
  border: 1px solid #30363d;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 8px 24px rgba(0,0,0,0.5);
}}
#webcam {{
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transform: scaleX(-1);
}}
#hud-canvas {{
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}}
.hud-bar {{
  position: absolute;
  left: 14px;
  right: 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  z-index: 10;
  pointer-events: none;
}}
.hud-top {{ top: 12px; }}
.hud-bottom {{ bottom: 12px; }}
.badge {{
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.5px;
  background: rgba(13, 17, 23, 0.85);
  backdrop-filter: blur(4px);
  border: 1px solid #30363d;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.badge-alert {{
  color: #10b981;
  border-color: rgba(16, 185, 129, 0.5);
  background: rgba(16, 185, 129, 0.15);
}}
.badge-drowsy {{
  color: #ef4444;
  border-color: rgba(239, 68, 68, 0.8);
  background: rgba(239, 68, 68, 0.3);
  animation: flashAlert 0.4s infinite alternate;
}}
@keyframes flashAlert {{
  0% {{ transform: scale(1); box-shadow: 0 0 10px rgba(239,68,68,0.5); }}
  100% {{ transform: scale(1.05); box-shadow: 0 0 25px rgba(239,68,68,0.9); }}
}}
.badge-info {{
  color: #94a3b8;
  font-size: 11px;
}}
#permission-modal {{
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(13, 17, 23, 0.95);
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  text-align: center;
  padding: 24px;
  z-index: 50;
}}
.modal-title {{ font-size: 18px; font-weight: 700; margin-bottom: 8px; color: #60a5fa; }}
.modal-desc {{ font-size: 13px; color: #94a3b8; max-width: 380px; line-height: 1.5; }}
.spinner {{
  width: 40px;
  height: 40px;
  border: 3px solid rgba(96, 165, 250, 0.2);
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 1s infinite linear;
  margin-bottom: 16px;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js" crossorigin="anonymous"></script>
</head>
<body>
<div id="cam-card">
  <video id="webcam" autoplay playsinline muted></video>
  <canvas id="hud-canvas"></canvas>

  <div class="hud-bar hud-top">
    <div id="status-badge" class="badge badge-alert">
      <span>●</span> <span id="status-text">INITIALIZING CAMERA...</span>
    </div>
    <div id="ear-badge" class="badge badge-info">
      <span id="ear-text">EAR: -- | OPEN: --%</span>
    </div>
  </div>

  <div class="hud-bar hud-bottom">
    <div class="badge badge-info">
      <span>🎯 ADAS REAL-TIME VISION</span>
    </div>
    <div id="drowsy-counter" class="badge badge-info">
      <span>DROWSY EVENTS: 0</span>
    </div>
  </div>

  <div id="permission-modal">
    <div class="spinner"></div>
    <div class="modal-title">Connecting to Laptop Webcam</div>
    <div class="modal-desc">
      Live real-time driver monitoring active.<br>
      Please allow camera access in your browser prompt.
    </div>
  </div>
</div>

<script>
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('hud-canvas');
const ctx = canvasEl.getContext('2d');
const modalEl = document.getElementById('permission-modal');
const statusBadge = document.getElementById('status-badge');
const statusText = document.getElementById('status-text');
const earText = document.getElementById('ear-text');
const drowsyCounterEl = document.getElementById('drowsy-counter');

const EAR_THRESHOLD = {ear_thresh:.3f};
let closedFrames = 0;
let isDrowsy = false;
let drowsyCount = 0;
let audioCtx = null;
let lastBeepTime = 0;

function playAlertTone() {{
  try {{
    const now = Date.now();
    if (now - lastBeepTime < 600) return;
    lastBeepTime = now;
    if (!audioCtx) {{
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }}
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(850, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.25, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.35);
  }} catch (e) {{}}
}}

function calcEAR(top, bottom, inner, outer) {{
  const v = Math.hypot(top.x - bottom.x, top.y - bottom.y);
  const h = Math.hypot(inner.x - outer.x, inner.y - outer.y);
  return h > 0 ? (v / (2.0 * h)) : 0;
}}

function drawHUD(w, h, faceBox, leftEye, rightEye, earVal, drowsyState) {{
  ctx.clearRect(0, 0, w, h);

  // Subtle ADAS scanlines
  ctx.strokeStyle = drowsyState ? 'rgba(239, 68, 68, 0.15)' : 'rgba(59, 130, 246, 0.1)';
  ctx.lineWidth = 1;
  const time = Date.now() * 0.002;
  const scanY = (Math.sin(time) * 0.5 + 0.5) * h;
  ctx.beginPath();
  ctx.moveTo(0, scanY);
  ctx.lineTo(w, scanY);
  ctx.stroke();

  if (!faceBox) return;

  const strokeColor = drowsyState ? '#ef4444' : '#10b981';
  const eyeColor = drowsyState ? '#ef4444' : '#06b6d4';

  // Face bounding brackets [ ]
  const fx = (1 - faceBox.xMax) * w; // mirrored
  const fw = (faceBox.xMax - faceBox.xMin) * w;
  const fy = faceBox.yMin * h;
  const fh = (faceBox.yMax - faceBox.yMin) * h;
  const cornerLen = Math.min(24, fw * 0.2);

  ctx.strokeStyle = strokeColor;
  ctx.lineWidth = 2.5;

  // Top-left
  ctx.beginPath();
  ctx.moveTo(fx, fy + cornerLen); ctx.lineTo(fx, fy); ctx.lineTo(fx + cornerLen, fy); ctx.stroke();
  // Top-right
  ctx.beginPath();
  ctx.moveTo(fx + fw - cornerLen, fy); ctx.lineTo(fx + fw, fy); ctx.lineTo(fx + fw, fy + cornerLen); ctx.stroke();
  // Bottom-left
  ctx.beginPath();
  ctx.moveTo(fx, fy + fh - cornerLen); ctx.lineTo(fx, fy + fh); ctx.lineTo(fx + cornerLen, fy + fh); ctx.stroke();
  // Bottom-right
  ctx.beginPath();
  ctx.moveTo(fx + fw - cornerLen, fy + fh); ctx.lineTo(fx + fw, fy + fh); ctx.lineTo(fx + fw, fy + fh - cornerLen); ctx.stroke();

  // Eye tracking boxes (mirrored)
  [leftEye, rightEye].forEach(eye => {{
    if (!eye) return;
    const ex = (1 - eye.xMax) * w;
    const ew = (eye.xMax - eye.xMin) * w;
    const ey = eye.yMin * h;
    const eh = (eye.yMax - eye.yMin) * h;

    ctx.strokeStyle = eyeColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(ex, ey, ew, eh);

    // Center dot
    ctx.fillStyle = eyeColor;
    ctx.beginPath();
    ctx.arc(ex + ew/2, ey + eh/2, 3, 0, Math.PI * 2);
    ctx.fill();
  }});
}}

function onResults(results) {{
  const w = canvasEl.width;
  const h = canvasEl.height;

  if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {{
    const lm = results.multiFaceLandmarks[0];

    // Left Eye Landmarks: top=386, bottom=374, inner=362, outer=263
    const earL = calcEAR(lm[386], lm[374], lm[362], lm[263]);
    // Right Eye Landmarks: top=159, bottom=145, inner=133, outer=33
    const earR = calcEAR(lm[159], lm[145], lm[133], lm[33]);
    const avgEAR = (earL + earR) / 2.0;

    // Openness percentage
    const openPct = Math.min(100, Math.max(0, Math.round(((avgEAR - 0.12) / 0.16) * 100)));

    // Bounding box calculations
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (let p of lm) {{
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }}
    const faceBox = {{ xMin: minX, xMax: maxX, yMin: minY, yMax: maxY }};

    const leftIndices = [362, 263, 386, 374, 385, 380];
    let lMinX = 1, lMaxX = 0, lMinY = 1, lMaxY = 0;
    leftIndices.forEach(idx => {{
      const p = lm[idx];
      if (p.x < lMinX) lMinX = p.x; if (p.x > lMaxX) lMaxX = p.x;
      if (p.y < lMinY) lMinY = p.y; if (p.y > lMaxY) lMaxY = p.y;
    }});
    const leftEye = {{ xMin: lMinX - 0.015, xMax: lMaxX + 0.015, yMin: lMinY - 0.015, yMax: lMaxY + 0.015 }};

    const rightIndices = [33, 133, 159, 145, 158, 153];
    let rMinX = 1, rMaxX = 0, rMinY = 1, rMaxY = 0;
    rightIndices.forEach(idx => {{
      const p = lm[idx];
      if (p.x < rMinX) rMinX = p.x; if (p.x > rMaxX) rMaxX = p.x;
      if (p.y < rMinY) rMinY = p.y; if (p.y > rMaxY) rMaxY = p.y;
    }});
    const rightEye = {{ xMin: rMinX - 0.015, xMax: rMaxX + 0.015, yMin: rMinY - 0.015, yMax: rMaxY + 0.015 }};

    // Drowsiness logic: EAR below threshold for > 8 frames (~300-400ms)
    if (avgEAR < EAR_THRESHOLD) {{
      closedFrames++;
      if (closedFrames >= 8) {{
        if (!isDrowsy) {{
          drowsyCount++;
          drowsyCounterEl.innerText = `DROWSY EVENTS: ${{drowsyCount}}`;
        }}
        isDrowsy = true;
        playAlertTone();
      }}
    }} else {{
      closedFrames = 0;
      isDrowsy = false;
    }}

    if (isDrowsy) {{
      statusBadge.className = 'badge badge-drowsy';
      statusText.innerText = '🚨 STATUS: DROWSY (EYES CLOSED)';
      earText.innerText = `EAR: ${{avgEAR.toFixed(2)}} | DROWSY (${{openPct}}%)`;
    }} else {{
      statusBadge.className = 'badge badge-alert';
      statusText.innerText = '🟢 STATUS: ALERT (BOTH EYES OPEN)';
      earText.innerText = `EAR: ${{avgEAR.toFixed(2)}} | ALERT (${{openPct}}%)`;
    }}

    drawHUD(w, h, faceBox, leftEye, rightEye, avgEAR, isDrowsy);
  }} else {{
    // No face detected
    statusBadge.className = 'badge badge-drowsy';
    statusText.innerText = '⚠️ DRIVER NOT IN CAMERA VIEW';
    earText.innerText = 'EAR: -- | NO FACE';
    drawHUD(w, h, null, null, null, 0, true);
  }}
}}

// Initialize Camera and MediaPipe FaceMesh
async function initTracker() {{
  try {{
    const stream = await navigator.mediaDevices.getUserMedia({{
      video: {{ width: {{ ideal: 640 }}, height: {{ ideal: 480 }}, facingMode: 'user' }},
      audio: false
    }});
    videoEl.srcObject = stream;
    await videoEl.play();

    modalEl.style.display = 'none';
    canvasEl.width = videoEl.videoWidth || 640;
    canvasEl.height = videoEl.videoHeight || 480;

    const faceMesh = new FaceMesh({{
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${{file}}`
    }});
    faceMesh.setOptions({{
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    }});
    faceMesh.onResults(onResults);

    const camera = new Camera(videoEl, {{
      onFrame: async () => {{
        await faceMesh.send({{ image: videoEl }});
      }},
      width: 640,
      height: 480
    }});
    camera.start();
  }} catch (err) {{
    console.error('Camera/Tracker Error:', err);
    modalEl.innerHTML = `
      <div style="font-size:36px; margin-bottom:12px;">📷</div>
      <div class="modal-title" style="color:#ef4444;">Camera Access Required</div>
      <div class="modal-desc">
        Please allow camera access in your browser to enable live driver eye tracking.<br>
        Check the camera icon in your browser URL bar.
      </div>
    `;
  }}
}}

initTracker();
</script>
</body>
</html>"""
    return html
