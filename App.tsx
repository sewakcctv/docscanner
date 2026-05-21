import React, { useRef, useState, useCallback, useEffect } from 'react';

type Point = { x: number; y: number };
type Mode  = 'camera' | 'crop' | 'preview';

// ─── Homography math ──────────────────────────────────────────────────────────

function solve8(A: number[][], b: number[]): number[] {
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let max = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[max][c])) max = r;
    [M[c], M[max]] = [M[max], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    x[i] = M[i][n];
    for (let j = i + 1; j < n; j++) x[i] -= M[i][j] * x[j];
    x[i] /= M[i][i];
  }
  return x;
}

function computeH(src: Point[], dst: Point[]): number[] {
  const A: number[][] = [], b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x: xs, y: ys } = src[i], { x: xd, y: yd } = dst[i];
    A.push([xs, ys, 1, 0, 0, 0, -xs * xd, -ys * xd]); b.push(xd);
    A.push([0, 0, 0, xs, ys, 1, -xs * yd, -ys * yd]);  b.push(yd);
  }
  return solve8(A, b);
}

function applyH(h: number[], x: number, y: number): Point {
  const d = h[6] * x + h[7] * y + 1;
  return { x: (h[0] * x + h[1] * y + h[2]) / d, y: (h[3] * x + h[4] * y + h[5]) / d };
}

// ─── Edge-based document corner detection ────────────────────────────────────

// Runs on a pre-downsampled canvas. Returns corners in that canvas's coords.
function findCornersInCanvas(canvas: HTMLCanvasElement): [Point, Point, Point, Point] | null {
  const { width: W, height: H } = canvas;
  const { data } = canvas.getContext('2d')!.getImageData(0, 0, W, H);

  const gray = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++)
    gray[i] = data[i * 4] * 0.299 + data[i * 4 + 1] * 0.587 + data[i * 4 + 2] * 0.114;

  // 3×3 Gaussian blur
  const blur = new Float32Array(W * H);
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++)
    blur[y * W + x] = (
      gray[(y-1)*W+(x-1)] + 2*gray[(y-1)*W+x] + gray[(y-1)*W+(x+1)] +
      2*gray[y*W+(x-1)]   + 4*gray[y*W+x]     + 2*gray[y*W+(x+1)] +
      gray[(y+1)*W+(x-1)] + 2*gray[(y+1)*W+x] + gray[(y+1)*W+(x+1)]
    ) / 16;

  // Sobel edge magnitude
  const mag = new Float32Array(W * H);
  let maxMag = 0;
  for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
    const gx =
      -blur[(y-1)*W+(x-1)] - 2*blur[y*W+(x-1)] - blur[(y+1)*W+(x-1)] +
       blur[(y-1)*W+(x+1)] + 2*blur[y*W+(x+1)] + blur[(y+1)*W+(x+1)];
    const gy =
      -blur[(y-1)*W+(x-1)] - 2*blur[(y-1)*W+x] - blur[(y-1)*W+(x+1)] +
       blur[(y+1)*W+(x-1)] + 2*blur[(y+1)*W+x] + blur[(y+1)*W+(x+1)];
    mag[y * W + x] = Math.sqrt(gx * gx + gy * gy);
    if (mag[y * W + x] > maxMag) maxMag = mag[y * W + x];
  }

  if (maxMag < 8) return null;
  const thr = maxMag * 0.22;

  // Four diagonal-extreme edge points
  // TL: min(x+y)  TR: max(x−y)  BR: max(x+y)  BL: min(x−y)
  let tlS = Infinity, trS = -Infinity, brS = -Infinity, blS = Infinity;
  let tl: Point | null = null, tr: Point | null = null;
  let br: Point | null = null, bl: Point | null = null;

  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (mag[y * W + x] < thr) continue;
    if (x + y < tlS) { tlS = x + y; tl = { x, y }; }
    if (x - y > trS) { trS = x - y; tr = { x, y }; }
    if (x + y > brS) { brS = x + y; br = { x, y }; }
    if (x - y < blS) { blS = x - y; bl = { x, y }; }
  }

  if (!tl || !tr || !br || !bl) return null;

  const cx = (tl.x + tr.x + br.x + bl.x) / 4;
  const cy = (tl.y + tr.y + br.y + bl.y) / 4;
  const valid =
    tl.x < cx && tl.y < cy && tr.x > cx && tr.y < cy &&
    br.x > cx && br.y > cy && bl.x < cx && bl.y > cy &&
    (tr.x - tl.x) / W > 0.18 && (bl.y - tl.y) / H > 0.18;

  return valid ? [tl, tr, br, bl] : null;
}

function downsample(
  src: HTMLCanvasElement | HTMLVideoElement,
  srcW: number, srcH: number, maxDim: number
): { canvas: HTMLCanvasElement; scale: number } {
  const scale = Math.min(1, maxDim / Math.max(srcW, srcH));
  const sw = Math.round(srcW * scale), sh = Math.round(srcH * scale);
  const c = document.createElement('canvas');
  c.width = sw; c.height = sh;
  (c.getContext('2d') as CanvasRenderingContext2D).drawImage(
    src as CanvasImageSource, 0, 0, sw, sh
  );
  return { canvas: c, scale };
}

function defaultCorners(W: number, H: number): [Point, Point, Point, Point] {
  const p = 0.09;
  return [
    { x: W * p,       y: H * p },
    { x: W * (1 - p), y: H * p },
    { x: W * (1 - p), y: H * (1 - p) },
    { x: W * p,       y: H * (1 - p) },
  ];
}

function detectCorners(
  cap: HTMLCanvasElement, W: number, H: number
): [Point, Point, Point, Point] {
  try {
    const { canvas: small, scale } = downsample(cap, W, H, 480);
    const result = findCornersInCanvas(small);
    if (!result) return defaultCorners(W, H);
    const inv = 1 / scale;
    return result.map(c => ({ x: c.x * inv, y: c.y * inv })) as [Point, Point, Point, Point];
  } catch { return defaultCorners(W, H); }
}

function detectCornersLive(
  video: HTMLVideoElement, W: number, H: number
): [Point, Point, Point, Point] | null {
  try {
    const { canvas: small, scale } = downsample(video, W, H, 320);
    const result = findCornersInCanvas(small);
    if (!result) return null;
    const inv = 1 / scale;
    return result.map(c => ({ x: c.x * inv, y: c.y * inv })) as [Point, Point, Point, Point];
  } catch { return null; }
}

// ─── Perspective warp ─────────────────────────────────────────────────────────

function perspectiveWarp(src: HTMLCanvasElement, corners: [Point, Point, Point, Point]): string {
  const [tl, tr, br, bl] = corners;
  const wTop = Math.hypot(tr.x - tl.x, tr.y - tl.y);
  const wBot = Math.hypot(br.x - bl.x, br.y - bl.y);
  const hL   = Math.hypot(bl.x - tl.x, bl.y - tl.y);
  const hR   = Math.hypot(br.x - tr.x, br.y - tr.y);

  const MAX = 2048;
  let outW = Math.round(Math.max(wTop, wBot));
  let outH = Math.round(Math.max(hL, hR));
  if (Math.max(outW, outH) > MAX) {
    const s = MAX / Math.max(outW, outH);
    outW = Math.round(outW * s); outH = Math.round(outH * s);
  }

  const h = computeH(
    [{ x: 0, y: 0 }, { x: outW, y: 0 }, { x: outW, y: outH }, { x: 0, y: outH }],
    corners
  );

  const { data: sData } = src.getContext('2d')!.getImageData(0, 0, src.width, src.height);
  const sW = src.width, sH = src.height;

  const out = document.createElement('canvas');
  out.width = outW; out.height = outH;
  const outCtx = out.getContext('2d')!;
  const outImg = outCtx.createImageData(outW, outH);
  const oData  = outImg.data;

  for (let yo = 0; yo < outH; yo++) for (let xo = 0; xo < outW; xo++) {
    const { x: xi, y: yi } = applyH(h, xo, yo);
    const x0 = Math.floor(xi), y0 = Math.floor(yi);
    const x1 = x0 + 1,         y1 = y0 + 1;
    const oi  = (yo * outW + xo) * 4;
    if (x0 < 0 || y0 < 0 || x1 >= sW || y1 >= sH) {
      oData[oi] = oData[oi+1] = oData[oi+2] = 255; oData[oi+3] = 255; continue;
    }
    const fx = xi - x0, fy = yi - y0;
    const w00 = (1-fx)*(1-fy), w10 = fx*(1-fy), w01 = (1-fx)*fy, w11 = fx*fy;
    const i00 = (y0*sW+x0)*4, i10 = (y0*sW+x1)*4;
    const i01 = (y1*sW+x0)*4, i11 = (y1*sW+x1)*4;
    oData[oi]   = sData[i00]*w00+sData[i10]*w10+sData[i01]*w01+sData[i11]*w11;
    oData[oi+1] = sData[i00+1]*w00+sData[i10+1]*w10+sData[i01+1]*w01+sData[i11+1]*w11;
    oData[oi+2] = sData[i00+2]*w00+sData[i10+2]*w10+sData[i01+2]*w01+sData[i11+2]*w11;
    oData[oi+3] = 255;
  }

  outCtx.putImageData(outImg, 0, 0);
  return out.toDataURL('image/jpeg', 0.95);
}

// ─── Zoom Mirror (shows magnified area around dragged corner) ─────────────────

const MIRROR_SIZE = 150;
const MIRROR_ZOOM = 3.5;

const ZoomMirror: React.FC<{
  srcCanvas: HTMLCanvasElement;
  cornerPos: Point;
  cornerIdx: number;
}> = ({ srcCanvas, cornerPos, cornerIdx }) => {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    const srcSz = MIRROR_SIZE / MIRROR_ZOOM;
    const sx = Math.max(0, Math.min(srcCanvas.width  - srcSz, cornerPos.x - srcSz / 2));
    const sy = Math.max(0, Math.min(srcCanvas.height - srcSz, cornerPos.y - srcSz / 2));

    ctx.clearRect(0, 0, MIRROR_SIZE, MIRROR_SIZE);
    ctx.drawImage(srcCanvas, sx, sy, srcSz, srcSz, 0, 0, MIRROR_SIZE, MIRROR_SIZE);

    // Crosshair centered on actual corner position
    const cx = Math.round((cornerPos.x - sx) * MIRROR_ZOOM);
    const cy = Math.round((cornerPos.y - sy) * MIRROR_ZOOM);
    ctx.strokeStyle = 'rgba(239,68,68,0.85)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, MIRROR_SIZE);
    ctx.moveTo(0, cy); ctx.lineTo(MIRROR_SIZE, cy);
    ctx.stroke();
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fill();
  }, [srcCanvas, cornerPos]);

  // Position mirror in the corner opposite the one being dragged
  const offset: React.CSSProperties[] = [
    { bottom: 12, right: 12 }, // TL → bottom-right
    { bottom: 12, left:  12 }, // TR → bottom-left
    { top:    12, left:  12 }, // BR → top-left
    { top:    12, right: 12 }, // BL → top-right
  ];

  return (
    <canvas
      ref={ref}
      width={MIRROR_SIZE}
      height={MIRROR_SIZE}
      style={{
        position: 'absolute',
        ...offset[cornerIdx],
        width:  MIRROR_SIZE,
        height: MIRROR_SIZE,
        borderRadius: 12,
        border: '3px solid rgba(255,255,255,0.92)',
        boxShadow: '0 6px 20px rgba(0,0,0,0.7)',
        pointerEvents: 'none',
        zIndex: 30,
        imageRendering: 'pixelated',
      }}
    />
  );
};

// ─── Crop Editor ──────────────────────────────────────────────────────────────

type CropEditorProps = {
  imageUrl: string;
  imgW: number;
  imgH: number;
  corners: [Point, Point, Point, Point];
  onChange: (c: [Point, Point, Point, Point]) => void;
  srcCanvas: HTMLCanvasElement;
};

const CropEditor: React.FC<CropEditorProps> = ({
  imageUrl, imgW, imgH, corners, onChange, srcCanvas,
}) => {
  const svgRef  = useRef<SVGSVGElement>(null);
  const dragRef = useRef<number | null>(null);
  const [zoom, setZoom] = useState<{ idx: number; pos: Point } | null>(null);

  const toSVGPt = (e: React.PointerEvent): Point => {
    const svg = svgRef.current!;
    const pt  = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const tp = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return { x: Math.max(0, Math.min(imgW, tp.x)), y: Math.max(0, Math.min(imgH, tp.y)) };
  };

  const onDown = (e: React.PointerEvent, i: number) => {
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = i;
    setZoom({ idx: i, pos: corners[i] });
  };

  const onMove = (e: React.PointerEvent) => {
    if (dragRef.current === null) return;
    e.preventDefault();
    const pt   = toSVGPt(e);
    const next = [...corners] as [Point, Point, Point, Point];
    next[dragRef.current] = pt;
    onChange(next);
    setZoom({ idx: dragRef.current, pos: pt });
  };

  const onUp = () => { dragRef.current = null; setZoom(null); };

  const [tl, tr, br, bl] = corners;
  const pts = `${tl.x},${tl.y} ${tr.x},${tr.y} ${br.x},${br.y} ${bl.x},${bl.y}`;
  const R   = Math.max(imgW, imgH) * 0.042;
  const SW  = R * 0.20;

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <img
        src={imageUrl}
        alt="Document"
        style={{ width: '100%', height: '100%', objectFit: 'contain', display: 'block', userSelect: 'none' }}
        draggable={false}
      />
      <svg
        ref={svgRef}
        viewBox={`0 0 ${imgW} ${imgH}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', touchAction: 'none' }}
        onPointerMove={onMove}
        onPointerUp={onUp}
        onPointerCancel={onUp}
      >
        {/* Darken outside selection */}
        <mask id="dm">
          <rect width={imgW} height={imgH} fill="white" />
          <polygon points={pts} fill="black" />
        </mask>
        <rect width={imgW} height={imgH} fill="rgba(0,0,0,0.5)" mask="url(#dm)" />

        {/* Document outline */}
        <polygon
          points={pts}
          fill="rgba(59,130,246,0.07)"
          stroke="#3b82f6"
          strokeWidth={SW}
          strokeLinejoin="round"
        />

        {/* Edge lines (easier to see exact shape) */}
        {([[tl,tr],[tr,br],[br,bl],[bl,tl]] as const).map(([a,b],i) => (
          <line key={i} x1={a.x} y1={a.y} x2={b.x} y2={b.y}
            stroke="rgba(255,255,255,0.5)" strokeWidth={SW * 0.5} />
        ))}

        {/* Corner handles — 0=TL 1=TR 2=BR 3=BL */}
        {corners.map((c, i) => (
          <g key={i} onPointerDown={e => onDown(e, i)} style={{ cursor: 'grab' }}>
            {/* Larger invisible hit area */}
            <circle cx={c.x} cy={c.y} r={R * 1.6} fill="transparent" />
            {/* Outer ring */}
            <circle cx={c.x} cy={c.y} r={R} fill="#2563eb" stroke="white" strokeWidth={SW} />
            {/* Crosshair */}
            <line x1={c.x - R*0.4} y1={c.y} x2={c.x + R*0.4} y2={c.y}
              stroke="white" strokeWidth={SW * 0.6} strokeLinecap="round" />
            <line x1={c.x} y1={c.y - R*0.4} x2={c.x} y2={c.y + R*0.4}
              stroke="white" strokeWidth={SW * 0.6} strokeLinecap="round" />
          </g>
        ))}
      </svg>

      {/* Zoom mirror when dragging */}
      {zoom && (
        <ZoomMirror
          srcCanvas={srcCanvas}
          cornerPos={zoom.pos}
          cornerIdx={zoom.idx}
        />
      )}
    </div>
  );
};

// ─── App ──────────────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const videoRef       = useRef<HTMLVideoElement>(null);
  const capturedCanvas = useRef<HTMLCanvasElement | null>(null);
  const streamRef      = useRef<MediaStream | null>(null);
  const stableCount    = useRef(0);

  const [mode,           setMode]           = useState<Mode>('camera');
  const [videoDims,      setVideoDims]      = useState({ w: 1920, h: 1080 });
  const [liveCorners,    setLiveCorners]    = useState<[Point,Point,Point,Point] | null>(null);
  const [capturedImage,  setCapturedImage]  = useState<string | null>(null);
  const [imgSize,        setImgSize]        = useState({ w: 0, h: 0 });
  const [corners,        setCorners]        = useState<[Point,Point,Point,Point] | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [processing,     setProcessing]     = useState(false);
  const [error,          setError]          = useState<string | null>(null);
  const [savedMsg,       setSavedMsg]       = useState('');

  // ── camera ─────────────────────────────────────────────────────────────────

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 3840 }, height: { ideal: 2160 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch {
      setError('Camera access denied.\nAllow camera permission in your browser settings, then refresh.');
    }
  }, []);

  useEffect(() => { startCamera(); return stopCamera; }, [startCamera, stopCamera]);

  // ── live document detection ────────────────────────────────────────────────

  useEffect(() => {
    if (mode !== 'camera') { setLiveCorners(null); return; }
    const id = setInterval(() => {
      const v = videoRef.current;
      if (!v || v.readyState < 2) return;
      const W = v.videoWidth, H = v.videoHeight;
      if (!W || !H) return;
      const found = detectCornersLive(v, W, H);
      if (found) {
        stableCount.current++;
        if (stableCount.current >= 2) setLiveCorners(found);
      } else {
        stableCount.current = 0;
        setLiveCorners(null);
      }
    }, 350);
    return () => clearInterval(id);
  }, [mode]);

  // ── capture ────────────────────────────────────────────────────────────────

  const capture = useCallback(() => {
    const v = videoRef.current;
    if (!v || v.readyState < 2) return;
    const W = v.videoWidth, H = v.videoHeight;
    const cap = document.createElement('canvas');
    cap.width = W; cap.height = H;
    cap.getContext('2d')!.drawImage(v, 0, 0);
    capturedCanvas.current = cap;

    // Use live-detected corners as starting point if available, else detect on full frame
    const detected = liveCorners
      ? liveCorners
      : detectCorners(cap, W, H);

    setCorners(detected);
    setImgSize({ w: W, h: H });
    setCapturedImage(cap.toDataURL('image/jpeg', 0.92));
    setMode('crop');
    stopCamera();
  }, [liveCorners, stopCamera]);

  const retake = useCallback(() => {
    setCapturedImage(null);
    setProcessedImage(null);
    setSavedMsg('');
    capturedCanvas.current = null;
    setMode('camera');
    startCamera();
  }, [startCamera]);

  // ── crop & warp ────────────────────────────────────────────────────────────

  const applyCrop = useCallback(async () => {
    if (!capturedCanvas.current || !corners) return;
    setProcessing(true);
    await new Promise(r => setTimeout(r, 30));
    try {
      setProcessedImage(perspectiveWarp(capturedCanvas.current, corners));
      setMode('preview');
    } finally {
      setProcessing(false);
    }
  }, [corners]);

  // ── download ───────────────────────────────────────────────────────────────

  const downloadImage = useCallback(() => {
    if (!processedImage) return;
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const a  = document.createElement('a');
    a.href     = processedImage;
    a.download = `scan_${ts}.jpg`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setSavedMsg('Saved!');
    setTimeout(() => setSavedMsg(''), 2000);
  }, [processedImage]);

  // ── render ─────────────────────────────────────────────────────────────────

  const docDetected = liveCorners !== null;
  const [vl, vt, vr, vb] = [8, 8, 92, 92]; // guide frame bounds (% of viewBox)
  const livePts = liveCorners
    ? liveCorners.map(c => `${(c.x / videoDims.w) * 100},${(c.y / videoDims.h) * 100}`).join(' ')
    : '';

  return (
    <div style={{
      height: '100dvh', display: 'flex', flexDirection: 'column',
      background: '#0f172a', color: '#f1f5f9',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      userSelect: 'none', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '10px 16px', fontSize: '15px', fontWeight: 700,
        letterSpacing: '0.08em', textAlign: 'center', color: '#94a3b8',
        flexShrink: 0, position: 'relative', display: 'flex',
        alignItems: 'center', justifyContent: 'center',
      }}>
        {mode !== 'camera' && (
          <button
            onClick={retake}
            style={{
              position: 'absolute', left: 12, background: 'none', border: 'none',
              color: '#64748b', fontSize: '13px', cursor: 'pointer', padding: '4px 8px',
            }}
          >
            ✕ Retake
          </button>
        )}
        DOC SCANNER
        {mode === 'crop' && (
          <span style={{ position: 'absolute', right: 12, fontSize: '11px', color: '#64748b', fontWeight: 400 }}>
            drag corners
          </span>
        )}
      </div>

      {/* Viewport */}
      <div style={{
        flex: 1, position: 'relative', overflow: 'hidden',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        background: '#000',
      }}>
        {error ? (
          <div style={{ padding: '32px', textAlign: 'center', color: '#f87171', fontSize: '15px', lineHeight: 1.65 }}>
            {error}
          </div>
        ) : mode === 'camera' ? (
          <>
            <video
              ref={videoRef}
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
              autoPlay playsInline muted
              onLoadedMetadata={() => {
                const v = videoRef.current;
                if (v) setVideoDims({ w: v.videoWidth, h: v.videoHeight });
              }}
            />
            {/* Live detection overlay — SVG viewBox matches video resolution */}
            <svg
              viewBox={`0 0 100 100`}
              preserveAspectRatio="none"
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
            >
              {docDetected && livePts ? (
                // Detected document outline — green, animated
                <polygon
                  points={livePts}
                  fill="rgba(34,197,94,0.12)"
                  stroke="#22c55e"
                  strokeWidth="0.6"
                  strokeLinejoin="round"
                  style={{ animation: 'pulse 1.4s ease-in-out infinite' }}
                />
              ) : (
                // Neutral guide frame when nothing detected
                <>
                  <rect x={vl} y={vt} width={vr-vl} height={vb-vt}
                    fill="none" stroke="rgba(255,255,255,0.18)"
                    strokeWidth="0.3" strokeDasharray="3 2" />
                  {([
                    { bx: vl, by: vt, dx:  1, dy:  1 },
                    { bx: vr, by: vt, dx: -1, dy:  1 },
                    { bx: vr, by: vb, dx: -1, dy: -1 },
                    { bx: vl, by: vb, dx:  1, dy: -1 },
                  ] as const).map(({ bx, by, dx, dy }, i) => (
                    <g key={i}>
                      <line x1={bx} y1={by} x2={bx + dx*6} y2={by} stroke="rgba(255,255,255,0.4)" strokeWidth="0.9" strokeLinecap="round" />
                      <line x1={bx} y1={by} x2={bx} y2={by + dy*6} stroke="rgba(255,255,255,0.4)" strokeWidth="0.9" strokeLinecap="round" />
                    </g>
                  ))}
                </>
              )}
            </svg>
            {/* Status hint */}
            <div style={{
              position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)',
              color: docDetected ? '#86efac' : 'rgba(255,255,255,0.45)',
              fontSize: '13px', pointerEvents: 'none', whiteSpace: 'nowrap',
              transition: 'color 0.3s',
            }}>
              {docDetected ? 'Document detected — tap to capture' : 'Point camera at a document'}
            </div>
          </>
        ) : mode === 'crop' && capturedImage && corners && imgSize.w > 0 ? (
          <>
            <CropEditor
              imageUrl={capturedImage}
              imgW={imgSize.w}
              imgH={imgSize.h}
              corners={corners}
              onChange={c => setCorners(c)}
              srcCanvas={capturedCanvas.current!}
            />
            {processing && (
              <div style={{
                position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.72)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                color: '#fff', fontSize: '16px', gap: 10,
              }}>
                <svg width="22" height="22" viewBox="0 0 22 22"
                  style={{ animation: 'spin 0.9s linear infinite' }}>
                  <circle cx="11" cy="11" r="9" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="2.5" />
                  <path d="M11 2 A9 9 0 0 1 20 11" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" />
                </svg>
                Processing…
              </div>
            )}
          </>
        ) : mode === 'preview' && processedImage ? (
          <>
            <img
              src={processedImage}
              alt="Scanned"
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            />
            {savedMsg && (
              <div style={{
                position: 'absolute', top: '50%', left: '50%',
                transform: 'translate(-50%,-50%)',
                background: 'rgba(0,0,0,0.78)', color: '#fff',
                padding: '12px 28px', borderRadius: '12px', fontSize: '15px',
                pointerEvents: 'none',
              }}>
                {savedMsg}
              </div>
            )}
          </>
        ) : null}
      </div>

      {/* Controls */}
      <div style={{
        padding: '18px 24px 38px', display: 'flex', alignItems: 'center',
        justifyContent: 'center', gap: 16, flexShrink: 0, background: '#0f172a',
      }}>
        {mode === 'camera' ? (
          <button
            onClick={capture}
            aria-label="Capture"
            style={{
              width: 72, height: 72, borderRadius: '50%',
              border: `4px solid ${docDetected ? '#22c55e' : '#f1f5f9'}`,
              background: 'transparent', cursor: 'pointer',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'border-color 0.3s',
            }}
          >
            <div style={{
              width: 54, height: 54, borderRadius: '50%',
              background: docDetected ? '#22c55e' : '#f1f5f9',
              transition: 'background 0.3s',
            }} />
          </button>
        ) : mode === 'crop' ? (
          <>
            <button
              onClick={retake}
              style={btnStyle('dark')}
            >Retake</button>
            <button
              onClick={applyCrop}
              disabled={processing}
              style={btnStyle('blue')}
            >{processing ? 'Processing…' : 'Crop & Apply'}</button>
          </>
        ) : (
          <>
            <button onClick={retake} style={btnStyle('dark')}>Retake</button>
            <button onClick={downloadImage} style={btnStyle('blue')}>
              ↓ Download
            </button>
          </>
        )}
      </div>

      <style>{`
        @keyframes spin   { to { transform: rotate(360deg); } }
        @keyframes pulse  { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }
      `}</style>
    </div>
  );
};

function btnStyle(variant: 'dark' | 'blue'): React.CSSProperties {
  return {
    padding: '13px 28px', borderRadius: 12, border: 'none',
    fontSize: 15, fontWeight: 600, cursor: 'pointer', letterSpacing: '0.02em',
    background: variant === 'blue' ? '#2563eb' : '#1e293b',
    color: variant === 'blue' ? '#fff' : '#cbd5e1',
    minWidth: variant === 'blue' ? 136 : undefined,
  };
}

export default App;
