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
    A.push([xs, ys, 1, 0, 0, 0, -xs*xd, -ys*xd]); b.push(xd);
    A.push([0, 0, 0, xs, ys, 1, -xs*yd, -ys*yd]);  b.push(yd);
  }
  return solve8(A, b);
}

function applyH(h: number[], x: number, y: number): Point {
  const d = h[6]*x + h[7]*y + 1;
  return { x: (h[0]*x + h[1]*y + h[2]) / d, y: (h[3]*x + h[4]*y + h[5]) / d };
}

// ─── Hough-line document detection ───────────────────────────────────────────

type HLine = { theta: number; r: number; votes: number };

function gaussBlur3(g: Float32Array, W: number, H: number): Float32Array {
  const o = new Float32Array(W * H);
  for (let y = 1; y < H-1; y++) for (let x = 1; x < W-1; x++)
    o[y*W+x] = (
      g[(y-1)*W+(x-1)] + 2*g[(y-1)*W+x] + g[(y-1)*W+(x+1)] +
      2*g[y*W+(x-1)]   + 4*g[y*W+x]     + 2*g[y*W+(x+1)] +
      g[(y+1)*W+(x-1)] + 2*g[(y+1)*W+x] + g[(y+1)*W+(x+1)]
    ) / 16;
  return o;
}

function sobelMag(b: Float32Array, W: number, H: number): { mag: Float32Array; peak: number } {
  const mag = new Float32Array(W * H);
  let peak = 0;
  for (let y = 1; y < H-1; y++) for (let x = 1; x < W-1; x++) {
    const gx = -b[(y-1)*W+(x-1)] - 2*b[y*W+(x-1)] - b[(y+1)*W+(x-1)]
               +b[(y-1)*W+(x+1)] + 2*b[y*W+(x+1)] + b[(y+1)*W+(x+1)];
    const gy = -b[(y-1)*W+(x-1)] - 2*b[(y-1)*W+x] - b[(y-1)*W+(x+1)]
               +b[(y+1)*W+(x-1)] + 2*b[(y+1)*W+x] + b[(y+1)*W+(x+1)];
    mag[y*W+x] = Math.sqrt(gx*gx + gy*gy);
    if (mag[y*W+x] > peak) peak = mag[y*W+x];
  }
  return { mag, peak };
}

function houghLines(mag: Float32Array, peak: number, W: number, H: number): HLine[] {
  const NT = 180;
  const diag = Math.ceil(Math.hypot(W, H));
  const RS = 2 * diag + 1;
  const acc = new Int32Array(NT * RS);
  const cos = new Float32Array(NT), sin = new Float32Array(NT);
  for (let t = 0; t < NT; t++) {
    cos[t] = Math.cos(t * Math.PI / NT);
    sin[t] = Math.sin(t * Math.PI / NT);
  }
  const thr = peak * 0.14;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (mag[y*W+x] < thr) continue;
    for (let t = 0; t < NT; t++) {
      const r = Math.round(x * cos[t] + y * sin[t]) + diag;
      if (r >= 0 && r < RS) acc[t * RS + r]++;
    }
  }
  // Greedy peak extraction with NMS
  const cells: {t: number; r: number; v: number}[] = [];
  for (let t = 0; t < NT; t++) for (let r = 0; r < RS; r++) {
    const v = acc[t*RS+r]; if (v >= 4) cells.push({t, r, v});
  }
  cells.sort((a, b) => b.v - a.v);
  const used = new Uint8Array(NT * RS);
  const out: HLine[] = [];
  for (const {t, r, v} of cells) {
    if (out.length >= 25) break;
    if (used[t*RS+r]) continue;
    out.push({ theta: t * Math.PI / NT, r: r - diag, votes: v });
    for (let dt = -14; dt <= 14; dt++) for (let dr = -18; dr <= 18; dr++) {
      const nt = ((t+dt)%NT+NT)%NT, nr = r+dr;
      if (nr >= 0 && nr < RS) used[nt*RS+nr] = 1;
    }
  }
  return out;
}

function linesIntersect(l1: HLine, l2: HLine): Point | null {
  const c1 = Math.cos(l1.theta), s1 = Math.sin(l1.theta);
  const c2 = Math.cos(l2.theta), s2 = Math.sin(l2.theta);
  const det = c1*s2 - s1*c2;
  if (Math.abs(det) < 1e-7) return null;
  return { x: (l1.r*s2 - l2.r*s1)/det, y: (l2.r*c1 - l1.r*c2)/det };
}

function angDiff(a: number, b: number): number {
  let d = Math.abs(a - b) % Math.PI;
  return d > Math.PI/2 ? Math.PI - d : d;
}

function orderCorners(pts: Point[]): [Point, Point, Point, Point] {
  const s = [...pts].sort((a, b) => (a.x+a.y) - (b.x+b.y));
  const tl = s[0], br = s[3], rest = [s[1], s[2]];
  const tr = rest[0].x > rest[1].x ? rest[0] : rest[1];
  const bl = tr === rest[0] ? rest[1] : rest[0];
  return [tl, tr, br, bl];
}

function bestQuad(lines: HLine[], W: number, H: number): [Point, Point, Point, Point] | null {
  const n = Math.min(lines.length, 22);
  if (n < 4) return null;
  const PAR = 0.27, PERP_ERR = 0.38;
  const MARGIN = Math.max(W, H) * 0.55;
  const MIN_AREA = W * H * 0.035;

  let bestScore = -1, best: [Point, Point, Point, Point] | null = null;
  for (let i = 0; i < n-1; i++) {
    for (let j = i+1; j < n; j++) {
      if (angDiff(lines[i].theta, lines[j].theta) > PAR) continue;
      for (let k = 0; k < n-1; k++) {
        if (k===i||k===j) continue;
        if (Math.abs(angDiff(lines[i].theta, lines[k].theta) - Math.PI/2) > PERP_ERR) continue;
        for (let l = k+1; l < n; l++) {
          if (l===i||l===j) continue;
          if (angDiff(lines[k].theta, lines[l].theta) > PAR) continue;
          const pts = [
            linesIntersect(lines[i], lines[k]),
            linesIntersect(lines[i], lines[l]),
            linesIntersect(lines[j], lines[l]),
            linesIntersect(lines[j], lines[k]),
          ];
          if (pts.some(p=>!p)) continue;
          if (pts.some(p => p!.x < -MARGIN || p!.x > W+MARGIN || p!.y < -MARGIN || p!.y > H+MARGIN)) continue;
          const [a,b,c,d] = pts as Point[];
          const area = 0.5*Math.abs((a.x-c.x)*(b.y-d.y)-(b.x-d.x)*(a.y-c.y));
          if (area < MIN_AREA) continue;
          const score = lines[i].votes+lines[j].votes+lines[k].votes+lines[l].votes;
          if (score > bestScore) { bestScore = score; best = orderCorners([a,b,c,d]); }
        }
      }
    }
  }
  return best;
}

function detectInCanvas(small: HTMLCanvasElement): [Point, Point, Point, Point] | null {
  const { width: W, height: H } = small;
  const { data } = small.getContext('2d')!.getImageData(0, 0, W, H);
  const gray = new Float32Array(W*H);
  for (let i = 0; i < W*H; i++) gray[i] = data[i*4]*0.299+data[i*4+1]*0.587+data[i*4+2]*0.114;
  const blur = gaussBlur3(gray, W, H);
  const { mag, peak } = sobelMag(blur, W, H);
  if (peak < 6) return null;
  const lines = houghLines(mag, peak, W, H);
  return bestQuad(lines, W, H);
}

function mkSmall(src: HTMLCanvasElement | HTMLVideoElement, W: number, H: number, maxDim: number) {
  const s = maxDim / Math.max(W, H);
  const sw = Math.round(W*s), sh = Math.round(H*s);
  const c = document.createElement('canvas');
  c.width = sw; c.height = sh;
  (c.getContext('2d') as CanvasRenderingContext2D).drawImage(src as CanvasImageSource, 0, 0, sw, sh);
  return { c, s };
}

function defaultCorners(W: number, H: number): [Point, Point, Point, Point] {
  const p = 0.09;
  return [{x:W*p,y:H*p},{x:W*(1-p),y:H*p},{x:W*(1-p),y:H*(1-p)},{x:W*p,y:H*(1-p)}];
}

function detectCornersLive(video: HTMLVideoElement): [Point, Point, Point, Point] | null {
  const W = video.videoWidth, H = video.videoHeight;
  if (!W || !H) return null;
  try {
    const { c, s } = mkSmall(video, W, H, 320);
    const r = detectInCanvas(c);
    if (!r) return null;
    const inv = 1/s;
    return r.map(p => ({x:p.x*inv, y:p.y*inv})) as [Point, Point, Point, Point];
  } catch { return null; }
}

function detectCornersStatic(cap: HTMLCanvasElement): [Point, Point, Point, Point] {
  const W = cap.width, H = cap.height;
  try {
    const { c, s } = mkSmall(cap, W, H, 500);
    const r = detectInCanvas(c);
    if (!r) return defaultCorners(W, H);
    const inv = 1/s;
    return r.map(p => ({x:p.x*inv, y:p.y*inv})) as [Point, Point, Point, Point];
  } catch { return defaultCorners(W, H); }
}

// ─── Perspective warp + auto-enhance ─────────────────────────────────────────

function warpAndEnhance(src: HTMLCanvasElement, corners: [Point, Point, Point, Point]): string {
  const [tl, tr, br, bl] = corners;
  const wTop = Math.hypot(tr.x-tl.x, tr.y-tl.y), wBot = Math.hypot(br.x-bl.x, br.y-bl.y);
  const hL   = Math.hypot(bl.x-tl.x, bl.y-tl.y), hR   = Math.hypot(br.x-tr.x, br.y-tr.y);
  const MAX = 2048;
  let outW = Math.round(Math.max(wTop, wBot));
  let outH = Math.round(Math.max(hL, hR));
  if (Math.max(outW, outH) > MAX) {
    const s = MAX/Math.max(outW, outH); outW = Math.round(outW*s); outH = Math.round(outH*s);
  }

  const h = computeH(
    [{x:0,y:0},{x:outW,y:0},{x:outW,y:outH},{x:0,y:outH}], corners
  );
  const { data: sD } = src.getContext('2d')!.getImageData(0, 0, src.width, src.height);
  const sW = src.width, sH = src.height;

  const out = document.createElement('canvas');
  out.width = outW; out.height = outH;
  const outCtx = out.getContext('2d')!;
  const img = outCtx.createImageData(outW, outH);
  const oD  = img.data;

  for (let yo = 0; yo < outH; yo++) for (let xo = 0; xo < outW; xo++) {
    const { x: xi, y: yi } = applyH(h, xo, yo);
    const x0 = Math.floor(xi), y0 = Math.floor(yi), x1 = x0+1, y1 = y0+1;
    const oi = (yo*outW+xo)*4;
    if (x0<0||y0<0||x1>=sW||y1>=sH) { oD[oi]=oD[oi+1]=oD[oi+2]=255; oD[oi+3]=255; continue; }
    const fx=xi-x0, fy=yi-y0;
    const w00=(1-fx)*(1-fy), w10=fx*(1-fy), w01=(1-fx)*fy, w11=fx*fy;
    const i00=(y0*sW+x0)*4, i10=(y0*sW+x1)*4, i01=(y1*sW+x0)*4, i11=(y1*sW+x1)*4;
    oD[oi]  =sD[i00]*w00+sD[i10]*w10+sD[i01]*w01+sD[i11]*w11;
    oD[oi+1]=sD[i00+1]*w00+sD[i10+1]*w10+sD[i01+1]*w01+sD[i11+1]*w11;
    oD[oi+2]=sD[i00+2]*w00+sD[i10+2]*w10+sD[i01+2]*w01+sD[i11+2]*w11;
    oD[oi+3]=255;
  }
  outCtx.putImageData(img, 0, 0);

  // Auto-levels: stretch to 1%–99% percentiles per channel
  const full = outCtx.getImageData(0, 0, outW, outH);
  const d = full.data, N = outW * outH;
  for (let ch = 0; ch < 3; ch++) {
    const hist = new Int32Array(256);
    for (let i = ch; i < d.length; i += 4) hist[d[i]]++;
    const clip = N * 0.01;
    let lo = 0, hi = 255, s = 0;
    for (let v = 0; v < 256; v++) { s += hist[v]; if (s < clip) lo = v; else break; }
    s = 0;
    for (let v = 255; v >= 0; v--) { s += hist[v]; if (s < clip) hi = v; else break; }
    if (hi <= lo) continue;
    const sc = 255 / (hi - lo);
    for (let i = ch; i < d.length; i += 4)
      d[i] = Math.max(0, Math.min(255, Math.round((d[i] - lo) * sc)));
  }
  outCtx.putImageData(full, 0, 0);
  return out.toDataURL('image/jpeg', 0.95);
}

// ─── Zoom Mirror ──────────────────────────────────────────────────────────────

const SZ = 148, ZOOM = 4;

const ZoomMirror: React.FC<{ srcCanvas: HTMLCanvasElement; pos: Point; idx: number }> = ({
  srcCanvas, pos, idx,
}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    const srcSz = SZ / ZOOM;
    const sx = Math.max(0, Math.min(srcCanvas.width  - srcSz, pos.x - srcSz/2));
    const sy = Math.max(0, Math.min(srcCanvas.height - srcSz, pos.y - srcSz/2));
    ctx.clearRect(0, 0, SZ, SZ);
    ctx.drawImage(srcCanvas, sx, sy, srcSz, srcSz, 0, 0, SZ, SZ);
    const cx = Math.round((pos.x - sx) * ZOOM), cy = Math.round((pos.y - sy) * ZOOM);
    ctx.strokeStyle = 'rgba(239,68,68,0.9)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx, 0); ctx.lineTo(cx, SZ);
    ctx.moveTo(0, cy); ctx.lineTo(SZ, cy);
    ctx.stroke();
    ctx.fillStyle = '#ef4444';
    ctx.beginPath(); ctx.arc(cx, cy, 3, 0, Math.PI*2); ctx.fill();
  }, [srcCanvas, pos]);

  const pos4: React.CSSProperties[] = [
    {bottom:12,right:12},{bottom:12,left:12},{top:12,left:12},{top:12,right:12},
  ];
  return (
    <canvas ref={ref} width={SZ} height={SZ} style={{
      position:'absolute', ...pos4[idx], width:SZ, height:SZ,
      borderRadius:10, border:'3px solid rgba(255,255,255,0.92)',
      boxShadow:'0 6px 20px rgba(0,0,0,0.65)', pointerEvents:'none',
      zIndex:30, imageRendering:'pixelated',
    }} />
  );
};

// ─── Crop Editor ──────────────────────────────────────────────────────────────

const CropEditor: React.FC<{
  imageUrl: string; imgW: number; imgH: number;
  corners: [Point,Point,Point,Point];
  onChange: (c:[Point,Point,Point,Point]) => void;
  srcCanvas: HTMLCanvasElement;
}> = ({ imageUrl, imgW, imgH, corners, onChange, srcCanvas }) => {
  const svgRef  = useRef<SVGSVGElement>(null);
  const dragRef = useRef<number | null>(null);
  const [zoom, setZoom] = useState<{idx:number;pos:Point}|null>(null);

  const toPt = (e: React.PointerEvent): Point => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    const tp = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return {x:Math.max(0,Math.min(imgW,tp.x)), y:Math.max(0,Math.min(imgH,tp.y))};
  };

  const onDown = (e: React.PointerEvent, i: number) => {
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    dragRef.current = i; setZoom({idx:i, pos:corners[i]});
  };
  const onMove = (e: React.PointerEvent) => {
    if (dragRef.current===null) return; e.preventDefault();
    const pt = toPt(e);
    const next = [...corners] as [Point,Point,Point,Point];
    next[dragRef.current] = pt; onChange(next); setZoom({idx:dragRef.current, pos:pt});
  };
  const onUp = () => { dragRef.current=null; setZoom(null); };

  const [tl,tr,br,bl] = corners;
  const pts = `${tl.x},${tl.y} ${tr.x},${tr.y} ${br.x},${br.y} ${bl.x},${bl.y}`;
  const R = Math.max(imgW,imgH)*0.042, SW = R*0.18;

  return (
    <div style={{position:'relative',width:'100%',height:'100%'}}>
      <img src={imageUrl} alt="Document" draggable={false}
        style={{width:'100%',height:'100%',objectFit:'contain',display:'block',userSelect:'none'}} />
      <svg ref={svgRef} viewBox={`0 0 ${imgW} ${imgH}`} preserveAspectRatio="xMidYMid meet"
        style={{position:'absolute',inset:0,width:'100%',height:'100%',touchAction:'none'}}
        onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        <mask id="dm">
          <rect width={imgW} height={imgH} fill="white"/>
          <polygon points={pts} fill="black"/>
        </mask>
        <rect width={imgW} height={imgH} fill="rgba(0,0,0,0.48)" mask="url(#dm)"/>
        <polygon points={pts} fill="rgba(251,191,36,0.1)"
          stroke="#fbbf24" strokeWidth={SW} strokeLinejoin="round"/>
        {corners.map((c,i) => (
          <g key={i} onPointerDown={e=>onDown(e,i)} style={{cursor:'grab'}}>
            <circle cx={c.x} cy={c.y} r={R*1.6} fill="transparent"/>
            <circle cx={c.x} cy={c.y} r={R} fill="#fbbf24" stroke="white" strokeWidth={SW*0.9}/>
            <line x1={c.x-R*0.38} y1={c.y} x2={c.x+R*0.38} y2={c.y}
              stroke="white" strokeWidth={SW*0.6} strokeLinecap="round"/>
            <line x1={c.x} y1={c.y-R*0.38} x2={c.x} y2={c.y+R*0.38}
              stroke="white" strokeWidth={SW*0.6} strokeLinecap="round"/>
          </g>
        ))}
      </svg>
      {zoom && <ZoomMirror srcCanvas={srcCanvas} pos={zoom.pos} idx={zoom.idx}/>}
    </div>
  );
};

// ─── App ──────────────────────────────────────────────────────────────────────

const STABLE_FRAMES = 4; // × 350 ms = 1.4 s to auto-capture

const App: React.FC = () => {
  const videoRef       = useRef<HTMLVideoElement>(null);
  const capturedCanvas = useRef<HTMLCanvasElement | null>(null);
  const streamRef      = useRef<MediaStream | null>(null);
  const prevCornersRef = useRef<[Point,Point,Point,Point]|null>(null);
  const stableRef      = useRef(0);

  const [mode,           setMode]           = useState<Mode>('camera');
  const [vDims,          setVDims]          = useState({w:1920,h:1080});
  const [liveCorners,    setLiveCorners]    = useState<[Point,Point,Point,Point]|null>(null);
  const [stableCount,    setStableCount]    = useState(0);
  const [capturedImage,  setCapturedImage]  = useState<string|null>(null);
  const [imgSize,        setImgSize]        = useState({w:0,h:0});
  const [corners,        setCorners]        = useState<[Point,Point,Point,Point]|null>(null);
  const [processedImage, setProcessedImage] = useState<string|null>(null);
  const [processing,     setProcessing]     = useState(false);
  const [error,          setError]          = useState<string|null>(null);
  const [savedMsg,       setSavedMsg]       = useState('');

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video:{facingMode:{ideal:'environment'},width:{ideal:3840},height:{ideal:2160}},
        audio:false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch {
      setError('Camera access denied.\nAllow camera permission in your browser settings, then refresh.');
    }
  }, []);

  useEffect(() => { startCamera(); return stopCamera; }, [startCamera, stopCamera]);

  // ── capture (can be called manually or by auto-capture) ───────────────────

  const capture = useCallback(() => {
    const v = videoRef.current;
    if (!v || v.readyState < 2) return;
    const W = v.videoWidth, H = v.videoHeight;
    const cap = document.createElement('canvas');
    cap.width = W; cap.height = H;
    cap.getContext('2d')!.drawImage(v, 0, 0);
    capturedCanvas.current = cap;
    // Use live corners if we had a confident detection, else re-detect on full frame
    const det = liveCorners ?? detectCornersStatic(cap);
    setCorners(det);
    setImgSize({w:W, h:H});
    setCapturedImage(cap.toDataURL('image/jpeg', 0.92));
    setMode('crop');
    stopCamera();
  }, [liveCorners, stopCamera]);

  // ── live detection loop ───────────────────────────────────────────────────

  useEffect(() => {
    if (mode !== 'camera') { setLiveCorners(null); setStableCount(0); return; }
    const id = setInterval(() => {
      const v = videoRef.current;
      if (!v || v.readyState < 2) return;
      const found = detectCornersLive(v);
      if (found) {
        const prev = prevCornersRef.current;
        const stable = prev !== null && found.every((c, i) =>
          Math.hypot(c.x - prev[i].x, c.y - prev[i].y) < 22
        );
        if (stable) {
          stableRef.current = Math.min(stableRef.current + 1, STABLE_FRAMES);
        } else {
          stableRef.current = 0;
        }
        prevCornersRef.current = found;
        setLiveCorners(found);
        setStableCount(stableRef.current);
        if (stableRef.current >= STABLE_FRAMES) {
          stableRef.current = 0;
          // Schedule capture on next tick so state can flush
          setTimeout(() => {
            const vv = videoRef.current;
            if (vv && vv.readyState >= 2) capture();
          }, 0);
        }
      } else {
        stableRef.current = 0;
        prevCornersRef.current = null;
        setLiveCorners(null);
        setStableCount(0);
      }
    }, 350);
    return () => clearInterval(id);
  }, [mode, capture]);

  const retake = useCallback(() => {
    setCapturedImage(null); setProcessedImage(null); setSavedMsg('');
    capturedCanvas.current = null;
    setMode('camera'); startCamera();
  }, [startCamera]);

  const applyCrop = useCallback(async () => {
    if (!capturedCanvas.current || !corners) return;
    setProcessing(true);
    await new Promise(r => setTimeout(r, 30));
    try {
      setProcessedImage(warpAndEnhance(capturedCanvas.current, corners));
      setMode('preview');
    } finally { setProcessing(false); }
  }, [corners]);

  const downloadImage = useCallback(() => {
    if (!processedImage) return;
    const ts = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
    const a  = document.createElement('a');
    a.href = processedImage; a.download = `scan_${ts}.jpg`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setSavedMsg('Saved!'); setTimeout(() => setSavedMsg(''), 2000);
  }, [processedImage]);

  // ── styles ────────────────────────────────────────────────────────────────

  const detected = liveCorners !== null;
  const progress = stableCount / STABLE_FRAMES; // 0–1

  // SVG circle for progress ring around capture button (r=34, circumference≈213.6)
  const CIRC = 2 * Math.PI * 34;
  const dash  = CIRC * progress;

  return (
    <div style={{
      height:'100dvh', display:'flex', flexDirection:'column',
      background:'#0f172a', color:'#f1f5f9',
      fontFamily:'system-ui,-apple-system,sans-serif',
      userSelect:'none', overflow:'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding:'10px 16px', fontSize:'15px', fontWeight:700,
        letterSpacing:'0.08em', textAlign:'center', color:'#64748b',
        flexShrink:0, position:'relative',
        display:'flex', alignItems:'center', justifyContent:'center',
      }}>
        {mode !== 'camera' && (
          <button onClick={retake} style={{
            position:'absolute', left:12, background:'none', border:'none',
            color:'#64748b', fontSize:'13px', cursor:'pointer', padding:'4px 8px',
          }}>✕ Retake</button>
        )}
        DOC SCANNER
        {mode==='crop' && (
          <span style={{position:'absolute',right:12,fontSize:'11px',color:'#475569',fontWeight:400}}>
            drag corners
          </span>
        )}
      </div>

      {/* Viewport */}
      <div style={{
        flex:1, position:'relative', overflow:'hidden',
        display:'flex', alignItems:'center', justifyContent:'center',
        background:'#000',
      }}>
        {error ? (
          <div style={{padding:'32px',textAlign:'center',color:'#f87171',fontSize:'15px',lineHeight:1.65}}>
            {error}
          </div>
        ) : mode === 'camera' ? (
          <>
            <video
              ref={videoRef}
              style={{width:'100%',height:'100%',objectFit:'contain'}}
              autoPlay playsInline muted
              onLoadedMetadata={() => {
                const v = videoRef.current;
                if (v) setVDims({w:v.videoWidth, h:v.videoHeight});
              }}
            />
            {/* Live detection overlay — viewBox matches actual video frame dimensions */}
            <svg
              viewBox={`0 0 ${vDims.w} ${vDims.h}`}
              preserveAspectRatio="xMidYMid meet"
              style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none'}}
            >
              {liveCorners ? (
                <polygon
                  points={liveCorners.map(c=>`${c.x},${c.y}`).join(' ')}
                  fill="rgba(251,191,36,0.15)"
                  stroke="#fbbf24"
                  strokeWidth={Math.max(vDims.w,vDims.h)*0.004}
                  strokeLinejoin="round"
                />
              ) : (
                // Neutral corner brackets when nothing detected
                (() => {
                  const bx = vDims.w*0.08, by = vDims.h*0.10;
                  const bw = vDims.w*0.84, bh = vDims.h*0.80;
                  const arm = Math.min(vDims.w,vDims.h)*0.06;
                  const sw  = Math.max(vDims.w,vDims.h)*0.004;
                  return ([
                    [bx,by,1,1],[bx+bw,by,-1,1],[bx+bw,by+bh,-1,-1],[bx,by+bh,1,-1]
                  ] as [number,number,number,number][]).map(([x,y,dx,dy],i) => (
                    <g key={i}>
                      <line x1={x} y1={y} x2={x+dx*arm} y2={y} stroke="rgba(255,255,255,0.35)" strokeWidth={sw} strokeLinecap="round"/>
                      <line x1={x} y1={y} x2={x} y2={y+dy*arm} stroke="rgba(255,255,255,0.35)" strokeWidth={sw} strokeLinecap="round"/>
                    </g>
                  ));
                })()
              )}
            </svg>
          </>
        ) : mode==='crop' && capturedImage && corners && imgSize.w > 0 ? (
          <>
            <CropEditor
              imageUrl={capturedImage}
              imgW={imgSize.w} imgH={imgSize.h}
              corners={corners}
              onChange={c => setCorners(c)}
              srcCanvas={capturedCanvas.current!}
            />
            {processing && (
              <div style={{
                position:'absolute',inset:0,background:'rgba(0,0,0,0.72)',
                display:'flex',alignItems:'center',justifyContent:'center',
                color:'#fff',fontSize:'16px',gap:10,
              }}>
                <svg width="22" height="22" viewBox="0 0 22 22" style={{animation:'spin 0.9s linear infinite'}}>
                  <circle cx="11" cy="11" r="9" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="2.5"/>
                  <path d="M11 2 A9 9 0 0 1 20 11" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"/>
                </svg>
                Processing…
              </div>
            )}
          </>
        ) : mode==='preview' && processedImage ? (
          <>
            <img src={processedImage} alt="Scan"
              style={{width:'100%',height:'100%',objectFit:'contain'}}/>
            {savedMsg && (
              <div style={{
                position:'absolute',top:'50%',left:'50%',
                transform:'translate(-50%,-50%)',
                background:'rgba(0,0,0,0.78)',color:'#fff',
                padding:'12px 28px',borderRadius:12,fontSize:15,
                pointerEvents:'none',
              }}>{savedMsg}</div>
            )}
          </>
        ) : null}
      </div>

      {/* Controls */}
      <div style={{
        padding:'18px 24px 38px',display:'flex',alignItems:'center',
        justifyContent:'center',gap:16,flexShrink:0,background:'#0f172a',
      }}>
        {mode==='camera' ? (
          <div style={{position:'relative',width:80,height:80,display:'flex',alignItems:'center',justifyContent:'center'}}>
            {/* Animated progress ring */}
            <svg width="80" height="80" style={{position:'absolute',inset:0,transform:'rotate(-90deg)'}}>
              <circle cx="40" cy="40" r="34" fill="none"
                stroke={detected ? '#fbbf24' : 'rgba(255,255,255,0.15)'}
                strokeWidth="3" strokeDasharray={`${CIRC}`}
                strokeDashoffset={CIRC - dash}
                style={{transition:'stroke-dashoffset 0.35s ease, stroke 0.3s'}}
              />
            </svg>
            <button
              onClick={capture}
              aria-label="Capture"
              style={{
                width:68,height:68,borderRadius:'50%',
                border:`4px solid ${detected?'#fbbf24':'rgba(255,255,255,0.6)'}`,
                background:'transparent',cursor:'pointer',
                display:'flex',alignItems:'center',justifyContent:'center',
                transition:'border-color 0.3s',
              }}
            >
              <div style={{
                width:52,height:52,borderRadius:'50%',
                background:detected?'#fbbf24':'rgba(255,255,255,0.85)',
                transition:'background 0.3s',
              }}/>
            </button>
          </div>
        ) : mode==='crop' ? (
          <>
            <button onClick={retake}     style={btn('dark')}>Retake</button>
            <button onClick={applyCrop}  style={btn('gold')} disabled={processing}>
              {processing ? 'Processing…' : 'Crop & Apply'}
            </button>
          </>
        ) : (
          <>
            <button onClick={retake}         style={btn('dark')}>Retake</button>
            <button onClick={downloadImage}  style={btn('gold')}>↓ Download</button>
          </>
        )}
      </div>

      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
};

function btn(v: 'dark'|'gold'): React.CSSProperties {
  return {
    padding:'13px 28px', borderRadius:12, border:'none',
    fontSize:15, fontWeight:600, cursor:'pointer',
    background: v==='gold' ? '#d97706' : '#1e293b',
    color: v==='gold' ? '#fff' : '#94a3b8',
    minWidth: v==='gold' ? 136 : undefined,
  };
}

export default App;
