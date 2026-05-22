import React, { useRef, useState, useCallback, useEffect } from 'react';

type Point = { x: number; y: number };
type Mode  = 'camera' | 'crop' | 'preview';

// ─── Homography math ──────────────────────────────────────────────────────────

function solve8(A: number[][], b: number[]): number[] {
  const n = 8, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let mx = c;
    for (let r = c+1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[mx][c])) mx = r;
    [M[c], M[mx]] = [M[mx], M[c]];
    for (let r = c+1; r < n; r++) { const f=M[r][c]/M[c][c]; for (let j=c;j<=n;j++) M[r][j]-=f*M[c][j]; }
  }
  const x = new Array(n).fill(0);
  for (let i=n-1;i>=0;i--) { x[i]=M[i][n]; for (let j=i+1;j<n;j++) x[i]-=M[i][j]*x[j]; x[i]/=M[i][i]; }
  return x;
}
function computeH(src: Point[], dst: Point[]): number[] {
  const A: number[][]=[], b: number[]=[];
  for (let i=0;i<4;i++) {
    const {x:xs,y:ys}=src[i],{x:xd,y:yd}=dst[i];
    A.push([xs,ys,1,0,0,0,-xs*xd,-ys*xd]);b.push(xd);
    A.push([0,0,0,xs,ys,1,-xs*yd,-ys*yd]);b.push(yd);
  }
  return solve8(A,b);
}
function applyH(h: number[], x: number, y: number): Point {
  const d=h[6]*x+h[7]*y+1;
  return {x:(h[0]*x+h[1]*y+h[2])/d,y:(h[3]*x+h[4]*y+h[5])/d};
}

// ─── Brightness-blob detection helpers ───────────────────────────────────────

function boxBlur(src: Uint8Array, W: number, H: number, r: number): Uint8Array {
  const tmp = new Float32Array(W * H);
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0, c = 0;
      for (let dx = -r; dx <= r; dx++) {
        const nx = x + dx;
        if (nx >= 0 && nx < W) { s += src[y*W+nx]; c++; }
      }
      tmp[y*W+x] = s / c;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let s = 0, c = 0;
      for (let dy = -r; dy <= r; dy++) {
        const ny = y + dy;
        if (ny >= 0 && ny < H) { s += tmp[ny*W+x]; c++; }
      }
      out[y*W+x] = Math.round(s / c);
    }
  }
  return out;
}

function otsu(gray: Uint8Array): number {
  const N = gray.length;
  const hist = new Int32Array(256);
  for (let i = 0; i < N; i++) hist[gray[i]]++;
  let sum = 0;
  for (let t = 0; t < 256; t++) sum += t * hist[t];
  let sumB = 0, wB = 0, maxVar = 0, T = 128;
  for (let t = 0; t < 256; t++) {
    wB += hist[t]; if (!wB) continue;
    const wF = N - wB; if (!wF) break;
    sumB += t * hist[t];
    const mB = sumB / wB, mF = (sum - sumB) / wF;
    const v = wB * wF * (mB - mF) ** 2;
    if (v > maxVar) { maxVar = v; T = t; }
  }
  return T;
}

function dilate(src: Uint8Array, W: number, H: number, n: number): Uint8Array {
  let cur = src;
  for (let it = 0; it < n; it++) {
    const nxt = new Uint8Array(W*H);
    for (let y = 1; y < H-1; y++) for (let x = 1; x < W-1; x++) {
      if (cur[y*W+x]||cur[(y-1)*W+x]||cur[(y+1)*W+x]||cur[y*W+x-1]||cur[y*W+x+1])
        nxt[y*W+x] = 1;
    }
    cur = nxt;
  }
  return cur;
}

function erode(src: Uint8Array, W: number, H: number, n: number): Uint8Array {
  let cur = src;
  for (let it = 0; it < n; it++) {
    const nxt = new Uint8Array(W*H);
    for (let y = 1; y < H-1; y++) for (let x = 1; x < W-1; x++) {
      if (cur[y*W+x] && cur[(y-1)*W+x] && cur[(y+1)*W+x] && cur[y*W+x-1] && cur[y*W+x+1])
        nxt[y*W+x] = 1;
    }
    cur = nxt;
  }
  return cur;
}

// ─── Convex hull + quad extraction ───────────────────────────────────────────

function ccross(O: Point, A: Point, B: Point) {
  return (A.x-O.x)*(B.y-O.y)-(A.y-O.y)*(B.x-O.x);
}
function convexHull(pts: Point[]): Point[] {
  if (pts.length<3) return pts;
  const s=[...pts].sort((a,b)=>a.x-b.x||a.y-b.y);
  const lo: Point[]=[], hi: Point[]=[];
  for (const p of s) { while (lo.length>=2&&ccross(lo[lo.length-2],lo[lo.length-1],p)<=0) lo.pop(); lo.push(p); }
  for (let i=s.length-1;i>=0;i--) { const p=s[i]; while (hi.length>=2&&ccross(hi[hi.length-2],hi[hi.length-1],p)<=0) hi.pop(); hi.push(p); }
  lo.pop(); hi.pop();
  return [...lo,...hi];
}

// 4 extreme points of convex hull along the two diagonals → document corners
function hullCorners(h: Point[]): [Point,Point,Point,Point] {
  let tlS=Infinity,trS=-Infinity,brS=-Infinity,blS=Infinity;
  let tl=h[0],tr=h[0],br=h[0],bl=h[0];
  for (const p of h) {
    if (p.x+p.y<tlS){tlS=p.x+p.y;tl=p;}
    if (p.x-p.y>trS){trS=p.x-p.y;tr=p;}
    if (p.x+p.y>brS){brS=p.x+p.y;br=p;}
    if (p.x-p.y<blS){blS=p.x-p.y;bl=p;}
  }
  return [tl,tr,br,bl];
}

// ─── Main document detection ──────────────────────────────────────────────────
// Strategy: heavy blur erases text/lines → document becomes uniform bright blob
// → Otsu threshold → largest bright connected component = document
// → convex hull of that blob → 4 diagonal extremes = corners

function detectInCanvas(small: HTMLCanvasElement): [Point,Point,Point,Point]|null {
  const W = small.width, H = small.height;
  const {data} = small.getContext('2d')!.getImageData(0, 0, W, H);

  // Grayscale
  const gray = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++)
    gray[i] = (data[i*4]*77 + data[i*4+1]*150 + data[i*4+2]*29) >> 8;

  // 3 passes of box blur (r=8) ≈ Gaussian σ≈14px on 320px image.
  // This completely erases text lines and table borders — the document
  // becomes a single solid bright blob, background stays dark.
  let b = boxBlur(gray, W, H, 8);
  b = boxBlur(b, W, H, 8);
  b = boxBlur(b, W, H, 8);

  // Otsu finds the threshold that best separates bright (doc) from dark (bg)
  const T = otsu(b);
  const bin = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) bin[i] = b[i] > T ? 1 : 0;

  // Tiny dilation only (2px) to bridge hairline gaps — NO erosion back.
  // Heavy closing (dilate+erode) was rounding corners by ~30px in original resolution.
  const closed = dilate(bin, W, H, 2);

  // BFS to find all connected components; pick the largest (= the document)
  const visited = new Uint8Array(W * H);
  const compOf  = new Int32Array(W * H);  // 0 = unassigned
  const DX = [-1, 1, 0, 0];
  const DY = [0, 0, -1, 1];
  let nextLabel = 1, bestLabel = 0, bestSize = 0;

  for (let start = 0; start < W * H; start++) {
    if (!closed[start] || visited[start]) continue;
    const label = nextLabel++;
    const q = [start];
    visited[start] = 1; compOf[start] = label;
    let size = 0;
    for (let qi = 0; qi < q.length; qi++) {
      const ci = q[qi]; size++;
      const cx = ci % W, cy = (ci / W) | 0;
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d], ny = cy + DY[d];
        if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
        const ni = ny*W+nx;
        if (closed[ni] && !visited[ni]) { visited[ni]=1; compOf[ni]=label; q.push(ni); }
      }
    }
    if (size > bestSize) { bestSize = size; bestLabel = label; }
  }

  if (bestSize < W * H * 0.05) return null;  // too small to be a document

  // Collect outermost pixels per row and column of the winning blob
  const pts: Point[] = [];
  for (let y = 0; y < H; y++) {
    let lo = -1, hi = -1;
    for (let x = 0; x < W; x++)
      if (compOf[y*W+x] === bestLabel) { if (lo < 0) lo = x; hi = x; }
    if (lo >= 0) { pts.push({x:lo, y}); if (hi !== lo) pts.push({x:hi, y}); }
  }
  for (let x = 0; x < W; x++) {
    let lo = -1, hi = -1;
    for (let y = 0; y < H; y++)
      if (compOf[y*W+x] === bestLabel) { if (lo < 0) lo = y; hi = y; }
    if (lo >= 0) { pts.push({x, y:lo}); if (hi !== lo) pts.push({x, y:hi}); }
  }

  if (pts.length < 8) return null;

  const hull = convexHull(pts);
  if (hull.length < 4) return null;

  const [tl, tr, br, bl] = hullCorners(hull);

  // Validate: each corner in its expected quadrant
  const mx = (tl.x+tr.x+br.x+bl.x)/4, my = (tl.y+tr.y+br.y+bl.y)/4;
  if (!(tl.x<=mx && tl.y<=my && tr.x>=mx && tr.y<=my && br.x>=mx && br.y>=my && bl.x<=mx && bl.y>=my))
    return null;

  // Validate area: 6%–92% of canvas
  const area = 0.5 * Math.abs((tl.x-br.x)*(tr.y-bl.y) - (tr.x-bl.x)*(tl.y-br.y));
  if (area < W*H*0.06 || area > W*H*0.92) return null;

  return [tl, tr, br, bl];
}

function mkSmall(
  src: HTMLCanvasElement|HTMLVideoElement,
  W: number, H: number, maxDim: number
): {c:HTMLCanvasElement; scale:number} {
  const scale = Math.min(1, maxDim/Math.max(W,H));
  const sw=Math.round(W*scale), sh=Math.round(H*scale);
  const c=document.createElement('canvas'); c.width=sw; c.height=sh;
  (c.getContext('2d') as CanvasRenderingContext2D).drawImage(src as CanvasImageSource,0,0,sw,sh);
  return {c, scale};
}

function defaultCorners(W: number, H: number): [Point,Point,Point,Point] {
  const p=0.1;
  return [{x:W*p,y:H*p},{x:W*(1-p),y:H*p},{x:W*(1-p),y:H*(1-p)},{x:W*p,y:H*(1-p)}];
}

function detectLive(video: HTMLVideoElement): [Point,Point,Point,Point]|null {
  const W=video.videoWidth, H=video.videoHeight;
  if (!W||!H) return null;
  try {
    const {c,scale}=mkSmall(video,W,H,480);
    const r=detectInCanvas(c);
    if (!r) return null;
    const inv=1/scale;
    return r.map(p=>({x:p.x*inv,y:p.y*inv})) as [Point,Point,Point,Point];
  } catch { return null; }
}

function detectStatic(cap: HTMLCanvasElement): [Point,Point,Point,Point] {
  const W=cap.width, H=cap.height;
  try {
    const {c,scale}=mkSmall(cap,W,H,800);
    const r=detectInCanvas(c);
    if (!r) return defaultCorners(W,H);
    const inv=1/scale;
    return r.map(p=>({x:p.x*inv,y:p.y*inv})) as [Point,Point,Point,Point];
  } catch { return defaultCorners(W,H); }
}

// ─── Perspective warp + auto-enhance ─────────────────────────────────────────

function warpAndEnhance(src: HTMLCanvasElement, corners: [Point,Point,Point,Point]): string {
  const [tl,tr,br,bl]=corners;
  const wT=Math.hypot(tr.x-tl.x,tr.y-tl.y), wB=Math.hypot(br.x-bl.x,br.y-bl.y);
  const hL=Math.hypot(bl.x-tl.x,bl.y-tl.y), hR=Math.hypot(br.x-tr.x,br.y-tr.y);
  const MAX=2048;
  let oW=Math.round(Math.max(wT,wB)), oH=Math.round(Math.max(hL,hR));
  if (Math.max(oW,oH)>MAX){const s=MAX/Math.max(oW,oH);oW=Math.round(oW*s);oH=Math.round(oH*s);}

  const h=computeH([{x:0,y:0},{x:oW,y:0},{x:oW,y:oH},{x:0,y:oH}],corners);
  const {data:sD}=src.getContext('2d')!.getImageData(0,0,src.width,src.height);
  const sW=src.width, sH=src.height;

  const out=document.createElement('canvas'); out.width=oW; out.height=oH;
  const ctx=out.getContext('2d')!;
  const img=ctx.createImageData(oW,oH); const oD=img.data;

  for (let yo=0;yo<oH;yo++) for (let xo=0;xo<oW;xo++) {
    const {x:xi,y:yi}=applyH(h,xo,yo);
    const x0=Math.floor(xi),y0=Math.floor(yi),x1=x0+1,y1=y0+1;
    const oi=(yo*oW+xo)*4;
    if (x0<0||y0<0||x1>=sW||y1>=sH){oD[oi]=oD[oi+1]=oD[oi+2]=255;oD[oi+3]=255;continue;}
    const fx=xi-x0,fy=yi-y0;
    const w00=(1-fx)*(1-fy),w10=fx*(1-fy),w01=(1-fx)*fy,w11=fx*fy;
    const i00=(y0*sW+x0)*4,i10=(y0*sW+x1)*4,i01=(y1*sW+x0)*4,i11=(y1*sW+x1)*4;
    oD[oi]  =sD[i00]*w00+sD[i10]*w10+sD[i01]*w01+sD[i11]*w11;
    oD[oi+1]=sD[i00+1]*w00+sD[i10+1]*w10+sD[i01+1]*w01+sD[i11+1]*w11;
    oD[oi+2]=sD[i00+2]*w00+sD[i10+2]*w10+sD[i01+2]*w01+sD[i11+2]*w11;
    oD[oi+3]=255;
  }
  ctx.putImageData(img,0,0);

  // Auto-levels: per-channel 1%–99% stretch
  const full=ctx.getImageData(0,0,oW,oH); const d=full.data; const N=oW*oH;
  for (let ch=0;ch<3;ch++) {
    const hist=new Int32Array(256);
    for (let i=ch;i<d.length;i+=4) hist[d[i]]++;
    const clip=N*0.01;
    let lo=0,hi=255,s=0;
    for (let v=0;v<256;v++){s+=hist[v];if(s<clip)lo=v;else break;}
    s=0;
    for (let v=255;v>=0;v--){s+=hist[v];if(s<clip)hi=v;else break;}
    if (hi<=lo) continue;
    const sc=255/(hi-lo);
    for (let i=ch;i<d.length;i+=4) d[i]=Math.max(0,Math.min(255,Math.round((d[i]-lo)*sc)));
  }
  ctx.putImageData(full,0,0);
  return out.toDataURL('image/jpeg',0.95);
}

// ─── Rotate helpers ───────────────────────────────────────────────────────────

function rotateCanvas90cw(src: HTMLCanvasElement): HTMLCanvasElement {
  const out = document.createElement('canvas');
  out.width = src.height; out.height = src.width;
  const ctx = out.getContext('2d')!;
  ctx.translate(out.width, 0);
  ctx.rotate(Math.PI / 2);
  ctx.drawImage(src, 0, 0);
  return out;
}

async function rotateDataUrl90cw(dataUrl: string): Promise<string> {
  const img = new Image();
  await new Promise<void>(r => { img.onload = () => r(); img.src = dataUrl; });
  const out = document.createElement('canvas');
  out.width = img.naturalHeight; out.height = img.naturalWidth;
  const ctx = out.getContext('2d')!;
  ctx.translate(out.width, 0);
  ctx.rotate(Math.PI / 2);
  ctx.drawImage(img, 0, 0);
  return out.toDataURL('image/jpeg', 0.95);
}

// ─── Icons ────────────────────────────────────────────────────────────────────

const IcoRotate = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 2v6h-6"/><path d="M21 13a9 9 0 1 1-3-7.7L21 8"/>
  </svg>
);
const IcoCheck = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="20 6 9 17 4 12"/>
  </svg>
);
const IcoX = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
  </svg>
);
const IcoDownload = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
    <polyline points="7 10 12 15 17 10"/>
    <line x1="12" y1="15" x2="12" y2="3"/>
  </svg>
);
const IcoArrowLeft = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M19 12H5M12 5l-7 7 7 7"/>
  </svg>
);
const IcoSpin = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" style={{animation:'spin .8s linear infinite'}}>
    <path d="M12 2a10 10 0 0 1 10 10"/>
  </svg>
);

// ─── Small reusable UI pieces ─────────────────────────────────────────────────

const IconBtn: React.FC<{onClick:()=>void; label:string; icon:React.ReactNode; disabled?:boolean}> = ({onClick,label,icon,disabled}) => (
  <button onClick={onClick} disabled={disabled} aria-label={label} style={{
    width:56, display:'flex', flexDirection:'column', alignItems:'center', gap:4,
    background:'none', border:'none', cursor: disabled?'default':'pointer',
    padding:0, opacity: disabled ? 0.35 : 1,
  }}>
    <div style={{
      width:44, height:44, borderRadius:14,
      background:'rgba(255,255,255,0.07)',
      border:'1px solid rgba(255,255,255,0.1)',
      display:'flex', alignItems:'center', justifyContent:'center',
      color:'#cbd5e1',
    }}>{icon}</div>
    <span style={{fontSize:10, color:'#4a5568', fontWeight:500, letterSpacing:'0.02em'}}>{label}</span>
  </button>
);

const PrimaryBtn: React.FC<{onClick:()=>void; label:string; icon?:React.ReactNode; disabled?:boolean}> = ({onClick,label,icon,disabled}) => (
  <button onClick={onClick} disabled={disabled} style={{
    flex:1, height:50, borderRadius:25,
    background: disabled ? '#7c4f0a' : 'linear-gradient(135deg,#f59e0b,#d97706)',
    color:'#fff', border:'none', cursor: disabled?'default':'pointer',
    display:'flex', alignItems:'center', justifyContent:'center', gap:8,
    fontSize:15, fontWeight:600, letterSpacing:'0.01em',
    boxShadow: disabled ? 'none' : '0 4px 14px rgba(217,119,6,0.4)',
  }}>
    {icon}{label}
  </button>
);

// ─── Zoom Mirror ──────────────────────────────────────────────────────────────

const MSIZ = 130;

const ZoomMirror: React.FC<{srcCanvas:HTMLCanvasElement; pos:Point; idx:number}> = ({srcCanvas,pos,idx}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    const ZOOM = 2.2, srcSz = MSIZ / ZOOM;
    const sx = Math.max(0, Math.min(srcCanvas.width  - srcSz, pos.x - srcSz/2));
    const sy = Math.max(0, Math.min(srcCanvas.height - srcSz, pos.y - srcSz/2));
    ctx.clearRect(0, 0, MSIZ, MSIZ);
    ctx.drawImage(srcCanvas, sx, sy, srcSz, srcSz, 0, 0, MSIZ, MSIZ);
    const cx = Math.round((pos.x - sx) * ZOOM), cy = Math.round((pos.y - sy) * ZOOM);
    ctx.strokeStyle = 'rgba(251,191,36,0.85)'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(cx,0); ctx.lineTo(cx,MSIZ);
    ctx.moveTo(0,cy); ctx.lineTo(MSIZ,cy);
    ctx.stroke();
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath(); ctx.arc(cx, cy, 2.5, 0, Math.PI*2); ctx.fill();
  }, [srcCanvas, pos]);

  const pos4: React.CSSProperties[] = [
    {bottom:12, right:12}, {bottom:12, left:12}, {top:12, left:12}, {top:12, right:12},
  ];
  return (
    <canvas ref={ref} width={MSIZ} height={MSIZ} style={{
      position:'absolute', ...pos4[idx], width:MSIZ, height:MSIZ,
      borderRadius:12,
      border:'1.5px solid rgba(255,255,255,0.2)',
      boxShadow:'0 8px 24px rgba(0,0,0,0.7)',
      pointerEvents:'none', zIndex:30,
      imageRendering:'pixelated',
    }}/>
  );
};

// ─── Crop Editor ──────────────────────────────────────────────────────────────

const CropEditor: React.FC<{
  imageUrl:string; imgW:number; imgH:number;
  corners:[Point,Point,Point,Point];
  onChange:(c:[Point,Point,Point,Point])=>void;
  srcCanvas:HTMLCanvasElement;
}> = ({imageUrl, imgW, imgH, corners, onChange, srcCanvas}) => {
  const svgRef    = useRef<SVGSVGElement>(null);
  const drag      = useRef<number|null>(null);
  const dragOff   = useRef<Point>({x:0, y:0});
  const [zoom, setZoom] = useState<{idx:number; pos:Point}|null>(null);

  const toSvgPt = (e: React.PointerEvent): Point => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    return pt.matrixTransform(svg.getScreenCTM()!.inverse());
  };

  const onDown = (e: React.PointerEvent, i: number) => {
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = i;
    const pt = toSvgPt(e);
    // Capture offset so the handle doesn't jump to finger tip on first move
    dragOff.current = {x: pt.x - corners[i].x, y: pt.y - corners[i].y};
    setZoom({idx:i, pos:corners[i]});
  };
  const onMove = (e: React.PointerEvent) => {
    if (drag.current === null) return;
    e.preventDefault();
    const raw = toSvgPt(e);
    const pt = {
      x: Math.max(0, Math.min(imgW, raw.x - dragOff.current.x)),
      y: Math.max(0, Math.min(imgH, raw.y - dragOff.current.y)),
    };
    const n = [...corners] as [Point,Point,Point,Point];
    n[drag.current] = pt; onChange(n); setZoom({idx:drag.current, pos:pt});
  };
  const onUp = () => { drag.current = null; setZoom(null); };

  const [tl,tr,br,bl] = corners;
  const pts = `${tl.x},${tl.y} ${tr.x},${tr.y} ${br.x},${br.y} ${bl.x},${bl.y}`;
  const R  = Math.max(imgW, imgH) * 0.018;
  const SW = Math.max(imgW, imgH) * 0.004;

  return (
    <div style={{position:'relative', width:'100%', height:'100%'}}>
      <img src={imageUrl} alt="" draggable={false}
        style={{width:'100%', height:'100%', objectFit:'contain', display:'block', userSelect:'none'}}/>
      <svg ref={svgRef} viewBox={`0 0 ${imgW} ${imgH}`} preserveAspectRatio="xMidYMid meet"
        style={{position:'absolute', inset:0, width:'100%', height:'100%', touchAction:'none'}}
        onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        <mask id="crop-mask">
          <rect width={imgW} height={imgH} fill="white"/>
          <polygon points={pts} fill="black"/>
        </mask>
        <rect width={imgW} height={imgH} fill="rgba(0,0,0,0.55)" mask="url(#crop-mask)"/>
        <polygon points={pts} fill="rgba(251,191,36,0.07)"
          stroke="#fbbf24" strokeWidth={SW} strokeLinejoin="round"/>
        {/* Corner handles: large invisible tap zone + small visible dot */}
        {corners.map((c, i) => (
          <g key={i} onPointerDown={e => onDown(e, i)} style={{cursor:'grab'}}>
            <circle cx={c.x} cy={c.y} r={R*3} fill="transparent"/>
            <circle cx={c.x} cy={c.y} r={R} fill="#fbbf24" stroke="rgba(255,255,255,0.9)" strokeWidth={SW}/>
          </g>
        ))}
      </svg>
      {zoom && <ZoomMirror srcCanvas={srcCanvas} pos={zoom.pos} idx={zoom.idx}/>}
    </div>
  );
};

// ─── App ──────────────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const videoRef       = useRef<HTMLVideoElement>(null);
  const capturedCanvas = useRef<HTMLCanvasElement|null>(null);
  const streamRef      = useRef<MediaStream|null>(null);
  const prevRef        = useRef<[Point,Point,Point,Point]|null>(null);

  const [mode,          setMode]         = useState<Mode>('camera');
  const [vDims,         setVDims]        = useState({w:1920, h:1080});
  const [liveCorners,   setLiveCorners]  = useState<[Point,Point,Point,Point]|null>(null);
  const [capturedImage, setCapturedImage]= useState<string|null>(null);
  const [imgSize,       setImgSize]      = useState({w:0, h:0});
  const [corners,       setCorners]      = useState<[Point,Point,Point,Point]|null>(null);
  const [processedImage,setProcessed]    = useState<string|null>(null);
  const [processing,    setProcessing]   = useState(false);
  const [error,         setError]        = useState<string|null>(null);
  const [toast,         setToast]        = useState('');

  const showToast = (msg: string) => { setToast(msg); setTimeout(() => setToast(''), 2000); };

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {facingMode:{ideal:'environment'}, width:{ideal:3840}, height:{ideal:2160}},
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
    } catch {
      setError('Camera access denied. Allow camera permission in your browser settings, then refresh.');
    }
  }, []);

  useEffect(() => { startCamera(); return stopCamera; }, [startCamera, stopCamera]);

  // Live detection every 300 ms
  useEffect(() => {
    if (mode !== 'camera') { setLiveCorners(null); prevRef.current = null; return; }
    const id = setInterval(() => {
      const v = videoRef.current;
      if (!v || v.readyState < 2) return;
      const found = detectLive(v);
      if (found) {
        const prev = prevRef.current;
        const smoothed = prev
          ? found.map((c,i) => ({x:c.x*.4+prev[i].x*.6, y:c.y*.4+prev[i].y*.6})) as [Point,Point,Point,Point]
          : found;
        prevRef.current = smoothed;
        setLiveCorners(smoothed);
      } else {
        prevRef.current = null;
        setLiveCorners(null);
      }
    }, 300);
    return () => clearInterval(id);
  }, [mode]);

  const capture = useCallback(() => {
    const v = videoRef.current;
    if (!v || v.readyState < 2) return;
    const W = v.videoWidth, H = v.videoHeight;
    const cap = document.createElement('canvas'); cap.width = W; cap.height = H;
    cap.getContext('2d')!.drawImage(v, 0, 0);
    capturedCanvas.current = cap;
    setCorners(detectStatic(cap));
    setImgSize({w:W, h:H});
    setCapturedImage(cap.toDataURL('image/jpeg', 0.92));
    setMode('crop');
    stopCamera();
  }, [stopCamera]);

  const rotateCrop = useCallback(() => {
    if (!capturedCanvas.current) return;
    const r = rotateCanvas90cw(capturedCanvas.current);
    capturedCanvas.current = r;
    setImgSize({w:r.width, h:r.height});
    setCorners(defaultCorners(r.width, r.height));
    setCapturedImage(r.toDataURL('image/jpeg', 0.92));
  }, []);

  const rotatePreview = useCallback(async () => {
    if (!processedImage) return;
    setProcessing(true);
    try { setProcessed(await rotateDataUrl90cw(processedImage)); }
    finally { setProcessing(false); }
  }, [processedImage]);

  const retake = useCallback(() => {
    setCapturedImage(null); setProcessed(null); setToast('');
    capturedCanvas.current = null;
    setMode('camera'); startCamera();
  }, [startCamera]);

  const applyCrop = useCallback(async () => {
    if (!capturedCanvas.current || !corners) return;
    setProcessing(true);
    await new Promise(r => setTimeout(r, 30));
    try { setProcessed(warpAndEnhance(capturedCanvas.current, corners)); setMode('preview'); }
    finally { setProcessing(false); }
  }, [corners]);

  const download = useCallback(() => {
    if (!processedImage) return;
    const ts = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
    const a = document.createElement('a'); a.href = processedImage; a.download = `scan_${ts}.jpg`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    showToast('Saved!');
  }, [processedImage]);

  const detected = liveCorners !== null;

  return (
    <div style={{
      height:'100dvh', display:'flex', flexDirection:'column', overflow:'hidden',
      background:'#0b0f1a', color:'#f1f5f9',
      fontFamily:'-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",Roboto,sans-serif',
      userSelect:'none',
    }}>

      {/* ── Header ── */}
      <div style={{
        height:52, flexShrink:0, position:'relative',
        display:'flex', alignItems:'center', justifyContent:'center',
        borderBottom:'1px solid rgba(255,255,255,0.06)',
      }}>
        {mode !== 'camera' && (
          <button onClick={retake} aria-label="Back" style={{
            position:'absolute', left:10,
            width:36, height:36, borderRadius:10,
            background:'rgba(255,255,255,0.06)', border:'none',
            display:'flex', alignItems:'center', justifyContent:'center',
            cursor:'pointer', color:'#94a3b8',
          }}>
            <IcoArrowLeft/>
          </button>
        )}
        <div style={{display:'flex', alignItems:'center', gap:8}}>
          {/* Scanner icon */}
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" strokeWidth="2" strokeLinecap="round">
            <path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>
            <path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>
            <rect x="7" y="7" width="10" height="10" rx="1"/>
          </svg>
          <span style={{fontSize:12, fontWeight:700, letterSpacing:'0.12em', color:'#6b7a94', textTransform:'uppercase' as const}}>
            Doc Scanner
          </span>
        </div>
        {mode === 'crop' && (
          <span style={{position:'absolute', right:14, fontSize:11, color:'#374151', fontWeight:500}}>
            drag corners
          </span>
        )}
      </div>

      {/* ── Main viewport ── */}
      <div style={{
        flex:1, position:'relative', overflow:'hidden',
        display:'flex', alignItems:'center', justifyContent:'center',
        background:'#000',
      }}>
        {error ? (
          <div style={{
            padding:'32px 28px', textAlign:'center', color:'#f87171',
            fontSize:14, lineHeight:1.7, maxWidth:280,
          }}>
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
              style={{marginBottom:16, opacity:.7, display:'block', margin:'0 auto 16px'}}>
              <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
              <circle cx="12" cy="13" r="4"/>
            </svg>
            {error}
          </div>
        ) : mode === 'camera' ? (
          <>
            <video ref={videoRef}
              style={{width:'100%', height:'100%', objectFit:'contain'}}
              autoPlay playsInline muted
              onLoadedMetadata={() => {
                const v = videoRef.current;
                if (v) setVDims({w:v.videoWidth, h:v.videoHeight});
              }}
            />
            {/* Detection overlay */}
            <svg viewBox={`0 0 ${vDims.w} ${vDims.h}`} preserveAspectRatio="xMidYMid meet"
              style={{position:'absolute', inset:0, width:'100%', height:'100%', pointerEvents:'none'}}>
              {liveCorners ? (
                <polygon
                  points={liveCorners.map(c=>`${c.x},${c.y}`).join(' ')}
                  fill="rgba(251,191,36,0.08)"
                  stroke="#fbbf24"
                  strokeWidth={Math.max(vDims.w,vDims.h)*0.003}
                  strokeLinejoin="round"
                />
              ) : (() => {
                const bx=vDims.w*.08, by=vDims.h*.1;
                const bw=vDims.w*.84, bh=vDims.h*.8;
                const arm=Math.min(vDims.w,vDims.h)*.05;
                const sw=Math.max(vDims.w,vDims.h)*.004;
                return ([
                  [bx,by,1,1],[bx+bw,by,-1,1],[bx+bw,by+bh,-1,-1],[bx,by+bh,1,-1]
                ] as [number,number,number,number][]).map(([x,y,dx,dy],i) => (
                  <g key={i}>
                    <line x1={x} y1={y} x2={x+dx*arm} y2={y} stroke="rgba(255,255,255,0.2)" strokeWidth={sw} strokeLinecap="round"/>
                    <line x1={x} y1={y} x2={x} y2={y+dy*arm} stroke="rgba(255,255,255,0.2)" strokeWidth={sw} strokeLinecap="round"/>
                  </g>
                ));
              })()}
            </svg>
          </>
        ) : mode === 'crop' && capturedImage && corners && imgSize.w > 0 ? (
          <>
            <CropEditor
              imageUrl={capturedImage} imgW={imgSize.w} imgH={imgSize.h}
              corners={corners} onChange={c => setCorners(c)}
              srcCanvas={capturedCanvas.current!}
            />
            {processing && <LoadingOverlay label="Processing…"/>}
          </>
        ) : mode === 'preview' && processedImage ? (
          <>
            <img src={processedImage} alt="Scanned document"
              style={{width:'100%', height:'100%', objectFit:'contain'}}/>
            {processing && <LoadingOverlay label="Rotating…"/>}
          </>
        ) : null}

        {/* Toast */}
        {toast && (
          <div style={{
            position:'absolute', bottom:20, left:'50%', transform:'translateX(-50%)',
            background:'rgba(15,23,42,0.92)', backdropFilter:'blur(12px)',
            color:'#f1f5f9', padding:'10px 22px', borderRadius:20,
            fontSize:13, fontWeight:500, pointerEvents:'none',
            border:'1px solid rgba(255,255,255,0.1)',
            boxShadow:'0 8px 24px rgba(0,0,0,0.5)',
          }}>{toast}</div>
        )}
      </div>

      {/* ── Controls bar ── */}
      <div style={{
        flexShrink:0, background:'#0b0f1a',
        borderTop:'1px solid rgba(255,255,255,0.06)',
        padding:'14px 20px',
        paddingBottom:'max(18px, env(safe-area-inset-bottom, 14px))',
      }}>
        {mode === 'camera' ? (
          /* Shutter button */
          <div style={{display:'flex', justifyContent:'center', alignItems:'center', paddingBottom:4}}>
            <button onClick={capture} aria-label="Capture" style={{
              width:72, height:72, borderRadius:'50%', cursor:'pointer',
              border:`3px solid ${detected ? '#fbbf24' : 'rgba(255,255,255,0.25)'}`,
              background:'transparent',
              display:'flex', alignItems:'center', justifyContent:'center',
              boxShadow: detected
                ? '0 0 0 5px rgba(251,191,36,0.18), 0 0 24px rgba(251,191,36,0.2)'
                : 'none',
              transition:'border-color .25s, box-shadow .25s',
            }}>
              <div style={{
                width:54, height:54, borderRadius:'50%',
                background: detected ? '#fbbf24' : 'rgba(255,255,255,0.88)',
                transition:'background .25s',
              }}/>
            </button>
          </div>
        ) : mode === 'crop' ? (
          <div style={{display:'flex', alignItems:'flex-start', gap:10}}>
            <IconBtn onClick={rotateCrop} label="Rotate" icon={<IcoRotate/>}/>
            <PrimaryBtn onClick={applyCrop} disabled={processing}
              label={processing ? 'Processing…' : 'Crop & Apply'}
              icon={processing ? <IcoSpin/> : <IcoCheck/>}
            />
            <IconBtn onClick={retake} label="Retake" icon={<IcoX/>}/>
          </div>
        ) : (
          <div style={{display:'flex', alignItems:'flex-start', gap:10}}>
            <IconBtn onClick={rotatePreview} label="Rotate" icon={<IcoRotate/>} disabled={processing}/>
            <PrimaryBtn onClick={download} label="Save Image" icon={<IcoDownload/>}/>
            <IconBtn onClick={retake} label="New Scan" icon={<IcoX/>}/>
          </div>
        )}
      </div>

      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </div>
  );
};

const LoadingOverlay: React.FC<{label:string}> = ({label}) => (
  <div style={{
    position:'absolute', inset:0,
    background:'rgba(11,15,26,0.75)', backdropFilter:'blur(4px)',
    display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
    gap:14, color:'#e2e8f0', fontSize:14,
  }}>
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" style={{animation:'spin .9s linear infinite'}}>
      <circle cx="18" cy="18" r="15" stroke="rgba(255,255,255,0.12)" strokeWidth="3"/>
      <path d="M18 3 A15 15 0 0 1 33 18" stroke="#fbbf24" strokeWidth="3" strokeLinecap="round"/>
    </svg>
    <span style={{fontWeight:500, color:'#94a3b8'}}>{label}</span>
  </div>
);

export default App;
