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

// ─── Zoom Mirror (2× zoom, no crosshair overload) ────────────────────────────

const MSIZ = 140;

const ZoomMirror: React.FC<{srcCanvas:HTMLCanvasElement;pos:Point;idx:number}> = ({srcCanvas,pos,idx}) => {
  const ref=useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c=ref.current; if (!c) return;
    const ctx=c.getContext('2d')!;
    const ZOOM=2.2;
    const srcSz=MSIZ/ZOOM;
    const sx=Math.max(0,Math.min(srcCanvas.width-srcSz,pos.x-srcSz/2));
    const sy=Math.max(0,Math.min(srcCanvas.height-srcSz,pos.y-srcSz/2));
    ctx.clearRect(0,0,MSIZ,MSIZ);
    ctx.drawImage(srcCanvas,sx,sy,srcSz,srcSz,0,0,MSIZ,MSIZ);
    // Thin crosshair
    const cx=Math.round((pos.x-sx)*ZOOM), cy=Math.round((pos.y-sy)*ZOOM);
    ctx.strokeStyle='rgba(251,191,36,0.9)'; ctx.lineWidth=1.5;
    ctx.beginPath();
    ctx.moveTo(cx,0);ctx.lineTo(cx,MSIZ);
    ctx.moveTo(0,cy);ctx.lineTo(MSIZ,cy);
    ctx.stroke();
    ctx.fillStyle='#fbbf24';
    ctx.beginPath();ctx.arc(cx,cy,2.5,0,Math.PI*2);ctx.fill();
  },[srcCanvas,pos]);

  // Position in the corner opposite the dragged corner
  const off: React.CSSProperties[] = [
    {bottom:12,right:12},{bottom:12,left:12},{top:12,left:12},{top:12,right:12}
  ];
  return (
    <canvas ref={ref} width={MSIZ} height={MSIZ} style={{
      position:'absolute',...off[idx],width:MSIZ,height:MSIZ,
      borderRadius:10,border:'2px solid rgba(255,255,255,0.88)',
      boxShadow:'0 4px 16px rgba(0,0,0,0.65)',pointerEvents:'none',zIndex:30,
      imageRendering:'pixelated',
    }}/>
  );
};

// ─── Crop Editor ──────────────────────────────────────────────────────────────

const CropEditor: React.FC<{
  imageUrl:string;imgW:number;imgH:number;
  corners:[Point,Point,Point,Point];
  onChange:(c:[Point,Point,Point,Point])=>void;
  srcCanvas:HTMLCanvasElement;
}> = ({imageUrl,imgW,imgH,corners,onChange,srcCanvas}) => {
  const svgRef=useRef<SVGSVGElement>(null);
  const drag=useRef<number|null>(null);
  const [zoom,setZoom]=useState<{idx:number;pos:Point}|null>(null);

  const toPt=(e:React.PointerEvent): Point => {
    const svg=svgRef.current!;
    const pt=svg.createSVGPoint(); pt.x=e.clientX; pt.y=e.clientY;
    const tp=pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return {x:Math.max(0,Math.min(imgW,tp.x)),y:Math.max(0,Math.min(imgH,tp.y))};
  };

  const onDown=(e:React.PointerEvent,i:number)=>{
    e.preventDefault();(e.target as Element).setPointerCapture(e.pointerId);
    drag.current=i;setZoom({idx:i,pos:corners[i]});
  };
  const onMove=(e:React.PointerEvent)=>{
    if (drag.current===null) return; e.preventDefault();
    const pt=toPt(e);
    const n=[...corners] as [Point,Point,Point,Point];
    n[drag.current]=pt; onChange(n); setZoom({idx:drag.current,pos:pt});
  };
  const onUp=()=>{drag.current=null;setZoom(null);};

  const [tl,tr,br,bl]=corners;
  const pts=`${tl.x},${tl.y} ${tr.x},${tr.y} ${br.x},${br.y} ${bl.x},${bl.y}`;
  const R=Math.max(imgW,imgH)*0.032;
  const SW=Math.max(imgW,imgH)*0.006;

  return (
    <div style={{position:'relative',width:'100%',height:'100%'}}>
      <img src={imageUrl} alt="Document" draggable={false}
        style={{width:'100%',height:'100%',objectFit:'contain',display:'block',userSelect:'none'}}/>
      <svg ref={svgRef} viewBox={`0 0 ${imgW} ${imgH}`} preserveAspectRatio="xMidYMid meet"
        style={{position:'absolute',inset:0,width:'100%',height:'100%',touchAction:'none'}}
        onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}>
        {/* Dim outside selection */}
        <mask id="dm">
          <rect width={imgW} height={imgH} fill="white"/>
          <polygon points={pts} fill="black"/>
        </mask>
        <rect width={imgW} height={imgH} fill="rgba(0,0,0,0.5)" mask="url(#dm)"/>
        {/* Yellow outline */}
        <polygon points={pts} fill="rgba(251,191,36,0.08)"
          stroke="#fbbf24" strokeWidth={SW} strokeLinejoin="round"/>
        {/* Corner handles — simple filled circles, no clutter */}
        {corners.map((c,i)=>(
          <g key={i} onPointerDown={e=>onDown(e,i)} style={{cursor:'grab'}}>
            <circle cx={c.x} cy={c.y} r={R*1.8} fill="transparent"/>
            <circle cx={c.x} cy={c.y} r={R} fill="#fbbf24" stroke="white" strokeWidth={SW*0.7}/>
          </g>
        ))}
      </svg>
      {zoom&&<ZoomMirror srcCanvas={srcCanvas} pos={zoom.pos} idx={zoom.idx}/>}
    </div>
  );
};

// ─── App ──────────────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const videoRef       = useRef<HTMLVideoElement>(null);
  const capturedCanvas = useRef<HTMLCanvasElement|null>(null);
  const streamRef      = useRef<MediaStream|null>(null);
  const prevRef        = useRef<[Point,Point,Point,Point]|null>(null);

  const [mode,           setMode]          = useState<Mode>('camera');
  const [vDims,          setVDims]         = useState({w:1920,h:1080});
  const [liveCorners,    setLiveCorners]   = useState<[Point,Point,Point,Point]|null>(null);
  const [capturedImage,  setCapturedImage] = useState<string|null>(null);
  const [imgSize,        setImgSize]       = useState({w:0,h:0});
  const [corners,        setCorners]       = useState<[Point,Point,Point,Point]|null>(null);
  const [processedImage, setProcessed]     = useState<string|null>(null);
  const [processing,     setProcessing]    = useState(false);
  const [error,          setError]         = useState<string|null>(null);
  const [savedMsg,       setSavedMsg]      = useState('');

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t=>t.stop());
    streamRef.current=null;
  },[]);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream=await navigator.mediaDevices.getUserMedia({
        video:{facingMode:{ideal:'environment'},width:{ideal:3840},height:{ideal:2160}},audio:false
      });
      streamRef.current=stream;
      if (videoRef.current) videoRef.current.srcObject=stream;
    } catch {
      setError('Camera access denied.\nAllow camera permission in your browser settings, then refresh.');
    }
  },[]);

  useEffect(()=>{startCamera();return stopCamera;},[startCamera,stopCamera]);

  // Live detection every 300 ms — no auto-capture, just visual feedback
  useEffect(()=>{
    if (mode!=='camera'){setLiveCorners(null);prevRef.current=null;return;}
    const id=setInterval(()=>{
      const v=videoRef.current;
      if (!v||v.readyState<2) return;
      // Smooth the detected corners with the previous frame (reduce jitter)
      const found=detectLive(v);
      if (found) {
        const prev=prevRef.current;
        if (prev) {
          const smoothed=found.map((c,i)=>({
            x:c.x*0.4+prev[i].x*0.6,
            y:c.y*0.4+prev[i].y*0.6,
          })) as [Point,Point,Point,Point];
          prevRef.current=smoothed;
          setLiveCorners(smoothed);
        } else {
          prevRef.current=found;
          setLiveCorners(found);
        }
      } else {
        prevRef.current=null;
        setLiveCorners(null);
      }
    },300);
    return ()=>clearInterval(id);
  },[mode]);

  const capture=useCallback(()=>{
    const v=videoRef.current;
    if (!v||v.readyState<2) return;
    const W=v.videoWidth, H=v.videoHeight;
    const cap=document.createElement('canvas'); cap.width=W; cap.height=H;
    cap.getContext('2d')!.drawImage(v,0,0);
    capturedCanvas.current=cap;
    // Re-detect on the full captured frame (higher resolution = more accurate)
    const det=detectStatic(cap);
    setCorners(det);
    setImgSize({w:W,h:H});
    setCapturedImage(cap.toDataURL('image/jpeg',0.92));
    setMode('crop');
    stopCamera();
  },[stopCamera]);

  const retake=useCallback(()=>{
    setCapturedImage(null);setProcessed(null);setSavedMsg('');
    capturedCanvas.current=null;
    setMode('camera');startCamera();
  },[startCamera]);

  const applyCrop=useCallback(async()=>{
    if (!capturedCanvas.current||!corners) return;
    setProcessing(true);
    await new Promise(r=>setTimeout(r,30));
    try{setProcessed(warpAndEnhance(capturedCanvas.current,corners));setMode('preview');}
    finally{setProcessing(false);}
  },[corners]);

  const download=useCallback(()=>{
    if (!processedImage) return;
    const ts=new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
    const a=document.createElement('a');
    a.href=processedImage; a.download=`scan_${ts}.jpg`;
    document.body.appendChild(a);a.click();document.body.removeChild(a);
    setSavedMsg('Saved!');setTimeout(()=>setSavedMsg(''),2000);
  },[processedImage]);

  const detected=liveCorners!==null;

  return (
    <div style={{
      height:'100dvh',display:'flex',flexDirection:'column',
      background:'#0f172a',color:'#f1f5f9',
      fontFamily:'system-ui,-apple-system,sans-serif',
      userSelect:'none',overflow:'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding:'10px 16px',fontSize:'15px',fontWeight:700,
        letterSpacing:'0.08em',textAlign:'center',color:'#475569',
        flexShrink:0,position:'relative',display:'flex',alignItems:'center',justifyContent:'center',
      }}>
        {mode!=='camera'&&(
          <button onClick={retake} style={{
            position:'absolute',left:12,background:'none',border:'none',
            color:'#64748b',fontSize:'13px',cursor:'pointer',padding:'4px 8px',
          }}>✕ Retake</button>
        )}
        DOC SCANNER
        {mode==='crop'&&(
          <span style={{position:'absolute',right:12,fontSize:'11px',color:'#475569',fontWeight:400}}>
            drag corners
          </span>
        )}
      </div>

      {/* Viewport */}
      <div style={{
        flex:1,position:'relative',overflow:'hidden',
        display:'flex',alignItems:'center',justifyContent:'center',background:'#000',
      }}>
        {error?(
          <div style={{padding:'32px',textAlign:'center',color:'#f87171',fontSize:'15px',lineHeight:1.65}}>
            {error}
          </div>
        ):mode==='camera'?(
          <>
            <video ref={videoRef}
              style={{width:'100%',height:'100%',objectFit:'contain'}}
              autoPlay playsInline muted
              onLoadedMetadata={()=>{
                const v=videoRef.current;
                if (v) setVDims({w:v.videoWidth,h:v.videoHeight});
              }}
            />
            {/* Live overlay — SVG viewBox matches actual video resolution */}
            <svg
              viewBox={`0 0 ${vDims.w} ${vDims.h}`}
              preserveAspectRatio="xMidYMid meet"
              style={{position:'absolute',inset:0,width:'100%',height:'100%',pointerEvents:'none'}}
            >
              {liveCorners?(
                // Detected: yellow quad with animated opacity
                <polygon
                  points={liveCorners.map(c=>`${c.x},${c.y}`).join(' ')}
                  fill="rgba(251,191,36,0.13)"
                  stroke="#fbbf24"
                  strokeWidth={Math.max(vDims.w,vDims.h)*0.004}
                  strokeLinejoin="round"
                  style={{animation:'fadeIn 0.2s ease'}}
                />
              ):(
                // Idle: subtle corner brackets
                (()=>{
                  const bx=vDims.w*0.08,by=vDims.h*0.10;
                  const bw=vDims.w*0.84,bh=vDims.h*0.80;
                  const arm=Math.min(vDims.w,vDims.h)*0.055;
                  const sw=Math.max(vDims.w,vDims.h)*0.004;
                  return ([
                    [bx,by,1,1],[bx+bw,by,-1,1],[bx+bw,by+bh,-1,-1],[bx,by+bh,1,-1]
                  ] as [number,number,number,number][]).map(([x,y,dx,dy],i)=>(
                    <g key={i}>
                      <line x1={x} y1={y} x2={x+dx*arm} y2={y} stroke="rgba(255,255,255,0.3)" strokeWidth={sw} strokeLinecap="round"/>
                      <line x1={x} y1={y} x2={x} y2={y+dy*arm} stroke="rgba(255,255,255,0.3)" strokeWidth={sw} strokeLinecap="round"/>
                    </g>
                  ));
                })()
              )}
            </svg>
          </>
        ):mode==='crop'&&capturedImage&&corners&&imgSize.w>0?(
          <>
            <CropEditor
              imageUrl={capturedImage} imgW={imgSize.w} imgH={imgSize.h}
              corners={corners} onChange={c=>setCorners(c)}
              srcCanvas={capturedCanvas.current!}
            />
            {processing&&(
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
        ):mode==='preview'&&processedImage?(
          <>
            <img src={processedImage} alt="Scan"
              style={{width:'100%',height:'100%',objectFit:'contain'}}/>
            {savedMsg&&(
              <div style={{
                position:'absolute',top:'50%',left:'50%',transform:'translate(-50%,-50%)',
                background:'rgba(0,0,0,0.78)',color:'#fff',
                padding:'12px 28px',borderRadius:12,fontSize:15,pointerEvents:'none',
              }}>{savedMsg}</div>
            )}
          </>
        ):null}
      </div>

      {/* Controls */}
      <div style={{
        padding:'18px 24px 38px',display:'flex',alignItems:'center',
        justifyContent:'center',gap:16,flexShrink:0,background:'#0f172a',
      }}>
        {mode==='camera'?(
          // Capture button — yellow ring when document detected
          <button onClick={capture} aria-label="Capture"
            style={{
              width:72,height:72,borderRadius:'50%',
              border:`4px solid ${detected?'#fbbf24':'rgba(255,255,255,0.45)'}`,
              background:'transparent',cursor:'pointer',
              display:'flex',alignItems:'center',justifyContent:'center',
              transition:'border-color 0.25s',
              boxShadow:detected?'0 0 0 3px rgba(251,191,36,0.25)':'none',
            }}>
            <div style={{
              width:54,height:54,borderRadius:'50%',
              background:detected?'#fbbf24':'rgba(255,255,255,0.8)',
              transition:'background 0.25s',
            }}/>
          </button>
        ):mode==='crop'?(
          <>
            <button onClick={retake}    style={btn('dark')}>Retake</button>
            <button onClick={applyCrop} style={btn('gold')} disabled={processing}>
              {processing?'Processing…':'Crop & Apply'}
            </button>
          </>
        ):(
          <>
            <button onClick={retake}   style={btn('dark')}>Retake</button>
            <button onClick={download} style={btn('gold')}>↓ Download</button>
          </>
        )}
      </div>

      <style>{`
        @keyframes spin   { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity:0; } to { opacity:1; } }
      `}</style>
    </div>
  );
};

function btn(v:'dark'|'gold'): React.CSSProperties {
  return {
    padding:'13px 28px',borderRadius:12,border:'none',
    fontSize:15,fontWeight:600,cursor:'pointer',
    background:v==='gold'?'#d97706':'#1e293b',
    color:v==='gold'?'#fff':'#94a3b8',
    minWidth:v==='gold'?136:undefined,
  };
}

export default App;
