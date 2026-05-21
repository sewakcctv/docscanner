import React, { useRef, useState, useCallback, useEffect } from 'react';

type Point = { x: number; y: number };
type Mode = 'camera' | 'crop' | 'preview';

// ─── Gaussian elimination (8×8) ──────────────────────────────────────────────

function solve8(A: number[][], b: number[]): number[] {
  const n = 8;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let max = c;
    for (let r = c + 1; r < n; r++)
      if (Math.abs(M[r][c]) > Math.abs(M[max][c])) max = r;
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

// Homography H mapping src[i] → dst[i]; returns [h00…h21] with h22=1
function computeH(src: Point[], dst: Point[]): number[] {
  const A: number[][] = [], b: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x: xs, y: ys } = src[i];
    const { x: xd, y: yd } = dst[i];
    A.push([xs, ys, 1, 0, 0, 0, -xs * xd, -ys * xd]); b.push(xd);
    A.push([0, 0, 0, xs, ys, 1, -xs * yd, -ys * yd]);  b.push(yd);
  }
  return solve8(A, b);
}

function applyH(h: number[], x: number, y: number): Point {
  const d = h[6] * x + h[7] * y + 1;
  return { x: (h[0] * x + h[1] * y + h[2]) / d, y: (h[3] * x + h[4] * y + h[5]) / d };
}

// ─── Document corner detection ───────────────────────────────────────────────

function detectCorners(cap: HTMLCanvasElement): [Point, Point, Point, Point] {
  const { width: W, height: H } = cap;
  const pad = 0.08;
  const def: [Point, Point, Point, Point] = [
    { x: W * pad,       y: H * pad },
    { x: W * (1 - pad), y: H * pad },
    { x: W * (1 - pad), y: H * (1 - pad) },
    { x: W * pad,       y: H * (1 - pad) },
  ];

  try {
    const MAXDIM = 480;
    const scale = MAXDIM / Math.max(W, H);
    const sw = Math.round(W * scale);
    const sh = Math.round(H * scale);

    const small = document.createElement('canvas');
    small.width = sw; small.height = sh;
    small.getContext('2d')!.drawImage(cap, 0, 0, sw, sh);
    const { data } = small.getContext('2d')!.getImageData(0, 0, sw, sh);

    // Median brightness from sampled pixels
    const samples: number[] = [];
    for (let i = 0; i < data.length; i += 16)
      samples.push(data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114);
    samples.sort((a, b) => a - b);
    const median = samples[Math.floor(samples.length / 2)];
    const thr = Math.min(Math.max(median * 1.08, 80), 230);

    // Four extreme bright pixels using diagonal projections
    // TL: min(x+y)  TR: max(x−y)  BR: max(x+y)  BL: min(x−y)
    let tlS = Infinity, trS = -Infinity, brS = -Infinity, blS = Infinity;
    let tl = { x: sw * pad, y: sh * pad };
    let tr = { x: sw * (1 - pad), y: sh * pad };
    let br = { x: sw * (1 - pad), y: sh * (1 - pad) };
    let bl = { x: sw * pad, y: sh * (1 - pad) };

    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        const idx = (y * sw + x) * 4;
        const g = data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114;
        if (g < thr) continue;
        if (x + y < tlS) { tlS = x + y; tl = { x, y }; }
        if (x - y > trS) { trS = x - y; tr = { x, y }; }
        if (x + y > brS) { brS = x + y; br = { x, y }; }
        if (x - y < blS) { blS = x - y; bl = { x, y }; }
      }
    }

    // Validate: each corner must be in its correct quadrant and span enough
    const cx = (tl.x + tr.x + br.x + bl.x) / 4;
    const cy = (tl.y + tr.y + br.y + bl.y) / 4;
    const valid =
      tl.x < cx && tl.y < cy &&
      tr.x > cx && tr.y < cy &&
      br.x > cx && br.y > cy &&
      bl.x < cx && bl.y > cy &&
      (tr.x - tl.x) / sw > 0.15 &&
      (bl.y - tl.y) / sh > 0.15;

    if (!valid) return def;

    const inv = 1 / scale;
    return [
      { x: tl.x * inv, y: tl.y * inv },
      { x: tr.x * inv, y: tr.y * inv },
      { x: br.x * inv, y: br.y * inv },
      { x: bl.x * inv, y: bl.y * inv },
    ];
  } catch {
    return def;
  }
}

// ─── Perspective warp ────────────────────────────────────────────────────────

function perspectiveWarp(
  src: HTMLCanvasElement,
  corners: [Point, Point, Point, Point],
): string {
  const [tl, tr, br, bl] = corners;
  const wTop   = Math.hypot(tr.x - tl.x, tr.y - tl.y);
  const wBot   = Math.hypot(br.x - bl.x, br.y - bl.y);
  const hLeft  = Math.hypot(bl.x - tl.x, bl.y - tl.y);
  const hRight = Math.hypot(br.x - tr.x, br.y - tr.y);

  const MAX = 2048;
  let outW = Math.round(Math.max(wTop, wBot));
  let outH = Math.round(Math.max(hLeft, hRight));
  if (Math.max(outW, outH) > MAX) {
    const s = MAX / Math.max(outW, outH);
    outW = Math.round(outW * s);
    outH = Math.round(outH * s);
  }

  // Homography from output rectangle → source document corners
  const h = computeH(
    [{ x: 0, y: 0 }, { x: outW, y: 0 }, { x: outW, y: outH }, { x: 0, y: outH }],
    corners,
  );

  const srcCtx = src.getContext('2d')!;
  const { data: sData } = srcCtx.getImageData(0, 0, src.width, src.height);
  const sW = src.width, sH = src.height;

  const out = document.createElement('canvas');
  out.width = outW; out.height = outH;
  const outCtx = out.getContext('2d')!;
  const outImg = outCtx.createImageData(outW, outH);
  const oData  = outImg.data;

  for (let yo = 0; yo < outH; yo++) {
    for (let xo = 0; xo < outW; xo++) {
      const { x: xi, y: yi } = applyH(h, xo, yo);
      const x0 = Math.floor(xi), y0 = Math.floor(yi);
      const x1 = x0 + 1,         y1 = y0 + 1;
      const oi  = (yo * outW + xo) * 4;
      if (x0 < 0 || y0 < 0 || x1 >= sW || y1 >= sH) {
        oData[oi] = oData[oi + 1] = oData[oi + 2] = 255; oData[oi + 3] = 255;
        continue;
      }
      const fx = xi - x0, fy = yi - y0;
      const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
      const w01 = (1 - fx) * fy,       w11 = fx * fy;
      const i00 = (y0 * sW + x0) * 4,  i10 = (y0 * sW + x1) * 4;
      const i01 = (y1 * sW + x0) * 4,  i11 = (y1 * sW + x1) * 4;
      oData[oi]     = sData[i00]*w00 + sData[i10]*w10 + sData[i01]*w01 + sData[i11]*w11;
      oData[oi + 1] = sData[i00+1]*w00 + sData[i10+1]*w10 + sData[i01+1]*w01 + sData[i11+1]*w11;
      oData[oi + 2] = sData[i00+2]*w00 + sData[i10+2]*w10 + sData[i01+2]*w01 + sData[i11+2]*w11;
      oData[oi + 3] = 255;
    }
  }

  outCtx.putImageData(outImg, 0, 0);
  return out.toDataURL('image/jpeg', 0.95);
}

// ─── Crop Editor ─────────────────────────────────────────────────────────────

type CropEditorProps = {
  imageUrl: string;
  imgW: number;
  imgH: number;
  corners: [Point, Point, Point, Point];
  onChange: (c: [Point, Point, Point, Point]) => void;
};

const CropEditor: React.FC<CropEditorProps> = ({ imageUrl, imgW, imgH, corners, onChange }) => {
  const svgRef  = useRef<SVGSVGElement>(null);
  const dragIdx = useRef<number | null>(null);

  const getSVGPt = (e: React.PointerEvent): Point => {
    const svg = svgRef.current!;
    const pt  = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const tp = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return {
      x: Math.max(0, Math.min(imgW, tp.x)),
      y: Math.max(0, Math.min(imgH, tp.y)),
    };
  };

  const onDown = (e: React.PointerEvent, i: number) => {
    e.preventDefault();
    (e.target as Element).setPointerCapture(e.pointerId);
    dragIdx.current = i;
  };

  const onMove = (e: React.PointerEvent) => {
    if (dragIdx.current === null) return;
    e.preventDefault();
    const pt   = getSVGPt(e);
    const next = [...corners] as [Point, Point, Point, Point];
    next[dragIdx.current] = pt;
    onChange(next);
  };

  const onUp = () => { dragIdx.current = null; };

  const [tl, tr, br, bl] = corners;
  const pts = `${tl.x},${tl.y} ${tr.x},${tr.y} ${br.x},${br.y} ${bl.x},${bl.y}`;
  const R   = Math.max(imgW, imgH) * 0.045;
  const SW  = R * 0.22;

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
        {/* Darken area outside the document selection */}
        <mask id="docMask">
          <rect width={imgW} height={imgH} fill="white" />
          <polygon points={pts} fill="black" />
        </mask>
        <rect width={imgW} height={imgH} fill="rgba(0,0,0,0.52)" mask="url(#docMask)" />

        {/* Document outline */}
        <polygon
          points={pts}
          fill="rgba(59,130,246,0.08)"
          stroke="#3b82f6"
          strokeWidth={SW}
          strokeLinejoin="round"
        />

        {/* Corner handles — index 0=TL 1=TR 2=BR 3=BL */}
        {corners.map((c, i) => (
          <g key={i} onPointerDown={e => onDown(e, i)} style={{ cursor: 'grab' }}>
            <circle cx={c.x} cy={c.y} r={R * 1.4} fill="transparent" />
            <circle cx={c.x} cy={c.y} r={R} fill="#2563eb" stroke="white" strokeWidth={SW * 0.9} />
            <line x1={c.x - R * 0.38} y1={c.y} x2={c.x + R * 0.38} y2={c.y} stroke="white" strokeWidth={SW * 0.55} strokeLinecap="round" />
            <line x1={c.x} y1={c.y - R * 0.38} x2={c.x} y2={c.y + R * 0.38} stroke="white" strokeWidth={SW * 0.55} strokeLinecap="round" />
          </g>
        ))}
      </svg>
    </div>
  );
};

// ─── App ─────────────────────────────────────────────────────────────────────

const App: React.FC = () => {
  const videoRef         = useRef<HTMLVideoElement>(null);
  const capturedCanvas   = useRef<HTMLCanvasElement | null>(null);
  const streamRef        = useRef<MediaStream | null>(null);

  const [mode,           setMode]           = useState<Mode>('camera');
  const [capturedImage,  setCapturedImage]  = useState<string | null>(null);
  const [imgSize,        setImgSize]        = useState({ w: 0, h: 0 });
  const [corners,        setCorners]        = useState<[Point, Point, Point, Point] | null>(null);
  const [processedImage, setProcessedImage] = useState<string | null>(null);
  const [processing,     setProcessing]     = useState(false);
  const [error,          setError]          = useState<string | null>(null);
  const [saving,         setSaving]         = useState(false);
  const [savedMsg,       setSavedMsg]       = useState('');

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

  const capture = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    const cap = document.createElement('canvas');
    cap.width  = video.videoWidth;
    cap.height = video.videoHeight;
    cap.getContext('2d')!.drawImage(video, 0, 0);
    capturedCanvas.current = cap;

    const detected = detectCorners(cap);
    setCorners(detected);
    setImgSize({ w: cap.width, h: cap.height });
    setCapturedImage(cap.toDataURL('image/jpeg', 0.92));
    setMode('crop');
    stopCamera();
  }, [stopCamera]);

  const retake = useCallback(() => {
    setCapturedImage(null);
    setProcessedImage(null);
    setSavedMsg('');
    capturedCanvas.current = null;
    setMode('camera');
    startCamera();
  }, [startCamera]);

  const applyCrop = useCallback(async () => {
    if (!capturedCanvas.current || !corners) return;
    setProcessing(true);
    await new Promise(r => setTimeout(r, 30)); // yield for "Processing…" to paint
    try {
      const result = perspectiveWarp(capturedCanvas.current, corners);
      setProcessedImage(result);
      setMode('preview');
    } finally {
      setProcessing(false);
    }
  }, [corners]);

  const saveImage = useCallback(async () => {
    const img = processedImage || capturedImage;
    if (!img || saving) return;
    setSaving(true);
    const ts       = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `scan_${ts}.jpg`;
    try {
      const res  = await fetch(img);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      if (typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], title: 'Scanned Document' });
        setSavedMsg('Shared!');
      } else throw new Error('no share');
    } catch {
      const a = document.createElement('a');
      a.href = img; a.download = filename;
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setSavedMsg('Saved!');
    }
    setSaving(false);
    setTimeout(() => setSavedMsg(''), 2500);
  }, [processedImage, capturedImage, saving]);

  // ── styles ────────────────────────────────────────────────────────────────

  const S = {
    root: {
      height: '100dvh', display: 'flex', flexDirection: 'column' as const,
      background: '#111827', color: '#f9fafb',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      userSelect: 'none' as const, overflow: 'hidden',
    } as React.CSSProperties,
    header: {
      padding: '12px 16px', fontSize: '16px', fontWeight: 700,
      letterSpacing: '0.08em', textAlign: 'center' as const,
      color: '#d1d5db', flexShrink: 0, position: 'relative' as const,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    } as React.CSSProperties,
    viewport: {
      flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center',
      overflow: 'hidden', position: 'relative' as const, background: '#000',
    } as React.CSSProperties,
    controls: {
      padding: '18px 24px 36px', display: 'flex', alignItems: 'center',
      justifyContent: 'center', gap: '16px', flexShrink: 0, background: '#111827',
    } as React.CSSProperties,
    btn: (variant: 'dark' | 'blue' | 'capture' | 'ghost') => ({
      ...(variant === 'capture' ? {
        width: '72px', height: '72px', borderRadius: '50%',
        border: '4px solid #f9fafb', background: 'transparent',
        cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
      } : {
        padding: '13px 26px', borderRadius: '12px', border: 'none',
        fontSize: '15px', fontWeight: 600, cursor: 'pointer', letterSpacing: '0.02em',
        background: variant === 'blue' ? '#2563eb' : variant === 'dark' ? '#374151' : 'transparent',
        color: variant === 'blue' ? '#fff' : '#f9fafb',
        minWidth: variant === 'blue' ? '130px' : undefined,
      }),
    }) as React.CSSProperties,
    retakeLink: {
      position: 'absolute' as const, left: 12, background: 'none', border: 'none',
      color: '#6b7280', fontSize: '13px', cursor: 'pointer', padding: '4px 8px',
      display: 'flex', alignItems: 'center', gap: '4px',
    } as React.CSSProperties,
    hint: {
      position: 'absolute' as const, bottom: 12, left: '50%',
      transform: 'translateX(-50%)', color: 'rgba(255,255,255,0.6)',
      fontSize: '12px', pointerEvents: 'none' as const, whiteSpace: 'nowrap' as const,
    } as React.CSSProperties,
    toast: {
      position: 'absolute' as const, top: '50%', left: '50%',
      transform: 'translate(-50%,-50%)', background: 'rgba(0,0,0,0.78)',
      color: '#fff', padding: '12px 28px', borderRadius: '12px', fontSize: '15px',
      pointerEvents: 'none' as const,
    } as React.CSSProperties,
    processing: {
      position: 'absolute' as const, inset: 0, background: 'rgba(0,0,0,0.68)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      color: '#fff', fontSize: '16px', gap: '10px',
    } as React.CSSProperties,
  };

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div style={S.root}>
      <div style={S.header}>
        {mode !== 'camera' && (
          <button style={S.retakeLink} onClick={retake}>✕ Retake</button>
        )}
        DOC SCANNER
        {mode === 'crop' && (
          <span style={{ position: 'absolute', right: 12, fontSize: '11px', color: '#6b7280', fontWeight: 400 }}>
            drag corners
          </span>
        )}
      </div>

      <div style={S.viewport}>
        {error ? (
          <div style={{ padding: '32px', textAlign: 'center', color: '#f87171', fontSize: '15px', lineHeight: 1.65 }}>
            {error}
          </div>
        ) : mode === 'camera' ? (
          <>
            <video
              ref={videoRef}
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
              autoPlay playsInline muted
            />
            {/* Scan-frame guide overlay */}
            <svg
              style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
              viewBox="0 0 100 100"
              preserveAspectRatio="none"
            >
              <rect x="8" y="10" width="84" height="80" rx="1"
                fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="0.3" strokeDasharray="2.5 2" />
              {([
                { bx: 8,  by: 10, dx:  1, dy:  1 },
                { bx: 92, by: 10, dx: -1, dy:  1 },
                { bx: 92, by: 90, dx: -1, dy: -1 },
                { bx: 8,  by: 90, dx:  1, dy: -1 },
              ] as const).map(({ bx, by, dx, dy }, i) => (
                <g key={i}>
                  <line x1={bx} y1={by} x2={bx + dx * 7} y2={by} stroke="#3b82f6" strokeWidth="1.1" strokeLinecap="round" />
                  <line x1={bx} y1={by} x2={bx} y2={by + dy * 7} stroke="#3b82f6" strokeWidth="1.1" strokeLinecap="round" />
                </g>
              ))}
            </svg>
            <div style={S.hint}>Align document within frame</div>
          </>
        ) : mode === 'crop' && capturedImage && corners && imgSize.w > 0 ? (
          <>
            <CropEditor
              imageUrl={capturedImage}
              imgW={imgSize.w}
              imgH={imgSize.h}
              corners={corners}
              onChange={c => setCorners(c)}
            />
            {processing && (
              <div style={S.processing}>
                <svg width="22" height="22" viewBox="0 0 22 22" style={{ animation: 'spin 1s linear infinite' }}>
                  <circle cx="11" cy="11" r="9" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="2.5" />
                  <path d="M11 2 A9 9 0 0 1 20 11" fill="none" stroke="#fff" strokeWidth="2.5" strokeLinecap="round" />
                </svg>
                Processing…
              </div>
            )}
          </>
        ) : mode === 'preview' && processedImage ? (
          <>
            <img
              src={processedImage}
              alt="Scanned document"
              style={{ width: '100%', height: '100%', objectFit: 'contain' }}
            />
            {savedMsg && <div style={S.toast}>{savedMsg}</div>}
          </>
        ) : null}
      </div>

      <div style={S.controls}>
        {mode === 'camera' ? (
          <button style={S.btn('capture')} onClick={capture} aria-label="Capture document">
            <div style={{ width: 54, height: 54, borderRadius: '50%', background: '#f9fafb' }} />
          </button>
        ) : mode === 'crop' ? (
          <>
            <button style={S.btn('dark')} onClick={retake}>Retake</button>
            <button style={S.btn('blue')} onClick={applyCrop} disabled={processing}>
              {processing ? 'Processing…' : 'Crop & Apply'}
            </button>
          </>
        ) : (
          <>
            <button style={S.btn('dark')} onClick={retake}>Retake</button>
            <button style={S.btn('blue')} onClick={saveImage} disabled={saving}>
              {saving ? 'Saving…' : 'Save / Share'}
            </button>
          </>
        )}
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
};

export default App;
