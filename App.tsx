import React, { useRef, useState, useCallback, useEffect } from 'react';

type Mode = 'camera' | 'preview';

const styles = {
  root: {
    height: '100dvh',
    display: 'flex',
    flexDirection: 'column' as const,
    background: '#111827',
    color: '#f9fafb',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    userSelect: 'none' as const,
    overflow: 'hidden',
  },
  header: {
    padding: '12px 16px',
    fontSize: '16px',
    fontWeight: 600,
    letterSpacing: '0.05em',
    textAlign: 'center' as const,
    color: '#d1d5db',
    flexShrink: 0,
  },
  viewportArea: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    overflow: 'hidden',
    position: 'relative' as const,
    background: '#000',
  },
  video: {
    width: '100%',
    height: '100%',
    objectFit: 'cover' as const,
  },
  previewImg: {
    width: '100%',
    height: '100%',
    objectFit: 'contain' as const,
  },
  errorBox: {
    padding: '24px',
    textAlign: 'center' as const,
    color: '#f87171',
    fontSize: '15px',
    lineHeight: 1.5,
  },
  overlay: {
    position: 'absolute' as const,
    inset: 0,
    pointerEvents: 'none' as const,
    border: '2px solid rgba(255,255,255,0.15)',
  },
  controls: {
    padding: '20px 24px 32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '24px',
    flexShrink: 0,
    background: '#111827',
  },
  captureBtn: {
    width: '72px',
    height: '72px',
    borderRadius: '50%',
    border: '4px solid #f9fafb',
    background: 'transparent',
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative' as const,
    flexShrink: 0,
  },
  captureBtnInner: {
    width: '54px',
    height: '54px',
    borderRadius: '50%',
    background: '#f9fafb',
  },
  secondaryBtn: {
    padding: '14px 28px',
    borderRadius: '12px',
    border: 'none',
    fontSize: '15px',
    fontWeight: 600,
    cursor: 'pointer',
    letterSpacing: '0.02em',
  },
  retakeBtn: {
    background: '#374151',
    color: '#f9fafb',
  },
  saveBtn: {
    background: '#2563eb',
    color: '#fff',
    minWidth: '120px',
  },
  savedNotice: {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    transform: 'translate(-50%,-50%)',
    background: 'rgba(0,0,0,0.75)',
    color: '#fff',
    padding: '12px 24px',
    borderRadius: '12px',
    fontSize: '15px',
    pointerEvents: 'none' as const,
  },
};

const App: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [mode, setMode] = useState<Mode>('camera');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedMsg, setSavedMsg] = useState('');

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 3840 },
          height: { ideal: 2160 },
        },
        audio: false,
      });
      streamRef.current = mediaStream;
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch {
      setError('Camera access denied.\nPlease allow camera permission in your browser settings, then refresh.');
    }
  }, []);

  useEffect(() => {
    startCamera();
    return stopCamera;
  }, [startCamera, stopCamera]);

  const capture = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    setCapturedImage(canvas.toDataURL('image/jpeg', 0.95));
    setMode('preview');
    stopCamera();
  }, [stopCamera]);

  const retake = useCallback(() => {
    setCapturedImage(null);
    setSavedMsg('');
    setMode('camera');
    startCamera();
  }, [startCamera]);

  const saveImage = useCallback(async () => {
    if (!capturedImage || saving) return;
    setSaving(true);
    const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filename = `scan_${ts}.jpg`;
    try {
      const res = await fetch(capturedImage);
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      if (typeof navigator.canShare === 'function' && navigator.canShare({ files: [file] })) {
        await navigator.share({ files: [file], title: 'Scanned Document' });
        setSavedMsg('Shared!');
      } else {
        throw new Error('share not supported');
      }
    } catch {
      // fallback: trigger download
      const a = document.createElement('a');
      a.href = capturedImage;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setSavedMsg('Saved!');
    }
    setSaving(false);
    setTimeout(() => setSavedMsg(''), 2500);
  }, [capturedImage, saving]);

  return (
    <div style={styles.root}>
      <div style={styles.header}>DOC SCANNER</div>

      <div style={styles.viewportArea}>
        {error ? (
          <div style={styles.errorBox}>{error}</div>
        ) : mode === 'camera' ? (
          <>
            <video
              ref={videoRef}
              style={styles.video}
              autoPlay
              playsInline
              muted
            />
            <div style={styles.overlay} />
          </>
        ) : (
          capturedImage && (
            <>
              <img src={capturedImage} alt="Captured document" style={styles.previewImg} />
              {savedMsg && <div style={styles.savedNotice}>{savedMsg}</div>}
            </>
          )
        )}
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      <div style={styles.controls}>
        {mode === 'camera' ? (
          <button
            style={styles.captureBtn}
            onClick={capture}
            aria-label="Capture document"
          >
            <div style={styles.captureBtnInner} />
          </button>
        ) : (
          <>
            <button
              style={{ ...styles.secondaryBtn, ...styles.retakeBtn }}
              onClick={retake}
            >
              Retake
            </button>
            <button
              style={{ ...styles.secondaryBtn, ...styles.saveBtn }}
              onClick={saveImage}
              disabled={saving}
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default App;
