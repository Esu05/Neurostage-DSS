/**
 * NeuroStage DSS — React Frontend (Fully Responsive)
 * Works on mobile (320px+), tablet, and desktop
 * Clinical Decision Support Interface for Alzheimer's Stage Classification
 * Model: Hybrid Dual-CNN (2×2 + 4×4 kernels) + Soft-Voting Ensemble (SVM+RF+KNN)
 * Paper: Zolfaghari et al., Scientific Reports (2025) 15:25342
 */

import { useState, useRef, useCallback, useEffect } from "react";

// ── Class colour palette ──────────────────────────────────────────────────────
const CLASS_CONFIG = {
  "Non-Demented": {
    color: "#22c55e", bg: "rgba(34,197,94,0.12)", border: "rgba(34,197,94,0.45)",
    glow: "0 0 24px rgba(34,197,94,0.35)", badge: "NEGATIVE", icon: "✦",
  },
  "Very Mild Demented": {
    color: "#f59e0b", bg: "rgba(245,158,11,0.12)", border: "rgba(245,158,11,0.45)",
    glow: "0 0 24px rgba(245,158,11,0.35)", badge: "STAGE I", icon: "◆",
  },
  "Mild Demented": {
    color: "#f97316", bg: "rgba(249,115,22,0.12)", border: "rgba(249,115,22,0.45)",
    glow: "0 0 24px rgba(249,115,22,0.35)", badge: "STAGE II", icon: "▲",
  },
  "Moderate Demented": {
    color: "#ef4444", bg: "rgba(239,68,68,0.12)", border: "rgba(239,68,68,0.45)",
    glow: "0 0 24px rgba(239,68,68,0.40)", badge: "STAGE III", icon: "●",
  },
};

// Uses VITE_API_URL env var in production, falls back to localhost for dev
const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000") + "/predict";

// ── useWindowWidth hook for responsive breakpoints ───────────────────────────
function useWindowWidth() {
  const [width, setWidth] = useState(typeof window !== "undefined" ? window.innerWidth : 1024);
  useEffect(() => {
    const handler = () => setWidth(window.innerWidth);
    window.addEventListener("resize", handler);
    return () => window.removeEventListener("resize", handler);
  }, []);
  return width;
}

// ── Global CSS ────────────────────────────────────────────────────────────────
const GLOBAL_STYLES = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { -webkit-text-size-adjust: 100%; }
  body { background: #020b18; overflow-x: hidden; }

  @keyframes spin        { to { transform: rotate(360deg); } }
  @keyframes pulse-ring  { 0%,100% { box-shadow: 0 0 0 0 rgba(96,165,250,0.25); } 50% { box-shadow: 0 0 0 6px rgba(96,165,250,0); } }
  @keyframes scanline    { 0% { transform: translateY(-100%); opacity:0; } 10%,90% { opacity:1; } 100% { transform: translateY(100%); opacity:0; } }
  @keyframes fadeInUp    { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:translateY(0); } }
  @keyframes fadeIn      { from { opacity:0; } to { opacity:1; } }

  .scan-anim::after {
    content:""; position:absolute; left:0; top:0; width:100%; height:3px;
    background: linear-gradient(90deg, transparent, #60a5fa, transparent);
    animation: scanline 1.8s ease-in-out infinite; pointer-events:none;
  }
  .fade-in-up { animation: fadeInUp 0.45s ease forwards; }
  .fade-in    { animation: fadeIn 0.35s ease forwards; }

  /* Touch-friendly tap targets */
  button { -webkit-tap-highlight-color: transparent; touch-action: manipulation; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
`;

// ── Tiny helpers ──────────────────────────────────────────────────────────────
const M = ({ children, s = {} }) => (
  <span style={{ fontFamily: "'IBM Plex Mono', monospace", ...s }}>{children}</span>
);

function PanelHeader({ dot = "#60a5fa", label, right }) {
  return (
    <div style={{ padding: "14px 16px", borderBottom: "1px solid rgba(255,255,255,0.06)", display: "flex", alignItems: "center", gap: "8px" }}>
      <div style={{ width: "6px", height: "6px", flexShrink: 0, borderRadius: "50%", background: dot, boxShadow: `0 0 6px ${dot}` }} />
      <M s={{ fontSize: "0.6rem", letterSpacing: "0.14em", color: "#64748b" }}>{label}</M>
      {right && <div style={{ marginLeft: "auto" }}>{right}</div>}
    </div>
  );
}

function Tag({ children, color = "#60a5fa" }) {
  return (
    <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.58rem", letterSpacing: "0.08em", color, border: `1px solid ${color}44`, background: `${color}11`, padding: "2px 7px", borderRadius: "3px", display: "inline-block", whiteSpace: "nowrap" }}>
      {children}
    </span>
  );
}

function StatCard({ value, label, color = "#60a5fa" }) {
  return (
    <div style={{ textAlign: "center", padding: "14px 10px", border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", background: "rgba(255,255,255,0.02)" }}>
      <div style={{ fontFamily: "'Syne', sans-serif", fontSize: "1.3rem", fontWeight: 800, color, marginBottom: "4px" }}>{value}</div>
      <M s={{ fontSize: "0.56rem", color: "#475569", letterSpacing: "0.08em", lineHeight: 1.5, whiteSpace: "pre-line" }}>{label}</M>
    </div>
  );
}

// ── Animated probability bar ──────────────────────────────────────────────────
function ProbBar({ label, value, config, delay = 0 }) {
  const [w, setW] = useState(0);
  useEffect(() => { const t = setTimeout(() => setW(value), delay); return () => clearTimeout(t); }, [value, delay]);
  return (
    <div style={{ marginBottom: "12px" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" }}>
        <M s={{ fontSize: "0.65rem", color: "#94a3b8", letterSpacing: "0.04em" }}>{label.toUpperCase()}</M>
        <M s={{ fontSize: "0.72rem", color: config.color, fontWeight: 700 }}>{value.toFixed(1)}%</M>
      </div>
      <div style={{ background: "rgba(255,255,255,0.05)", borderRadius: "4px", height: "8px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.06)" }}>
        <div style={{ height: "100%", width: `${w}%`, background: `linear-gradient(90deg,${config.color}99,${config.color})`, borderRadius: "4px", transition: "width 0.9s cubic-bezier(0.34,1.56,0.64,1)", boxShadow: `0 0 10px ${config.color}66` }} />
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "40px 0" }}>
      <div style={{ width: "36px", height: "36px", border: "3px solid rgba(96,165,250,0.15)", borderTop: "3px solid #60a5fa", borderRadius: "50%", animation: "spin 0.8s linear infinite" }} />
    </div>
  );
}

function IdlePlaceholder() {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "200px", gap: "14px", opacity: 0.4, padding: "24px" }}>
      <svg width="56" height="56" viewBox="0 0 72 72" fill="none">
        <circle cx="36" cy="36" r="34" stroke="#60a5fa" strokeWidth="1.5" strokeDasharray="4 3"/>
        <circle cx="36" cy="36" r="20" stroke="#60a5fa" strokeWidth="1" strokeDasharray="2 4"/>
        <circle cx="36" cy="36" r="5" fill="#60a5fa" opacity="0.5"/>
        <line x1="36" y1="2" x2="36" y2="14" stroke="#60a5fa" strokeWidth="1.5"/>
        <line x1="36" y1="58" x2="36" y2="70" stroke="#60a5fa" strokeWidth="1.5"/>
        <line x1="2" y1="36" x2="14" y2="36" stroke="#60a5fa" strokeWidth="1.5"/>
        <line x1="58" y1="36" x2="70" y2="36" stroke="#60a5fa" strokeWidth="1.5"/>
      </svg>
      <M s={{ fontSize: "0.68rem", color: "#60a5fa", letterSpacing: "0.1em", textAlign: "center", lineHeight: 1.7 }}>
        AWAITING SCAN INPUT<br/><span style={{ opacity: 0.6 }}>— upload an MRI slice to begin —</span>
      </M>
    </div>
  );
}

function ProcessingState() {
  const [dot, setDot] = useState(0);
  useEffect(() => { const t = setInterval(() => setDot(d => (d + 1) % 4), 500); return () => clearInterval(t); }, []);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "200px", gap: "16px", padding: "24px" }}>
      <Spinner />
      <div style={{ textAlign: "center" }}>
        <M s={{ fontSize: "0.68rem", color: "#60a5fa", letterSpacing: "0.08em", display: "block", marginBottom: "4px" }}>PROCESSING DUAL-CNN STREAMS{".".repeat(dot)}</M>
        <M s={{ fontSize: "0.6rem", color: "#475569", letterSpacing: "0.06em" }}>OPTIMISING SOFT-VOTING GRID</M>
      </div>
    </div>
  );
}

function ResultPanel({ result }) {
  const cfg = CLASS_CONFIG[result.predicted_label] || CLASS_CONFIG["Non-Demented"];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      {/* Verdict */}
      <div style={{ border: `1px solid ${cfg.border}`, background: cfg.bg, borderRadius: "12px", padding: "18px", boxShadow: cfg.glow, textAlign: "center" }}>
        <M s={{ fontSize: "0.6rem", color: cfg.color, letterSpacing: "0.16em", display: "block", marginBottom: "6px", opacity: 0.8 }}>DIAGNOSTIC VERDICT</M>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "10px", marginBottom: "6px" }}>
          <span style={{ color: cfg.color, fontSize: "1.2rem" }}>{cfg.icon}</span>
          <span style={{ fontFamily: "'Syne', sans-serif", fontSize: "1.3rem", fontWeight: 800, color: cfg.color }}>{result.predicted_label}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "10px", flexWrap: "wrap" }}>
          <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.6rem", color: "#0f172a", background: cfg.color, padding: "2px 10px", borderRadius: "3px", fontWeight: 700, letterSpacing: "0.1em" }}>{cfg.badge}</span>
          <M s={{ fontSize: "0.68rem", color: "#94a3b8" }}>confidence: <span style={{ color: cfg.color }}>{result.confidence.toFixed(1)}%</span></M>
        </div>
      </div>
      {/* Bars */}
      <div style={{ border: "1px solid rgba(255,255,255,0.07)", background: "rgba(255,255,255,0.02)", borderRadius: "12px", padding: "16px" }}>
        <M s={{ fontSize: "0.58rem", color: "#475569", letterSpacing: "0.12em", display: "block", marginBottom: "14px" }}>PROBABILITY DISTRIBUTION — ALL CLASSES</M>
        {Object.entries(result.probabilities).map(([label, prob], i) => (
          <ProbBar key={label} label={label} value={prob} config={CLASS_CONFIG[label] || { color: "#60a5fa" }} delay={i * 120} />
        ))}
      </div>
      <M s={{ fontSize: "0.56rem", color: "#334155", letterSpacing: "0.07em", textAlign: "center" }}>DUAL-CNN (2×2 + 4×4) → DENSE-128 → SOFT-VOTE (SVM · RF · KNN)</M>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DIAGNOSTIC PAGE
// ═══════════════════════════════════════════════════════════════════════════════
function DiagnosticPage() {
  const width = useWindowWidth();
  const isTablet = width >= 768;

  const [imageFile, setImageFile]   = useState(null);
  const [imageUrl, setImageUrl]     = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus]         = useState("idle");
  const [result, setResult]         = useState(null);
  const [errorMsg, setErrorMsg]     = useState("");
  const inputRef = useRef(null);

  const loadFile = useCallback((file) => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file); setImageUrl(URL.createObjectURL(file));
    setStatus("idle"); setResult(null); setErrorMsg("");
  }, []);

  const onDrop      = useCallback((e) => { e.preventDefault(); setIsDragging(false); loadFile(e.dataTransfer.files[0]); }, [loadFile]);
  const onDragOver  = (e) => { e.preventDefault(); setIsDragging(true); };
  const onDragLeave = () => setIsDragging(false);
  const onChange    = (e) => loadFile(e.target.files[0]);

  const runScan = async () => {
    if (!imageFile) return;
    setStatus("loading"); setResult(null); setErrorMsg("");
    const fd = new FormData();
    fd.append("file", imageFile);
    try {
      const resp = await fetch(API_URL, { method: "POST", body: fd });
      if (!resp.ok) { const e = await resp.json().catch(() => ({ detail: "Server error" })); throw new Error(e.detail || `HTTP ${resp.status}`); }
      setResult(await resp.json()); setStatus("success");
    } catch (e) {
      setErrorMsg(e.message || "Connection failed. Is the backend running on :8000?");
      setStatus("error");
    }
  };

  return (
    <main style={{ maxWidth: "1280px", margin: "0 auto", padding: isTablet ? "28px 24px 48px" : "16px 12px 40px" }}>
      <div style={{ display: "grid", gridTemplateColumns: isTablet ? "1fr 1fr" : "1fr", gap: "16px" }}>

        {/* ── LEFT: Upload panel ─────────────────────────────────────────── */}
        <div style={{ border: "1px solid rgba(255,255,255,0.07)", borderRadius: "14px", background: "rgba(255,255,255,0.02)", overflow: "hidden" }}>
          <PanelHeader label="MRI INPUT — SCAN UPLOAD" />
          <div style={{ padding: "16px" }}>

            {/* Drop zone */}
            <div
              onDrop={onDrop} onDragOver={onDragOver} onDragLeave={onDragLeave}
              onClick={() => inputRef.current?.click()}
              style={{
                position: "relative", height: isTablet ? "240px" : "200px",
                border: `1.5px dashed ${isDragging ? "rgba(96,165,250,0.7)" : "rgba(255,255,255,0.1)"}`,
                borderRadius: "10px", cursor: "pointer", overflow: "hidden",
                transition: "border-color 0.2s, background 0.2s",
                background: isDragging ? "rgba(96,165,250,0.05)" : "rgba(0,0,0,0.15)",
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              {imageUrl ? (
                <img src={imageUrl} alt="MRI preview" style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain", borderRadius: "6px" }} />
              ) : (
                <div style={{ textAlign: "center", pointerEvents: "none", padding: "16px" }}>
                  <svg width="36" height="36" viewBox="0 0 40 40" fill="none" style={{ margin: "0 auto 10px", display: "block", opacity: 0.35 }}>
                    <rect x="4" y="4" width="32" height="32" rx="6" stroke="#60a5fa" strokeWidth="1.5" strokeDasharray="3 2"/>
                    <path d="M20 14v12M14 20h12" stroke="#60a5fa" strokeWidth="2" strokeLinecap="round"/>
                  </svg>
                  <M s={{ fontSize: "0.68rem", color: "#475569", letterSpacing: "0.08em", display: "block" }}>TAP TO UPLOAD MRI SLICE</M>
                  <M s={{ fontSize: "0.6rem", color: "#334155", marginTop: "4px", display: "block" }}>JPG / PNG · drag & drop supported</M>
                </div>
              )}
              <input ref={inputRef} type="file" accept="image/*" onChange={onChange} style={{ display: "none" }} capture="environment" />
            </div>

            {/* File info */}
            {imageFile && (
              <div style={{ marginTop: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
                <div style={{ width: "5px", height: "5px", flexShrink: 0, borderRadius: "50%", background: "#22c55e", boxShadow: "0 0 5px #22c55e" }} />
                <M s={{ fontSize: "0.6rem", color: "#64748b", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "75%" }}>{imageFile.name}</M>
                <M s={{ fontSize: "0.58rem", color: "#334155", marginLeft: "auto", flexShrink: 0 }}>{(imageFile.size / 1024).toFixed(0)} KB</M>
              </div>
            )}

            {/* Run button */}
            <button
              onClick={runScan} disabled={!imageFile || status === "loading"}
              style={{
                marginTop: "12px", width: "100%", padding: "14px 0",
                borderRadius: "8px", border: "none",
                cursor: imageFile && status !== "loading" ? "pointer" : "not-allowed",
                fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.72rem",
                letterSpacing: "0.12em", fontWeight: 700, color: "#020b18",
                background: imageFile && status !== "loading" ? "linear-gradient(135deg,#3b82f6,#60a5fa)" : "rgba(255,255,255,0.08)",
                opacity: imageFile && status !== "loading" ? 1 : 0.4,
                transition: "all 0.2s",
                animation: imageFile && status !== "loading" ? "pulse-ring 2s infinite" : "none",
                display: "flex", alignItems: "center", justifyContent: "center", gap: "10px",
              }}
            >
              {status === "loading"
                ? <><div style={{ width: "13px", height: "13px", border: "2px solid rgba(2,11,24,0.3)", borderTop: "2px solid #020b18", borderRadius: "50%", animation: "spin 0.7s linear infinite" }} />ANALYSING…</>
                : "▶  RUN DIAGNOSTIC SCAN"}
            </button>
          </div>

          {/* Error */}
          {status === "error" && (
            <div style={{ margin: "0 16px 16px", padding: "12px", borderRadius: "8px", border: "1px solid rgba(239,68,68,0.35)", background: "rgba(239,68,68,0.08)" }}>
              <M s={{ fontSize: "0.62rem", color: "#f87171", lineHeight: 1.6, display: "block" }}>⚠  {errorMsg}</M>
            </div>
          )}

          {/* Pipeline info */}
          <div style={{ margin: "0 16px 16px", padding: "10px 12px", borderRadius: "8px", background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}>
            <M s={{ fontSize: "0.56rem", color: "#334155", letterSpacing: "0.08em", lineHeight: 2, display: "block" }}>
              INPUT PIPELINE<br/>
              <span style={{ color: "#475569" }}>→ GRAYSCALE → 176×176 → [0,1] NORM<br/>→ CNN-FEATURE (128D) → ENSEMBLE VOTE<br/>→ ND | VMD | MID | MOD</span>
            </M>
          </div>
        </div>

        {/* ── RIGHT: Output panel ────────────────────────────────────────── */}
        <div style={{ border: "1px solid rgba(255,255,255,0.07)", borderRadius: "14px", background: "rgba(255,255,255,0.02)", overflow: "hidden", display: "flex", flexDirection: "column" }}>
          <PanelHeader
            dot={status === "success" ? "#22c55e" : "#60a5fa"}
            label="ANALYTICS OUTPUT"
            right={<M s={{ fontSize: "0.56rem", color: "#1e293b", letterSpacing: "0.08em" }}>{status === "success" ? "COMPLETE" : status === "loading" ? "RUNNING…" : "STANDBY"}</M>}
          />
          <div style={{ flex: 1, padding: "16px", position: "relative" }} className={status === "loading" ? "scan-anim" : ""}>
            {status === "idle"    && <IdlePlaceholder />}
            {status === "loading" && <ProcessingState />}
            {status === "error"   && <IdlePlaceholder />}
            {status === "success" && result && <ResultPanel result={result} />}
          </div>
          <div style={{ borderTop: "1px solid rgba(255,255,255,0.04)", padding: "8px 16px", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: "4px" }}>
            <M s={{ fontSize: "0.52rem", color: "#1e293b", letterSpacing: "0.08em" }}>ACCURACY: 99.30% · PAPER BASELINE</M>
            <M s={{ fontSize: "0.52rem", color: "#1e293b", letterSpacing: "0.08em" }}>SVM · RF · KNN → SOFT-VOTE</M>
          </div>
        </div>
      </div>
    </main>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ABOUT PAGE
// ═══════════════════════════════════════════════════════════════════════════════
function AboutPage() {
  const width = useWindowWidth();
  const isTablet = width >= 768;
  const isDesktop = width >= 1024;

  const Section = ({ title, children, delay = 0 }) => (
    <div className="fade-in-up" style={{ animationDelay: `${delay}ms`, animationFillMode: "both", marginBottom: "24px" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "12px" }}>
        <div style={{ width: "3px", height: "16px", flexShrink: 0, background: "linear-gradient(180deg,#60a5fa,#3b82f6)", borderRadius: "2px" }} />
        <M s={{ fontSize: "0.62rem", letterSpacing: "0.16em", color: "#60a5fa" }}>{title}</M>
      </div>
      {children}
    </div>
  );

  const architectureSteps = [
    { step: "01", label: "INPUT",    detail: "176×176 grayscale MRI slice",                            color: "#60a5fa" },
    { step: "02", label: "NETWORK 1", detail: "2×2 kernels · Max-Pooling · Attention → 3,872 feats",  color: "#818cf8" },
    { step: "03", label: "NETWORK 2", detail: "4×4 kernels · Avg-Pooling · Attention → 3,872 feats",  color: "#a78bfa" },
    { step: "04", label: "FUSION",   detail: "Concatenate → Dense(512) → Dense(128)",                 color: "#c084fc" },
    { step: "05", label: "ENSEMBLE", detail: "SVM + RF + KNN → Soft-Voting",                          color: "#e879f9" },
    { step: "06", label: "OUTPUT",   detail: "ND · VMD · MID · MOD with probabilities",               color: "#f472b6" },
  ];

  const classInfo = [
    { short: "ND",  full: "Non-Demented",      desc: "Normal cognitive function. No signs of atrophy or memory impairment.",                                    color: "#22c55e", badge: "NEGATIVE"  },
    { short: "VMD", full: "Very Mild Demented", desc: "Earliest detectable stage. Subtle hippocampal changes. Mild forgetfulness.",                              color: "#f59e0b", badge: "STAGE I"   },
    { short: "MID", full: "Mild Demented",      desc: "Noticeable temporal lobe atrophy. Memory loss affecting daily activities.",                               color: "#f97316", badge: "STAGE II"  },
    { short: "MOD", full: "Moderate Demented",  desc: "Widespread cortical atrophy. Significant cognitive decline across multiple brain regions.",               color: "#ef4444", badge: "STAGE III" },
  ];

  const metrics = [
    { value: "99.06%", label: "ENSEMBLE\nACCURACY",   color: "#22c55e" },
    { value: "99.33%", label: "WEIGHTED\nF1-SCORE",   color: "#60a5fa" },
    { value: "4,052K", label: "MODEL\nPARAMETERS",    color: "#a78bfa" },
    { value: "< 2s",   label: "INFERENCE\nTIME",      color: "#f59e0b" },
    { value: "10-fold", label: "CROSS\nVALIDATION",   color: "#e879f9" },
    { value: "6,400",  label: "DATASET\nIMAGES",      color: "#f97316" },
  ];

  const pad = isTablet ? "28px 24px 64px" : "16px 12px 48px";
  const twoCol = isTablet ? "1fr 1fr" : "1fr";

  return (
    <div style={{ maxWidth: "1280px", margin: "0 auto", padding: pad }}>

      {/* Hero */}
      <div className="fade-in-up" style={{ marginBottom: "28px", padding: isTablet ? "28px" : "18px", border: "1px solid rgba(96,165,250,0.2)", borderRadius: "14px", background: "linear-gradient(135deg,rgba(96,165,250,0.06),rgba(167,139,250,0.03))", position: "relative", overflow: "hidden" }}>
        <div style={{ position: "absolute", top: 0, right: 0, width: "200px", height: "200px", background: "radial-gradient(circle,rgba(96,165,250,0.06),transparent 70%)", pointerEvents: "none" }} />
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px", marginBottom: "12px" }}>
          <Tag color="#60a5fa">SCIENTIFIC REPORTS · 2025</Tag>
          <Tag color="#a78bfa">DOI: 10.1038/s41598-025-11743-y</Tag>
        </div>
        <h1 style={{ fontFamily: "'Syne', sans-serif", fontSize: isTablet ? "1.3rem" : "1.05rem", fontWeight: 800, color: "#f1f5f9", lineHeight: 1.4, marginBottom: "12px" }}>
          A Hybrid Learning Approach for MRI-Based Detection of Alzheimer's Disease Stages Using Dual CNNs and Ensemble Classifier
        </h1>
        <M s={{ fontSize: "0.65rem", color: "#64748b", lineHeight: 1.8, display: "block", marginBottom: "12px" }}>
          Sepideh Zolfaghari · Atra Joudaki · Yashar Sarbaz<br/>
          <span style={{ color: "#475569" }}>University of Tabriz, Dept. of Biomedical Engineering, Iran</span>
        </M>
        <p style={{ fontSize: "0.75rem", color: "#94a3b8", lineHeight: 1.7 }}>
          This app deploys the hybrid dual-CNN + ensemble classifier from this paper, enabling Alzheimer's disease staging from T1-weighted MRI images.
        </p>
      </div>

      {/* Metrics */}
      <Section title="MODEL PERFORMANCE METRICS" delay={80}>
        <div style={{ display: "grid", gridTemplateColumns: `repeat(${isDesktop ? 6 : isTablet ? 3 : 2}, 1fr)`, gap: "10px" }}>
          {metrics.map(m => <StatCard key={m.label} value={m.value} label={m.label} color={m.color} />)}
        </div>
      </Section>

      {/* Architecture + Network details */}
      <div style={{ display: "grid", gridTemplateColumns: twoCol, gap: "16px", marginBottom: "24px" }}>
        <Section title="PIPELINE ARCHITECTURE" delay={160}>
          <div style={{ border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", overflow: "hidden" }}>
            {architectureSteps.map((s, i) => (
              <div key={s.step} style={{ display: "flex", alignItems: "center", gap: "12px", padding: "10px 14px", borderBottom: i < 5 ? "1px solid rgba(255,255,255,0.04)" : "none", background: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent" }}>
                <M s={{ fontSize: "0.56rem", color: s.color, opacity: 0.6, minWidth: "18px" }}>{s.step}</M>
                <div style={{ width: "2px", height: "28px", background: s.color, borderRadius: "2px", opacity: 0.5, flexShrink: 0 }} />
                <div>
                  <M s={{ fontSize: "0.62rem", color: s.color, fontWeight: 700, display: "block", letterSpacing: "0.08em" }}>{s.label}</M>
                  <M s={{ fontSize: "0.56rem", color: "#475569", lineHeight: 1.5 }}>{s.detail}</M>
                </div>
              </div>
            ))}
          </div>
        </Section>

        <Section title="DUAL-CNN ARCHITECTURE" delay={200}>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {[
              { title: "NETWORK 1", subtitle: "2×2 KERNELS", color: "#60a5fa", rows: [["Conv2D ×2","8→16 filters, 2×2, ReLU"],["Max-Pooling","2×2 stride"],["Batch Norm + Dropout","0.4 rate"],["Conv2D","32 filters, 4×4, ReLU"],["Attention Layer","1×1 sigmoid"],["Flatten","→ 3,872 features"]] },
              { title: "NETWORK 2", subtitle: "4×4 KERNELS", color: "#a78bfa", rows: [["Conv2D","8 filters, 4×4, ReLU"],["Avg-Pooling","2×2 stride"],["Attention Layer","1×1 sigmoid"],["Conv2D ×2","16→32 filters, 4×4, ReLU"],["Dropout","0.4 rate"],["Flatten","→ 3,872 features"]] },
            ].map(net => (
              <div key={net.title} style={{ border: `1px solid ${net.color}33`, borderRadius: "10px", padding: "12px 14px", background: `${net.color}06` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
                  <M s={{ fontSize: "0.62rem", color: net.color, fontWeight: 700, letterSpacing: "0.08em" }}>{net.title}</M>
                  <Tag color={net.color}>{net.subtitle}</Tag>
                </div>
                {net.rows.map(([k, v]) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", padding: "3px 0", borderBottom: "1px solid rgba(255,255,255,0.03)", gap: "8px" }}>
                    <M s={{ fontSize: "0.58rem", color: "#64748b", flexShrink: 0 }}>{k}</M>
                    <M s={{ fontSize: "0.58rem", color: "#475569", textAlign: "right" }}>{v}</M>
                  </div>
                ))}
              </div>
            ))}
          </div>
        </Section>
      </div>

      {/* Disease Classes */}
      <Section title="ALZHEIMER'S DISEASE CLASSIFICATION STAGES" delay={240}>
        <div style={{ display: "grid", gridTemplateColumns: isDesktop ? "repeat(4,1fr)" : isTablet ? "repeat(2,1fr)" : "1fr", gap: "10px" }}>
          {classInfo.map(c => (
            <div key={c.short} style={{ border: `1px solid ${c.color}33`, borderRadius: "10px", padding: "14px", background: `${c.color}08` }}>
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "8px" }}>
                <span style={{ fontFamily: "'Syne', sans-serif", fontSize: "1.2rem", fontWeight: 800, color: c.color }}>{c.short}</span>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.56rem", color: "#0f172a", background: c.color, padding: "2px 7px", borderRadius: "3px", fontWeight: 700, letterSpacing: "0.08em" }}>{c.badge}</span>
              </div>
              <M s={{ fontSize: "0.65rem", color: c.color, display: "block", marginBottom: "5px", fontWeight: 700 }}>{c.full}</M>
              <p style={{ fontSize: "0.7rem", color: "#64748b", lineHeight: 1.6 }}>{c.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      {/* Dataset + Training */}
      <div style={{ display: "grid", gridTemplateColumns: twoCol, gap: "16px", marginBottom: "24px" }}>
        <Section title="DATASET INFORMATION" delay={300}>
          <div style={{ border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", overflow: "hidden", marginBottom: "10px" }}>
            <div style={{ padding: "10px 14px", background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
                {["CLASS","TOTAL","TRAIN","TEST"].map(h => <M key={h} s={{ fontSize: "0.55rem", color: "#334155", letterSpacing: "0.08em" }}>{h}</M>)}
              </div>
            </div>
            {[["ND","3,200","2,560","640"],["VMD","2,240","1,792","448"],["MID","896","716","180"],["MOD","64","51","13"]].map(([cls,...vals], i) => (
              <div key={cls} style={{ padding: "9px 14px", borderBottom: i < 3 ? "1px solid rgba(255,255,255,0.04)" : "none", background: i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent" }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
                  <M s={{ fontSize: "0.62rem", color: "#60a5fa" }}>{cls}</M>
                  {vals.map((v, j) => <M key={j} s={{ fontSize: "0.62rem", color: "#64748b" }}>{v}</M>)}
                </div>
              </div>
            ))}
            <div style={{ padding: "9px 14px", background: "rgba(96,165,250,0.05)", borderTop: "1px solid rgba(96,165,250,0.1)" }}>
              <M s={{ fontSize: "0.58rem", color: "#475569" }}>SMOTE → 2,560 per class (balanced)</M>
            </div>
          </div>
          <p style={{ fontSize: "0.7rem", color: "#475569", lineHeight: 1.65 }}>
            Source: Kaggle — Alzheimer MRI 4 Classes Dataset (Pinamonti, 2021). T1-weighted slices from 200 anonymous patients, 32 slices each.
          </p>
        </Section>

        <Section title="TRAINING CONFIGURATION" delay={340}>
          <div style={{ border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", padding: "14px", display: "flex", flexDirection: "column", gap: "8px" }}>
            {[
              ["Optimizer","Adam (lr = 0.001)"],["Batch Size","100"],["Epochs","38"],
              ["Image Size","176 × 176 × 1"],["Train/Test Split","80% / 20%"],
              ["Augmentation","SMOTE (training only)"],["Regularization","Batch Norm + Dropout 0.4"],
              ["SVM Param","C = 1 (GridSearchCV)"],["KNN Param","K = 1 (GridSearchCV)"],
              ["RF Param","N = 30 estimators"],["CV Strategy","10-fold Stratified"],
            ].map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", padding: "5px 0", borderBottom: "1px solid rgba(255,255,255,0.04)", gap: "8px" }}>
                <M s={{ fontSize: "0.6rem", color: "#475569", flexShrink: 0 }}>{k}</M>
                <M s={{ fontSize: "0.6rem", color: "#94a3b8", textAlign: "right" }}>{v}</M>
              </div>
            ))}
          </div>
        </Section>
      </div>

      {/* Comparison table — scrollable on mobile */}
      <Section title="COMPARISON WITH STATE-OF-THE-ART METHODS" delay={380}>
        <div style={{ overflowX: "auto", WebkitOverflowScrolling: "touch", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.07)" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "520px" }}>
            <thead>
              <tr style={{ background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}>
                {["METHOD","DATASET","F1-SCORE","ACCURACY"].map(h => (
                  <th key={h} style={{ padding: "10px 14px", textAlign: "left" }}>
                    <M s={{ fontSize: "0.56rem", color: "#334155", letterSpacing: "0.1em" }}>{h}</M>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ["AlexNet (Al-Adhaileh)","Kaggle","94.12%","94.53%",false],
                ["VGG-19 (Assmi et al.)","Kaggle","78.50%","92.86%",false],
                ["SMOTE+DEMNET (Murugan)","Kaggle","95.50%","95.23%",false],
                ["Lightweight CNN (El-Latif)","Kaggle","95.90%","95.93%",false],
                ["Deep CNN+Ensemble (Balasundaram)","Kaggle","95.50%","94.10%",false],
                ["ResNet50+EfficientNet (Al Rahbani)","Kaggle","99.49%","99.41%",false],
                ["Dual CNNs+Attention+Ensemble ★","Kaggle","99.33%","99.06%",true],
              ].map(([method, ds, f1, acc, hi], i) => (
                <tr key={method} style={{ background: hi ? "rgba(96,165,250,0.07)" : i % 2 === 0 ? "rgba(255,255,255,0.01)" : "transparent", borderBottom: i < 6 ? "1px solid rgba(255,255,255,0.04)" : "none" }}>
                  <td style={{ padding: "9px 14px" }}><M s={{ fontSize: "0.62rem", color: hi ? "#60a5fa" : "#64748b", fontWeight: hi ? 700 : 400 }}>{method}</M></td>
                  <td style={{ padding: "9px 14px" }}><M s={{ fontSize: "0.62rem", color: "#475569" }}>{ds}</M></td>
                  <td style={{ padding: "9px 14px" }}><M s={{ fontSize: "0.62rem", color: hi ? "#22c55e" : "#475569", fontWeight: hi ? 700 : 400 }}>{f1}</M></td>
                  <td style={{ padding: "9px 14px" }}><M s={{ fontSize: "0.62rem", color: hi ? "#22c55e" : "#475569", fontWeight: hi ? 700 : 400 }}>{acc}</M></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* About app + Citation */}
      <div style={{ display: "grid", gridTemplateColumns: twoCol, gap: "16px" }}>
        <Section title="ABOUT THIS APPLICATION" delay={440}>
          <div style={{ border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", padding: "14px", display: "flex", flexDirection: "column", gap: "10px" }}>
            {[
              ["BACKEND","FastAPI (Python 3.10+) · Uvicorn ASGI"],
              ["FRONTEND","React 18 · Vite · Custom CSS"],
              ["ML FRAMEWORK","TensorFlow / Keras 3.x + scikit-learn"],
              ["INFERENCE","CPU-based (TF 2.21)"],
              ["API","POST /predict — multipart/form-data"],
              ["MODEL VERSION","v2026.1 — Fusion CNN + VotingClassifier"],
            ].map(([k, v]) => (
              <div key={k}>
                <M s={{ fontSize: "0.56rem", color: "#334155", letterSpacing: "0.1em", display: "block", marginBottom: "2px" }}>{k}</M>
                <M s={{ fontSize: "0.62rem", color: "#64748b" }}>{v}</M>
              </div>
            ))}
            <div style={{ marginTop: "4px", padding: "10px", border: "1px solid rgba(239,68,68,0.2)", borderRadius: "8px", background: "rgba(239,68,68,0.05)" }}>
              <M s={{ fontSize: "0.58rem", color: "#ef4444", lineHeight: 1.7, display: "block" }}>
                ⚠ RESEARCH USE ONLY — Not a certified medical device. Must not replace professional clinical diagnosis.
              </M>
            </div>
          </div>
        </Section>

        <Section title="CITE THIS WORK" delay={480}>
          <div style={{ border: "1px solid rgba(96,165,250,0.15)", borderRadius: "10px", padding: "14px", background: "rgba(96,165,250,0.03)" }}>
            <M s={{ fontSize: "0.58rem", color: "#334155", letterSpacing: "0.1em", display: "block", marginBottom: "10px" }}>APA CITATION</M>
            <p style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.62rem", color: "#64748b", lineHeight: 1.8, marginBottom: "14px" }}>
              Zolfaghari, S., Joudaki, A., & Sarbaz, Y. (2025). A hybrid learning approach for MRI-based detection of alzheimer's disease stages using dual CNNs and ensemble classifier. <em style={{ color: "#94a3b8" }}>Scientific Reports, 15</em>, 25342.
            </p>
            <div style={{ padding: "10px", borderRadius: "8px", background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.05)", marginBottom: "12px" }}>
              <M s={{ fontSize: "0.56rem", color: "#475569", lineHeight: 1.9, display: "block" }}>
                DOI: 10.1038/s41598-025-11743-y<br/>
                Published: 14 July 2025<br/>
                Journal: Scientific Reports (Nature Portfolio)<br/>
                License: CC BY-NC-ND 4.0
              </M>
            </div>
            <a href="https://doi.org/10.1038/s41598-025-11743-y" target="_blank" rel="noopener noreferrer"
              style={{ display: "block", textAlign: "center", padding: "10px", borderRadius: "6px", border: "1px solid rgba(96,165,250,0.3)", background: "rgba(96,165,250,0.08)", fontFamily: "'IBM Plex Mono', monospace", fontSize: "0.62rem", color: "#60a5fa", letterSpacing: "0.08em", textDecoration: "none" }}>
              ↗ VIEW PAPER ON NATURE.COM
            </a>
          </div>
        </Section>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT APP
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [page, setPage] = useState("diagnostic");
  const width = useWindowWidth();
  const isMobile = width < 640;

  return (
    <>
      <link rel="preconnect" href="https://fonts.googleapis.com" />
      <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
      <link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=IBM+Plex+Mono:wght@400;500;700&display=swap" rel="stylesheet" />
      <style>{GLOBAL_STYLES}</style>

      <div style={{ minHeight: "100vh", background: "#020b18", color: "#e2e8f0" }}>

        {/* ── Header ──────────────────────────────────────────────────────── */}
        <header style={{ borderBottom: "1px solid rgba(96,165,250,0.12)", background: "rgba(2,11,24,0.92)", backdropFilter: "blur(12px)", position: "sticky", top: 0, zIndex: 100, padding: isMobile ? "0 14px" : "0 24px" }}>
          <div style={{ maxWidth: "1280px", margin: "0 auto", display: "flex", alignItems: "center", justifyContent: "space-between", height: isMobile ? "54px" : "60px", gap: "10px" }}>

            {/* Logo */}
            <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
              <svg width="24" height="24" viewBox="0 0 28 28" fill="none">
                <circle cx="14" cy="14" r="13" stroke="#60a5fa" strokeWidth="1.2"/>
                <path d="M8 14 Q10 8 14 10 Q18 8 20 14 Q18 20 14 18 Q10 20 8 14Z" stroke="#60a5fa" strokeWidth="1" fill="rgba(96,165,250,0.1)"/>
                <circle cx="14" cy="14" r="2" fill="#60a5fa" opacity="0.6"/>
              </svg>
              <span style={{ fontFamily: "'Syne', sans-serif", fontSize: isMobile ? "1rem" : "1.1rem", fontWeight: 800, color: "#f1f5f9", letterSpacing: "0.03em", whiteSpace: "nowrap" }}>
                Neuro<span style={{ color: "#60a5fa" }}>Stage</span>
                {!isMobile && <span style={{ color: "#475569", fontWeight: 400, marginLeft: "5px", fontSize: "0.8rem" }}>DSS</span>}
              </span>
            </div>

            {/* Nav */}
            <nav style={{ display: "flex", alignItems: "center", gap: "2px" }}>
              {[
                { id: "diagnostic", label: isMobile ? "▶ SCAN" : "▶  DIAGNOSTIC" },
                { id: "about",      label: isMobile ? "ℹ INFO" : "ℹ  ABOUT" },
              ].map(({ id, label }) => (
                <button key={id} onClick={() => setPage(id)} style={{
                  fontFamily: "'IBM Plex Mono', monospace", fontSize: isMobile ? "0.6rem" : "0.62rem",
                  letterSpacing: "0.1em", padding: isMobile ? "6px 10px" : "6px 14px",
                  borderRadius: "6px", border: "none", cursor: "pointer", transition: "all 0.2s",
                  background: page === id ? "rgba(96,165,250,0.15)" : "transparent",
                  color: page === id ? "#60a5fa" : "#475569",
                  borderBottom: page === id ? "1px solid #60a5fa" : "1px solid transparent",
                }}>{label}</button>
              ))}
            </nav>

            {/* Badge — hidden on small mobile */}
            {!isMobile && (
              <M s={{ fontSize: "0.6rem", color: "#60a5fa", border: "1px solid rgba(96,165,250,0.3)", background: "rgba(96,165,250,0.07)", padding: "3px 10px", borderRadius: "4px", letterSpacing: "0.08em", flexShrink: 0, whiteSpace: "nowrap" }}>
                MODEL v2026.1
              </M>
            )}
          </div>
        </header>

        {/* Sub-header — scrollable on tiny screens */}
        <div style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", padding: isMobile ? "8px 14px" : "8px 24px", background: "rgba(255,255,255,0.01)", overflowX: "auto", whiteSpace: "nowrap" }}>
          <div style={{ maxWidth: "1280px", margin: "0 auto", display: "flex", alignItems: "center", gap: "8px" }}>
            <M s={{ fontSize: "0.54rem", color: "#334155", letterSpacing: "0.1em" }}>AD STAGING SYSTEM</M>
            <span style={{ color: "#1e293b" }}>·</span>
            <M s={{ fontSize: "0.54rem", color: "#334155", letterSpacing: "0.08em" }}>ZOLFAGHARI ET AL. 2025</M>
            <span style={{ color: "#1e293b" }}>·</span>
            <M s={{ fontSize: "0.54rem", color: "#334155", letterSpacing: "0.08em" }}>99.06% ACCURACY</M>
          </div>
        </div>

        {/* Page */}
        <div key={page} className="fade-in">
          {page === "diagnostic" && <DiagnosticPage />}
          {page === "about"      && <AboutPage />}
        </div>

        {/* ── Disclaimer Footer ────────────────────────────────────────────── */}
        <footer style={{ borderTop: "1px solid rgba(239,68,68,0.15)", background: "rgba(239,68,68,0.03)", padding: isMobile ? "20px 14px" : "24px 32px" }}>
          <div style={{ maxWidth: "1280px", margin: "0 auto" }}>

            {/* Warning banner */}
            <div style={{ display: "flex", alignItems: "flex-start", gap: "12px", padding: isMobile ? "14px" : "16px 20px", border: "1px solid rgba(239,68,68,0.25)", borderRadius: "10px", background: "rgba(239,68,68,0.06)", marginBottom: "20px" }}>
              {/* Icon */}
              <div style={{ flexShrink: 0, width: "32px", height: "32px", borderRadius: "50%", border: "1.5px solid rgba(239,68,68,0.5)", display: "flex", alignItems: "center", justifyContent: "center" }}>
                <span style={{ color: "#ef4444", fontSize: "0.9rem", lineHeight: 1 }}>⚠</span>
              </div>
              <div>
                <M s={{ fontSize: "0.65rem", color: "#ef4444", fontWeight: 700, letterSpacing: "0.14em", display: "block", marginBottom: "6px" }}>
                  EDUCATIONAL & RESEARCH USE ONLY — NOT FOR CLINICAL DIAGNOSIS
                </M>
                <p style={{ fontSize: isMobile ? "0.68rem" : "0.72rem", color: "#94a3b8", lineHeight: 1.75 }}>
                  NeuroStage DSS is a <strong style={{ color: "#cbd5e1" }}>research prototype</strong> built to demonstrate the hybrid Dual-CNN + Ensemble Classifier methodology described in Zolfaghari et al. (2025). This application is intended solely for <strong style={{ color: "#cbd5e1" }}>educational purposes, academic study, and technical demonstration</strong>.
                </p>
              </div>
            </div>

            {/* Three disclaimer points */}
            <div style={{ display: "grid", gridTemplateColumns: isMobile ? "1fr" : "repeat(3, 1fr)", gap: "12px", marginBottom: "20px" }}>
              {[
                {
                  icon: "🚫",
                  title: "NOT A MEDICAL DEVICE",
                  body: "This tool has not been reviewed, approved, or certified by any medical regulatory authority (FDA, CE, CDSCO, or equivalent). It must not be used to make or influence any clinical decision.",
                },
                {
                  icon: "👨‍⚕️",
                  title: "CONSULT A PROFESSIONAL",
                  body: "Any concern regarding Alzheimer's disease or cognitive decline must be evaluated by a qualified neurologist or physician. AI predictions cannot substitute professional clinical assessment.",
                },
                {
                  icon: "📊",
                  title: "RESEARCH LIMITATIONS",
                  body: "The model was trained and evaluated at slice-level on the Kaggle dataset. Performance may not generalise to different MRI scanners, acquisition protocols, or patient demographics.",
                },
              ].map(({ icon, title, body }) => (
                <div key={title} style={{ padding: "14px", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "10px", background: "rgba(255,255,255,0.02)" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "8px" }}>
                    <span style={{ fontSize: "1rem" }}>{icon}</span>
                    <M s={{ fontSize: "0.58rem", color: "#ef4444", fontWeight: 700, letterSpacing: "0.12em" }}>{title}</M>
                  </div>
                  <p style={{ fontSize: "0.68rem", color: "#64748b", lineHeight: 1.7 }}>{body}</p>
                </div>
              ))}
            </div>

            {/* Bottom bar */}
            <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: "8px", paddingTop: "16px", borderTop: "1px solid rgba(255,255,255,0.04)" }}>
              <M s={{ fontSize: "0.56rem", color: "#1e293b", letterSpacing: "0.08em" }}>
                © 2025 NEUROSTAGE DSS · BUILT FOR ACADEMIC DEMONSTRATION
              </M>
              <div style={{ display: "flex", gap: "16px", flexWrap: "wrap" }}>
                <M s={{ fontSize: "0.56rem", color: "#1e293b", letterSpacing: "0.08em" }}>
                  PAPER: DOI 10.1038/s41598-025-11743-y
                </M>
                <M s={{ fontSize: "0.56rem", color: "#1e293b", letterSpacing: "0.08em" }}>
                  LICENSE: CC BY-NC-ND 4.0
                </M>
              </div>
            </div>

          </div>
        </footer>

      </div>
    </>
  );
}