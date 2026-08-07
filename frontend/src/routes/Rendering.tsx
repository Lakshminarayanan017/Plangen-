/**
 * Rendering.tsx — the drawing board.
 *
 * A sheet pinned to a board, drawing itself, with a small crew of architects
 * working along the line currently being struck.
 *
 * The important design decision: this is NOT a decorative loader. Each of the
 * five passes maps to a real pipeline stage and only advances when the
 * backend advances, because /pipeline/status reports a step index and a live
 * log. What gets drawn in each pass is what that stage actually computes:
 *
 *   1 PARSE     the plot boundary          — the brief becomes a site
 *   2 MATCH     setbacks + the module grid — measured against 4,983 plans
 *   3 ENRICH    zone blocks                — rooms get sizes and positions
 *   4 GENERATE  the walls                  — space is carved
 *   5 RENDER    doors, swings, dimensions  — the sheet is annotated
 *
 * So the animation is honest: if step 4 takes eleven seconds, the walls take
 * eleven seconds. Motion here is doing the "is the system working?" job, and
 * it answers with the truth rather than a spinner.
 *
 * The crew is four figures, not forty. A crowd reads as noise and costs frame
 * budget; four reading as a team is the same idea, legible.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { api, STAGES, type PipelineStatus } from "../lib/api";
import { SheetFrame, TitleBlock } from "../components/Sheet";
import { navigate, type Brief } from "../App";

const POLL_MS = 900;

export default function Rendering({ runId, brief }: { runId: string; brief: Brief }) {
  const reduced = useReducedMotion() ?? false;
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;

    const tick = async () => {
      try {
        const s = await api.pipelineStatus(runId);
        if (!alive) return;
        setStatus(s);
        if (s.status === "complete") {
          // Let the last pass finish drawing before handing over the sheet.
          window.setTimeout(() => alive && navigate(`/sheet/${runId}`), 1400);
          return;
        }
        if (s.status === "error") {
          setError(s.error || "The pipeline stopped with an error.");
          return;
        }
        timer.current = window.setTimeout(tick, POLL_MS);
      } catch (e) {
        if (!alive) return;
        setError(e instanceof Error ? e.message : String(e));
      }
    };

    void tick();
    return () => {
      alive = false;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [runId]);

  const step = status?.step ?? 1;
  const done = status?.status === "complete";
  const logs = status?.logs ?? [];

  return (
    <div className="sheet-grid min-h-screen p-3 sm:p-5">
      <div className="mx-auto grid max-w-6xl gap-4 lg:grid-cols-[1fr_280px]">
        {/* ── the board ─────────────────────────────────────────────── */}
        <SheetFrame className="relative min-h-[62vh] overflow-hidden">
          <div className="absolute inset-0 p-4 sm:p-8">
            <DrawingBoard step={done ? 6 : step} reduced={reduced} />
          </div>

          <div className="pointer-events-none absolute bottom-3 left-4 right-4 flex items-end justify-between">
            <span className="annot">Run {runId}</span>
            <span className="annot" style={{ color: "var(--accent)" }}>
              {done ? "Sheet complete" : `Pass ${Math.min(step, 5)} of 5`}
            </span>
          </div>
        </SheetFrame>

        {/* ── the docket ────────────────────────────────────────────── */}
        <aside className="flex flex-col gap-4">
          <TitleBlock
            title="IN PROGRESS"
            sheet={`RUN ${runId.slice(-6)}`}
            rows={[
              { label: "Plot", value: brief.plot ?? "—" },
              { label: "Facing", value: brief.facing ?? "—" },
              { label: "Floors", value: brief.floors ?? "—" },
              {
                label: "Stage",
                value: done ? "Complete" : STAGES[Math.min(step, 5) - 1].code,
                live: true,
              },
            ]}
          />

          <ol className="border" style={{ borderColor: "var(--rule)" }}>
            {STAGES.map((s) => {
              const state = done || step > s.step ? "done" : step === s.step ? "live" : "todo";
              return (
                <li
                  key={s.step}
                  className="flex items-center gap-3 border-b px-3 py-2.5 last:border-b-0"
                  style={{ borderColor: "var(--rule-2)" }}
                >
                  <StageMark state={state} />
                  <span
                    className="text-[13px]"
                    style={{
                      color: state === "todo" ? "var(--ink-3)" : "var(--ink)",
                      fontWeight: state === "live" ? 600 : 400,
                    }}
                  >
                    {s.label}
                  </span>
                </li>
              );
            })}
          </ol>

          {/* Real log lines, not invented reassurance. */}
          <div
            className="max-h-48 overflow-y-auto border p-3"
            style={{ borderColor: "var(--rule)" }}
            aria-live="polite"
            aria-atomic="false"
          >
            <p className="annot mb-2">Log</p>
            {logs.length === 0 && <p className="dim">Waiting for the first pass…</p>}
            {logs.slice(-14).map((l, i) => (
              <p key={i} className="dim mb-1 leading-snug" style={{ fontSize: 10.5 }}>
                {l}
              </p>
            ))}
          </div>

          {error && (
            <div
              className="border-l-2 py-2 pl-3"
              style={{ borderColor: "var(--accent)" }}
              role="alert"
            >
              <p className="annot mb-1" style={{ color: "var(--accent)" }}>
                Stopped
              </p>
              <p className="text-[13px]" style={{ color: "var(--ink-2)" }}>
                {error}
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

/* ── stage marker ────────────────────────────────────────────────────── */

function StageMark({ state }: { state: "done" | "live" | "todo" }) {
  if (state === "done") {
    return (
      <svg width="12" height="12" aria-hidden="true" style={{ color: "var(--ink-2)" }}>
        <path d="M1 6 L5 10 L11 2" fill="none" stroke="currentColor" strokeWidth="1.4" />
      </svg>
    );
  }
  if (state === "live") {
    return (
      <motion.span
        className="block h-2 w-2"
        style={{ background: "var(--accent)" }}
        animate={{ opacity: [1, 0.25, 1] }}
        transition={{ duration: 1.4, repeat: Infinity }}
        aria-hidden="true"
      />
    );
  }
  return (
    <span
      className="block h-2 w-2 border"
      style={{ borderColor: "var(--rule)" }}
      aria-hidden="true"
    />
  );
}

/* ── the drawing itself ──────────────────────────────────────────────── */

const W = 620;
const H = 420;

/** Geometry for the schematic being struck. Deterministic, not random. */
const PLOT = { x: 90, y: 60, w: 440, h: 300 };
const SETBACK = 26;
const ROOMS = [
  { x: 116, y: 86, w: 190, h: 130, label: "LIVING" },
  { x: 306, y: 86, w: 130, h: 130, label: "KITCHEN" },
  { x: 436, y: 86, w: 68, h: 130, label: "UTIL" },
  { x: 116, y: 216, w: 150, h: 118, label: "BED 1" },
  { x: 266, y: 216, w: 140, h: 118, label: "BED 2" },
  { x: 406, y: 216, w: 98, h: 118, label: "BATH" },
];

function DrawingBoard({ step, reduced }: { step: number; reduced: boolean }) {
  // Where the crew works during each pass — the frontier of that pass's line.
  const crew = useMemo(
    () => [
      { id: 0, byStep: [[PLOT.x, PLOT.y], [PLOT.x + SETBACK, PLOT.y + SETBACK], [140, 150], [300, 86], [180, 216]] },
      { id: 1, byStep: [[PLOT.x + PLOT.w, PLOT.y], [PLOT.x + PLOT.w - SETBACK, PLOT.y + SETBACK], [370, 150], [436, 150], [400, 216]] },
      { id: 2, byStep: [[PLOT.x + PLOT.w, PLOT.y + PLOT.h], [PLOT.x + PLOT.w - SETBACK, PLOT.y + PLOT.h - SETBACK], [470, 280], [306, 216], [266, 334]] },
      { id: 3, byStep: [[PLOT.x, PLOT.y + PLOT.h], [PLOT.x + SETBACK, PLOT.y + PLOT.h - SETBACK], [200, 280], [116, 216], [116, 300]] },
    ],
    [],
  );

  const pass = Math.min(Math.max(step, 1), 5);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-full w-full"
      style={{ color: "var(--ink)" }}
      role="img"
      aria-label={`Floor plan being drawn — pass ${pass} of 5`}
    >
      <defs>
        <pattern id="brd-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="currentColor" strokeWidth="0.6" opacity="0.3" />
        </pattern>
      </defs>

      {/* board clips + tape */}
      {[[70, 42], [W - 70, 42], [70, H - 42], [W - 70, H - 42]].map(([cx, cy], i) => (
        <rect
          key={i}
          x={cx - 13}
          y={cy - 7}
          width="26"
          height="14"
          fill="none"
          stroke="currentColor"
          strokeWidth="0.8"
          opacity="0.45"
        />
      ))}

      {/* PASS 1 — the plot boundary */}
      <Stroke show={pass >= 1} reduced={reduced} length={2 * (PLOT.w + PLOT.h)} delay={0}>
        <rect
          x={PLOT.x}
          y={PLOT.y}
          width={PLOT.w}
          height={PLOT.h}
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
        />
      </Stroke>

      {/* PASS 2 — setbacks and the module grid */}
      <Stroke show={pass >= 2} reduced={reduced} length={2 * (PLOT.w + PLOT.h)} delay={0.1}>
        <rect
          x={PLOT.x + SETBACK}
          y={PLOT.y + SETBACK}
          width={PLOT.w - SETBACK * 2}
          height={PLOT.h - SETBACK * 2}
          fill="none"
          stroke="currentColor"
          strokeWidth="0.8"
          strokeDasharray="7 4"
          opacity="0.8"
        />
      </Stroke>

      {/* PASS 3 — zone blocks */}
      {ROOMS.map((r, i) => (
        <Stroke
          key={`z${i}`}
          show={pass >= 3}
          reduced={reduced}
          length={2 * (r.w + r.h)}
          delay={i * 0.09}
        >
          <rect
            x={r.x}
            y={r.y}
            width={r.w}
            height={r.h}
            fill={i % 3 === 0 ? "url(#brd-hatch)" : "none"}
            stroke="currentColor"
            strokeWidth="0.7"
            opacity="0.55"
          />
        </Stroke>
      ))}

      {/* PASS 4 — the walls */}
      {ROOMS.map((r, i) => (
        <Stroke
          key={`w${i}`}
          show={pass >= 4}
          reduced={reduced}
          length={2 * (r.w + r.h)}
          delay={i * 0.11}
        >
          <rect
            x={r.x}
            y={r.y}
            width={r.w}
            height={r.h}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
          />
        </Stroke>
      ))}

      {/* PASS 5 — doors, swings, labels, dimensions */}
      <g
        style={{
          opacity: pass >= 5 ? 1 : 0,
          transition: "opacity 600ms var(--ease-out)",
        }}
      >
        {ROOMS.map((r, i) => (
          <g key={`d${i}`}>
            <path
              d={`M ${r.x + 16} ${r.y + r.h} a 20 20 0 0 1 20 -20`}
              fill="none"
              stroke="currentColor"
              strokeWidth="0.7"
              opacity="0.75"
            />
            <text
              x={r.x + r.w / 2}
              y={r.y + r.h / 2}
              textAnchor="middle"
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 9,
                letterSpacing: "0.12em",
              }}
              fill="currentColor"
              opacity="0.75"
            >
              {r.label}
            </text>
          </g>
        ))}
        {/* dimension run along the bottom */}
        <g opacity="0.8">
          <line x1={PLOT.x} y1={H - 26} x2={PLOT.x + PLOT.w} y2={H - 26} stroke="currentColor" strokeWidth="0.7" />
          <line x1={PLOT.x} y1={H - 32} x2={PLOT.x} y2={H - 20} stroke="currentColor" strokeWidth="0.7" />
          <line x1={PLOT.x + PLOT.w} y1={H - 32} x2={PLOT.x + PLOT.w} y2={H - 20} stroke="currentColor" strokeWidth="0.7" />
          <text
            x={PLOT.x + PLOT.w / 2}
            y={H - 30}
            textAnchor="middle"
            fill="currentColor"
            style={{ fontFamily: "var(--font-mono)", fontSize: 9 }}
          >
            PLOT WIDTH
          </text>
        </g>
      </g>

      {/* the crew */}
      {crew.map((c) => {
        const [x, y] = c.byStep[Math.min(pass, 5) - 1];
        return <Architect key={c.id} x={x} y={y} seed={c.id} reduced={reduced} />;
      })}
    </svg>
  );
}

/**
 * A line that draws itself. stroke-dashoffset is animated, which is a paint
 * operation on a short path — cheap, and the only honest way to make a line
 * appear to be struck rather than faded in.
 */
function Stroke({
  show,
  reduced,
  length,
  delay,
  children,
}: {
  show: boolean;
  reduced: boolean;
  length: number;
  delay: number;
  children: React.ReactNode;
}) {
  if (!show) return null;
  if (reduced) return <g style={{ opacity: 1 }}>{children}</g>;
  return (
    <g
      style={{
        strokeDasharray: length,
        strokeDashoffset: length,
        animation: `draw-in 1100ms var(--ease-out) ${delay}s forwards`,
      }}
    >
      {children}
    </g>
  );
}

/**
 * One of the crew: ~16px tall, kneeling at the board with a pen. Small enough
 * to read as a person and simple enough to cost nothing.
 */
function Architect({
  x,
  y,
  seed,
  reduced,
}: {
  x: number;
  y: number;
  seed: number;
  reduced: boolean;
}) {
  return (
    <motion.g
      initial={false}
      animate={{ x, y }}
      transition={{ type: "spring", stiffness: 55, damping: 16, delay: seed * 0.07 }}
      style={{ color: "var(--accent)" }}
      aria-hidden="true"
    >
      <g transform="translate(-5,-14)" stroke="currentColor" fill="none" strokeWidth="1.1" strokeLinecap="round">
        {/* head */}
        <circle cx="5" cy="3" r="2.4" />
        {/* torso, leaning into the work */}
        <path d="M5 5.4 L4 10" />
        {/* legs, kneeling */}
        <path d="M4 10 L1.5 13.5 M4 10 L7 12.5 L6 14" />
        {/* drawing arm — animates in short strokes */}
        <motion.path
          d="M4.4 6.8 L9 9"
          animate={reduced ? undefined : { rotate: [0, -9, 0, 6, 0] }}
          transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut", delay: seed * 0.3 }}
          style={{ originX: "4.4px", originY: "6.8px" }}
        />
        {/* trailing arm */}
        <path d="M4.6 6.6 L1.6 8.6" opacity="0.7" />
      </g>
    </motion.g>
  );
}
