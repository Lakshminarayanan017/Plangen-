/**
 * Wordmark.tsx — PLANGEN, drawn as floor plans.
 *
 * The signature. Each letter is built from rectangular "rooms" that abut, so
 * the shared edges read as interior partitions and the outline reads as an
 * external wall. Some cells are hatched, a few carry door swings, and the
 * stack is dimensioned down the margin — exactly how the letterform would be
 * documented if someone actually had to build it.
 *
 * This mark means something for a floor-plan generator and nothing for any
 * other product, which is the entire point of a signature element.
 */

import { useEffect, useRef, useState } from "react";

type Rect = [x: number, y: number, w: number, h: number];

/** Blocky 88x140 letterforms. Rooms, not curves — curves don't carve. */
const GLYPHS: Record<string, Rect[]> = {
  P: [
    [0, 0, 26, 140],
    [26, 0, 62, 26],
    [62, 0, 26, 62],
    [26, 62, 62, 26],
  ],
  L: [
    [0, 0, 26, 140],
    [26, 114, 62, 26],
  ],
  A: [
    [0, 0, 26, 140],
    [62, 0, 26, 140],
    [26, 0, 36, 26],
    [26, 58, 36, 26],
  ],
  N: [
    [0, 0, 26, 140],
    [62, 0, 26, 140],
    [26, 0, 36, 30],
    [26, 46, 36, 30],
    [26, 92, 36, 30],
  ],
  G: [
    [0, 0, 26, 140],
    [26, 0, 62, 26],
    [26, 114, 62, 26],
    [62, 70, 26, 44],
    [44, 70, 44, 22],
  ],
  E: [
    [0, 0, 26, 140],
    [26, 0, 62, 26],
    [26, 58, 50, 24],
    [26, 114, 62, 26],
  ],
};

const WORD = "PLANGEN".split("");

/** Per-letter dimension figures, mirroring the reference drawing's margin. */
const DIMS = ["4200", "2100", "3600", "3500", "4500", "3300", "3500"];

/** Cells that get hatched — chosen so the texture reads as varied, not noisy. */
const HATCHED: Record<string, number[]> = {
  P: [1],
  L: [1],
  A: [3],
  N: [2],
  G: [4],
  E: [2],
};

function Letter({
  char,
  index,
  animate,
}: {
  char: string;
  index: number;
  animate: boolean;
}) {
  const rects = GLYPHS[char] ?? [];
  const hatched = HATCHED[char] ?? [];

  return (
    <svg
      viewBox="-2 -2 92 144"
      className="block h-full w-auto overflow-visible"
      role="presentation"
      aria-hidden="true"
    >
      {rects.map((r, i) => {
        const [x, y, w, h] = r;
        const perimeter = 2 * (w + h);
        return (
          <g key={i}>
            {hatched.includes(i) && (
              <rect x={x} y={y} width={w} height={h} fill="url(#pg-hatch)" />
            )}
            <rect
              x={x}
              y={y}
              width={w}
              height={h}
              fill="none"
              stroke="currentColor"
              strokeWidth={1.4}
              style={
                animate
                  ? {
                      strokeDasharray: perimeter,
                      strokeDashoffset: perimeter,
                      animation: `draw-in 900ms var(--ease-out) forwards`,
                      animationDelay: `${index * 90 + i * 70}ms`,
                    }
                  : undefined
              }
            />
            {/* Interior subdivision — one partition per tall cell */}
            {h > 90 && (
              <line
                x1={x}
                y1={y + h * 0.62}
                x2={x + w}
                y2={y + h * 0.62}
                stroke="currentColor"
                strokeWidth={0.6}
                opacity={0.75}
              />
            )}
            {/* Door swing on the first wide cell */}
            {i === 1 && w > 40 && (
              <path
                d={`M ${x + 8} ${y + h} A 14 14 0 0 1 ${x + 22} ${y + h - 14}`}
                fill="none"
                stroke="currentColor"
                strokeWidth={0.6}
                opacity={0.8}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}

export function Wordmark({
  className = "",
  showDimensions = true,
}: {
  className?: string;
  showDimensions?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [animate, setAnimate] = useState(false);

  // Draw the mark once, when it first comes into view. Never on scroll-up:
  // re-animating something the user has already read is pure cost to them.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setAnimate(true);
          io.disconnect();
        }
      },
      { threshold: 0.2 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`flex items-stretch gap-3 ${className}`}
      style={{ color: "var(--ink)" }}
      role="img"
      aria-label="PlanGen"
    >
      <HatchDef />
      <div className="flex flex-col items-end gap-3">
        {WORD.map((char, i) => (
          <div key={i} className="h-[clamp(34px,5.2vh,62px)]">
            <Letter char={char} index={i} animate={animate} />
          </div>
        ))}
      </div>

      {showDimensions && (
        <div className="flex flex-col justify-between py-1">
          {DIMS.map((d, i) => (
            <span
              key={i}
              className="dim leading-none"
              style={{ fontSize: 9, opacity: 0.8 }}
            >
              {d}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

/** Shared hatch pattern. Defined once; every glyph references it. */
export function HatchDef() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden="true">
      <defs>
        <pattern
          id="pg-hatch"
          width="5"
          height="5"
          patternUnits="userSpaceOnUse"
          patternTransform="rotate(45)"
        >
          <line
            x1="0"
            y1="0"
            x2="0"
            y2="5"
            stroke="currentColor"
            strokeWidth="0.7"
            opacity="0.35"
          />
        </pattern>
      </defs>
    </svg>
  );
}
