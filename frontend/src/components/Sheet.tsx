/**
 * Sheet.tsx — the drafting vocabulary.
 *
 * Every surface in PlanGen is a drawing sheet, so these are the parts a real
 * sheet has: a bordered frame with corner registration marks, a title block,
 * dimension strings, and rules. Building the app out of these rather than
 * generic cards is what makes the theme structural instead of decorative.
 *
 * Nothing here has a radius, a shadow, or a gradient. Depth is line weight,
 * exactly as on paper.
 */

import type { ReactNode } from "react";

/* ── Frame ───────────────────────────────────────────────────────────── */

export function SheetFrame({
  children,
  className = "",
  inset = true,
}: {
  children: ReactNode;
  className?: string;
  inset?: boolean;
}) {
  return (
    <div className={`relative ${className}`}>
      <div
        className="pointer-events-none absolute inset-0 border"
        style={{ borderColor: "var(--rule)" }}
        aria-hidden="true"
      />
      {inset && (
        <div
          className="pointer-events-none absolute border"
          style={{
            inset: "6px",
            borderColor: "var(--rule-2)",
          }}
          aria-hidden="true"
        />
      )}
      <CornerMarks />
      {children}
    </div>
  );
}

/** Registration crosses — how a real print is aligned on the press. */
function CornerMarks() {
  const corners = [
    { top: 0, left: 0 },
    { top: 0, right: 0 },
    { bottom: 0, left: 0 },
    { bottom: 0, right: 0 },
  ];
  return (
    <>
      {corners.map((pos, i) => (
        <svg
          key={i}
          width="14"
          height="14"
          className="pointer-events-none absolute"
          style={{ ...pos, color: "var(--ink-3)" }}
          aria-hidden="true"
        >
          <path
            d="M7 0 V14 M0 7 H14"
            stroke="currentColor"
            strokeWidth="0.75"
            fill="none"
          />
        </svg>
      ))}
    </>
  );
}

/* ── Title block ─────────────────────────────────────────────────────── */

export interface TitleRow {
  label: string;
  value: ReactNode;
  /** Marks a row as live/current — the one place accent is allowed. */
  live?: boolean;
}

export function TitleBlock({
  rows,
  title,
  sheet,
  className = "",
}: {
  rows: TitleRow[];
  title?: string;
  sheet?: string;
  className?: string;
}) {
  return (
    <div
      className={`border ${className}`}
      style={{ borderColor: "var(--rule)", background: "var(--paper)" }}
    >
      {(title || sheet) && (
        <div
          className="flex items-baseline justify-between border-b px-3 py-2"
          style={{ borderColor: "var(--rule)" }}
        >
          <span
            className="font-semibold tracking-wide"
            style={{ fontFamily: "var(--font-display)", fontSize: 13 }}
          >
            {title}
          </span>
          {sheet && <span className="annot">{sheet}</span>}
        </div>
      )}
      <dl className="divide-y" style={{ borderColor: "var(--rule-2)" }}>
        {rows.map((r) => (
          <div
            key={r.label}
            className="flex items-baseline justify-between gap-3 px-3 py-1.5"
            style={{ borderColor: "var(--rule-2)" }}
          >
            <dt className="annot shrink-0">{r.label}</dt>
            <dd
              className="dim text-right"
              style={{
                color: r.live ? "var(--accent)" : "var(--ink)",
                fontWeight: r.live ? 600 : 400,
              }}
            >
              {r.value}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/* ── Dimension string ────────────────────────────────────────────────── */

/**
 * The measured-length notation from a real drawing: a run with end ticks and
 * the figure sitting on it. Used down the margin beside the wordmark and
 * around the plan on the output sheet.
 */
export function DimensionString({
  value,
  orientation = "vertical",
  length = 120,
  className = "",
}: {
  value: string;
  orientation?: "vertical" | "horizontal";
  length?: number;
  className?: string;
}) {
  const vertical = orientation === "vertical";
  return (
    <div
      className={`relative flex items-center justify-center ${className}`}
      style={{
        width: vertical ? 34 : length,
        height: vertical ? length : 34,
        color: "var(--ink-2)",
      }}
      aria-hidden="true"
    >
      <svg
        width={vertical ? 34 : length}
        height={vertical ? length : 34}
        className="absolute inset-0"
      >
        {vertical ? (
          <>
            <line x1="17" y1="2" x2="17" y2={length - 2} stroke="currentColor" strokeWidth="0.75" />
            <line x1="11" y1="2" x2="23" y2="2" stroke="currentColor" strokeWidth="0.75" />
            <line x1="11" y1={length - 2} x2="23" y2={length - 2} stroke="currentColor" strokeWidth="0.75" />
          </>
        ) : (
          <>
            <line x1="2" y1="17" x2={length - 2} y2="17" stroke="currentColor" strokeWidth="0.75" />
            <line x1="2" y1="11" x2="2" y2="23" stroke="currentColor" strokeWidth="0.75" />
            <line x1={length - 2} y1="11" x2={length - 2} y2="23" stroke="currentColor" strokeWidth="0.75" />
          </>
        )}
      </svg>
      <span
        className="dim relative px-1"
        style={{
          background: "var(--paper)",
          writingMode: vertical ? "vertical-rl" : undefined,
          fontSize: 10,
        }}
      >
        {value}
      </span>
    </div>
  );
}

/* ── Rules & labels ──────────────────────────────────────────────────── */

export function SectionRule({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 py-2" role="presentation">
      <span className="h-px flex-1" style={{ background: "var(--rule)" }} />
      {label && <span className="annot shrink-0">{label}</span>}
      <span className="h-px flex-1" style={{ background: "var(--rule)" }} />
    </div>
  );
}

/* ── Button ──────────────────────────────────────────────────────────── */

export function DraftButton({
  children,
  onClick,
  type = "button",
  variant = "primary",
  disabled,
  className = "",
  ...rest
}: {
  children: ReactNode;
  onClick?: () => void;
  type?: "button" | "submit";
  variant?: "primary" | "ghost";
  disabled?: boolean;
  className?: string;
} & Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "onClick" | "type">) {
  const primary = variant === "primary";
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={`group relative inline-flex items-center gap-2 border px-5 py-2.5 transition-colors
                  disabled:cursor-not-allowed disabled:opacity-40 ${className}`}
      style={{
        borderColor: primary ? "var(--accent)" : "var(--rule)",
        color: primary ? "var(--accent)" : "var(--ink)",
        background: "transparent",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
      }}
      {...rest}
    >
      {/* Hover fill sweeps from the left — a pen stroke, not a fade. */}
      <span
        className="absolute inset-0 origin-left scale-x-0 transition-transform duration-200 group-hover:scale-x-100 group-disabled:scale-x-0"
        style={{
          background: primary ? "var(--accent-soft)" : "var(--rule-2)",
          transitionTimingFunction: "var(--ease-out)",
        }}
        aria-hidden="true"
      />
      <span className="relative">{children}</span>
    </button>
  );
}
