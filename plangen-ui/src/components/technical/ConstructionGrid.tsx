/**
 * Setting-out lines: the compass arcs and centre marks a draughtsman leaves on
 * the sheet. Purely faint, purely behind, and drawn as real paths.
 */
export function ConstructionGrid() {
  return (
    <svg
      className="construction"
      viewBox="0 0 1440 960"
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
    >
      <g className="construction__stroke">
        <path d="M -40 250 A 290 290 0 0 1 250 -40" />
        <path d="M -40 300 A 340 340 0 0 1 300 -40" />
        <path d="M 1480 300 A 340 340 0 0 0 1140 -40" />
        <path d="M 1140 260 A 300 300 0 0 1 1440 -40" />
      </g>
    </svg>
  )
}

/**
 * The little crosshair rosette that sits under the assistant title in the
 * mockup — a centre mark, not an icon.
 */
export function CentreMark({ className }: { className?: string }) {
  return (
    <svg
      className={`centre-mark ${className ?? ''}`}
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <g className="centre-mark__stroke">
        <circle cx="12" cy="12" r="3.4" />
        <circle cx="12" cy="12" r="1.1" />
        <path d="M12 0.8v6.6M12 16.6v6.6M0.8 12h6.6M16.6 12h6.6" />
        <path d="M4.6 4.6 8 8M16 16l3.4 3.4M19.4 4.6 16 8M8 16l-3.4 3.4" />
      </g>
    </svg>
  )
}
