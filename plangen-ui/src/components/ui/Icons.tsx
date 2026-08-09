/**
 * UI icons only — stroked, 0.5px-family weight, no fills.
 * Monuments and illustrations never live here; they are assets (rule 7).
 */

type IconProps = { className?: string }

const base = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

export function GridIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <rect x="3.5" y="3.5" width="7" height="7" />
      <rect x="13.5" y="3.5" width="7" height="7" />
      <rect x="3.5" y="13.5" width="7" height="7" />
      <rect x="13.5" y="13.5" width="7" height="7" />
    </svg>
  )
}

export function GearIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="3.1" />
      <circle cx="12" cy="12" r="6" />
      <path d="M12 3.4v2.6M12 18v2.6M3.4 12H6M18 12h2.6M5.9 5.9l1.9 1.9M16.2 16.2l1.9 1.9M18.1 5.9l-1.9 1.9M7.8 16.2l-1.9 1.9" />
    </svg>
  )
}

export function UserIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="9" />
      <circle cx="12" cy="9.6" r="3.2" />
      <path d="M5.9 19.4a6.6 6.6 0 0 1 12.2 0" />
    </svg>
  )
}

export function MenuIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M3.5 7h17M3.5 12h17M3.5 17h17" />
    </svg>
  )
}

export function ArrowRightIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M4 12h15.5M14 6.5 19.8 12 14 17.5" />
    </svg>
  )
}

export function PaperclipIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M17.6 10.3 11 16.9a3.4 3.4 0 0 1-4.8-4.8l7.2-7.2a2.3 2.3 0 0 1 3.2 3.2l-7.1 7.1a1.1 1.1 0 0 1-1.6-1.6l6.4-6.4" />
    </svg>
  )
}

export function CompassIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <circle cx="12" cy="12" r="3.4" />
      <path d="M12 1.8v5.2M12 17v5.2M1.8 12H7M17 12h5.2" />
      <path d="M5.4 5.4 8.4 8.4M15.6 15.6l3 3M18.6 5.4l-3 3M8.4 15.6l-3 3" />
    </svg>
  )
}

export function MinimizeIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M5 12h14" />
    </svg>
  )
}

export function CloseIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  )
}

export function FileIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M6.5 2.8h7.2l4.3 4.3v14.1H6.5z" />
      <path d="M13.7 2.8v4.3H18" />
      <path d="M9.2 12h5.6M9.2 15.2h5.6M9.2 18.4h3.4" />
    </svg>
  )
}

export function DownloadIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M12 3.5v11.4M7.6 10.9 12 15.3l4.4-4.4" />
      <path d="M4.4 19.2h15.2" />
    </svg>
  )
}

export function CheckCheckIcon({ className }: IconProps) {
  return (
    <svg {...base} className={className} aria-hidden="true">
      <path d="M2 13.2 6.2 17.4 13.6 10" />
      <path d="M9.6 13.2 13.8 17.4 21.2 10" />
    </svg>
  )
}
