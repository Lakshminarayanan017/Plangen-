import { useState } from 'react'

/**
 * Monuments are assets, never code (rule 7). This component only places one.
 * When the WebP is missing it draws a dashed box of the same dimensions so the
 * layout is still honest about the hole.
 */

export type MonumentName = 'taj' | 'brihadeeswarar' | 'liberty' | 'eiffel'

/** Aspect ratios measured off docs/mockups — the box must reserve the real shape. */
const RATIO: Record<MonumentName, number> = {
  taj: 1.52,
  brihadeeswarar: 1.08,
  liberty: 0.54,
  eiffel: 1.06,
}

const LABEL: Record<MonumentName, string> = {
  taj: 'Taj Mahal',
  brihadeeswarar: 'Brihadeeswarar',
  liberty: 'Statue of Liberty',
  eiffel: 'Eiffel Tower',
}

type Props = {
  name: MonumentName
  className?: string
}

export function Monument({ name, className }: Props) {
  const [missing, setMissing] = useState(false)

  if (missing) {
    return (
      <div
        className={`monument monument--missing ${className ?? ''}`}
        style={{ aspectRatio: RATIO[name] }}
        role="img"
        aria-label={`${LABEL[name]} — asset not yet supplied`}
      >
        <span className="type-dim monument__slug">
          {name}.webp — asset missing
        </span>
      </div>
    )
  }

  return (
    <img
      className={`monument ${className ?? ''}`}
      src={`/monuments/${name}.webp`}
      alt={LABEL[name]}
      style={{ aspectRatio: RATIO[name] }}
      onError={() => setMissing(true)}
    />
  )
}
