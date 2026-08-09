import { useRef } from 'react'
import { useSize } from './useSize'

/**
 * A drafted dimension annotation: witness ticks, a running line, 45° slashes at
 * the extremes, and mono figures. Every stroke is a real <path> in pixel space
 * so `stroke-dashoffset` can draw it later.
 *
 *   <Dimension segments={['15.20', '30.40', '15.20']} total="60.80" />
 *   <Dimension orientation="vertical" segments={['73.50']} />
 *
 * Segment widths are proportional to the figures themselves — a 30.40 bay is
 * twice a 15.20 one, which is what makes the annotation look measured rather
 * than decorative.
 */

const TICK = 5 // witness tick, each side of the line
const SLASH = 4 // half-length of the terminal 45° slash

type Props = {
  segments: string[]
  total?: string
  orientation?: 'horizontal' | 'vertical'
  className?: string
}

function weights(segments: string[]) {
  const values = segments.map((s) => Number.parseFloat(s))
  const usable = values.every((v) => Number.isFinite(v) && v > 0)
  const raw = usable ? values : segments.map(() => 1)
  const sum = raw.reduce((a, b) => a + b, 0)
  return raw.map((v) => v / sum)
}

/** Cumulative boundary offsets, 0 … length, including both ends. */
function boundaries(segments: string[], length: number) {
  const stops = [0]
  let acc = 0
  for (const w of weights(segments)) {
    acc += w
    stops.push(acc * length)
  }
  return stops
}

export function Dimension({
  segments,
  total,
  orientation = 'horizontal',
  className,
}: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const { w, h } = useSize(ref)
  const vertical = orientation === 'vertical'
  const length = vertical ? h : w

  const stops = length > 0 ? boundaries(segments, length) : []
  const axis = TICK + 0.5 // the running line, offset so a 0.5px stroke lands crisp

  const run = vertical
    ? `M ${axis} 0 V ${length}`
    : `M 0 ${axis} H ${length}`

  const witness = stops
    .map((p) =>
      vertical
        ? `M ${axis - TICK} ${p} H ${axis + TICK}`
        : `M ${p} ${axis - TICK} V ${axis + TICK}`,
    )
    .join(' ')

  const slashes = [0, length]
    .map((p) =>
      vertical
        ? `M ${axis - SLASH} ${p + SLASH} L ${axis + SLASH} ${p - SLASH}`
        : `M ${p - SLASH} ${axis + SLASH} L ${p + SLASH} ${axis - SLASH}`,
    )
    .join(' ')

  const cumulative = weights(segments).reduce<number[]>((acc, weight, i) => {
    acc.push((acc[i - 1] ?? 0) + weight)
    return acc
  }, [])

  const labels = segments.map((label, i) => {
    const start = i === 0 ? 0 : cumulative[i - 1]
    const centre = (start + cumulative[i]) / 2
    return { label, centre: `${centre * 100}%` }
  })

  return (
    <div
      ref={ref}
      className={`dim ${vertical ? 'dim--v' : 'dim--h'} ${className ?? ''}`}
      role="presentation"
    >
      <div className="dim__figures">
        {labels.map(({ label, centre }, i) => (
          <span
            key={`${label}-${i}`}
            className="type-dim dim__figure"
            style={vertical ? { top: centre } : { left: centre }}
          >
            {label}
          </span>
        ))}
      </div>

      <svg
        className="dim__svg"
        width={vertical ? TICK * 2 + 1 : Math.max(length, 0)}
        height={vertical ? Math.max(length, 0) : TICK * 2 + 1}
        aria-hidden="true"
      >
        <path className="dim__line" d={run} />
        <path className="dim__line" d={witness} />
        <path className="dim__line" d={slashes} />
      </svg>

      {total ? (
        <div className="dim__total">
          <Dimension segments={[total]} orientation={orientation} />
        </div>
      ) : null}
    </div>
  )
}
