import { useState, type CSSProperties } from 'react'
import { Nav } from '../components/chrome/Nav'
import { ConstructionGrid, CentreMark } from '../components/technical/ConstructionGrid'
import { Dimension } from '../components/technical/Dimension'
import { Monument, type MonumentName } from '../components/ui/Monument'
import { InputBar } from '../components/chat/InputBar'
import { ChatPanel } from '../components/chat/ChatPanel'

/**
 * Slide 2 — three surveyed monuments under the assistant title.
 * Slide 1 is this same sheet with `<ChatPanel />` open, not a second route.
 *
 * Figures are the ones drawn in docs/mockups/second_slide.png.
 */
type Plate = {
  name: MonumentName
  side: 'left' | 'right'
  height: string
  /** Second vertical figure taken from the base up — Liberty's pedestal. */
  pedestal?: string
  top?: string
  segments: string[]
  total: string
}

const PLATES: Plate[] = [
  {
    name: 'taj',
    side: 'left',
    height: '73.50',
    top: '73.00',
    segments: ['15.20', '30.40', '15.20'],
    total: '60.80',
  },
  {
    name: 'brihadeeswarar',
    side: 'left',
    height: '66.40',
    segments: ['16.40', '32.80', '16.40'],
    total: '65.60',
  },
  {
    name: 'liberty',
    side: 'right',
    height: '93.00',
    pedestal: '46.00',
    segments: ['21.00', '31.00', '21.00'],
    total: '73.00',
  },
]

/** Tallest figure in the row — the monuments are drawn to a shared scale. */
const TALLEST = Math.max(...PLATES.map((p) => Number.parseFloat(p.height)))

export default function Assistant({ chatOpen = false }: { chatOpen?: boolean }) {
  const [open, setOpen] = useState(chatOpen)

  return (
    <div className="sheet">
      <Nav />

      <main className="assistant">
        <ConstructionGrid />

        <header className="assistant__head">
          <p className="type-eyebrow flanked">Plan. Generate. Visualize.</p>

          <h1 className="type-display assistant__title">Plangen AI Assistant</h1>

          <div className="assistant__mark flanked">
            <CentreMark />
          </div>

          <p className="type-body assistant__lede">
            Your intelligent partner in architectural design.
            <br />
            Ask anything. Generate ideas. Build better.
          </p>
        </header>

        <section className="plates" aria-label="Surveyed reference elevations">
          {PLATES.map((plate) => (
            <article
              key={plate.name}
              className={`plate plate--${plate.side}`}
              style={
                {
                  '--rise': `${(Number.parseFloat(plate.height) / TALLEST) * 100}%`,
                } as CSSProperties
              }
            >
              <div className="plate__body">
                <div className="plate__rise">
                  <Dimension orientation="vertical" segments={[plate.height]} />
                  {plate.pedestal ? (
                    <Dimension
                      orientation="vertical"
                      segments={[plate.pedestal]}
                      className="plate__pedestal"
                    />
                  ) : null}
                </div>

                <div className="plate__stack">
                  {plate.top ? (
                    <Dimension segments={[plate.top]} className="plate__top" />
                  ) : null}
                  <Monument name={plate.name} className="plate__art" />
                </div>
              </div>

              <Dimension segments={plate.segments} total={plate.total} className="plate__run" />
            </article>
          ))}
        </section>
      </main>

      {open ? (
        <ChatPanel onClose={() => setOpen(false)} />
      ) : (
        <InputBar onActivate={() => setOpen(true)} />
      )}
    </div>
  )
}
