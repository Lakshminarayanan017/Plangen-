import { Nav } from '../components/chrome/Nav'
import { Monument } from '../components/ui/Monument'
import { ArrowRightIcon } from '../components/ui/Icons'

/** Slide 3 — the landing sheet. Left column of type, Eiffel plate on the right. */
export default function Hero() {
  return (
    <div className="sheet">
      <Nav />

      <main className="hero">
        <section className="hero__col">
          <p className="type-eyebrow hero__eyebrow">Plan. Generate. Visualize.</p>

          <h1 className="type-display hero__title">
            AI-Powered
            <br />
            Architectural
            <br />
            Blueprint
            <br />
            Generator
          </h1>

          <div className="rule-h hero__rule" />

          <p className="type-body hero__lede">
            Transform ideas into precise architectural blueprints with the power of
            Generative AI.
          </p>

          <span className="btn-framed hero__cta">
            <a className="btn" href="/assistant">
              Start Designing
              <ArrowRightIcon className="icon" />
            </a>
          </span>
        </section>

        <section className="hero__plate">
          <Monument name="eiffel" className="hero__eiffel" />
        </section>
      </main>
    </div>
  )
}
