/**
 * Landing.tsx — the drawing sheet you arrive on.
 *
 * Composition follows the reference: the structure occupies the left two
 * thirds, the wordmark runs vertically down the right margin with its
 * dimension string, and the whole thing sits inside a bordered sheet with
 * registration marks.
 *
 * The scroll does one job and does it literally: the tower rotates and
 * descends as you read, so moving down the page moves down the building.
 */

import { Suspense, lazy, useRef } from "react";
import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { Wordmark } from "../components/Wordmark";
import { DraftButton, SectionRule, SheetFrame, TitleBlock } from "../components/Sheet";
import { navigate } from "../App";

const TowerScene = lazy(() => import("../three/TowerScene"));

export default function Landing() {
  const reduced = useReducedMotion() ?? false;
  const scrollRef = useRef<HTMLDivElement>(null);

  // Scroll progress across the pinned hero. Passed to the 3D as a MotionValue
  // so the canvas can read it every frame without React re-rendering.
  const { scrollYProgress } = useScroll({
    target: scrollRef,
    offset: ["start start", "end start"],
  });

  const headingY = useTransform(scrollYProgress, [0, 1], [0, reduced ? 0 : -40]);
  const headingOpacity = useTransform(scrollYProgress, [0, 0.55], [1, 0]);

  return (
    <div className="sheet-grid min-h-screen">
      {/* ── Hero: 200vh of scroll driving one pinned sheet ─────────────── */}
      <div ref={scrollRef} className="relative h-[220vh]">
        <div className="sticky top-0 h-screen overflow-hidden p-3 sm:p-5">
          <SheetFrame className="h-full w-full">
            <div className="relative grid h-full grid-cols-[1fr_auto] gap-2 p-4 sm:p-8">
              {/* structure */}
              <div className="relative min-w-0">
                <Suspense fallback={<TowerFallback />}>
                  <div
                    className="absolute inset-0"
                    role="img"
                    aria-label="Rotating wireframe elevation of the Eiffel Tower, drawn as an engineering lattice"
                  >
                    <TowerScene progress={scrollYProgress} reduced={reduced} />
                  </div>
                </Suspense>

                {/* Copy sits over the drawing, bottom-left, like a note. */}
                <motion.div
                  style={{ y: headingY, opacity: headingOpacity }}
                  className="pointer-events-none absolute bottom-2 left-0 max-w-[34ch]"
                >
                  <p className="annot mb-3">Est. 1889 — drawn, not guessed</p>
                  <h1
                    className="mb-4 text-[clamp(1.9rem,4.4vw,3.4rem)] font-semibold leading-[1.04] tracking-tight"
                    style={{ fontFamily: "var(--font-display)" }}
                  >
                    Every building starts
                    <br />
                    as a drawing.
                  </h1>
                  <p
                    className="mb-6 text-[15px] leading-relaxed"
                    style={{ color: "var(--ink-2)" }}
                  >
                    Describe your plot in plain words. PlanGen carves a
                    buildable floor plan against NBC minimums, Vastu and your
                    setbacks — and hands you the sheet.
                  </p>
                  <div className="pointer-events-auto">
                    <DraftButton onClick={() => navigate("/design")}>
                      Start designing
                    </DraftButton>
                  </div>
                </motion.div>
              </div>

              {/* margin: the wordmark + its dimensions */}
              <div className="flex items-center justify-end pr-1">
                <Wordmark />
              </div>
            </div>
          </SheetFrame>

          <ScrollHint reduced={reduced} />
        </div>
      </div>

      {/* ── Below the fold ─────────────────────────────────────────────── */}
      <section className="mx-auto max-w-5xl px-5 pb-24">
        <SectionRule label="What it actually does" />
        <div className="grid gap-px sm:grid-cols-3" style={{ background: "var(--rule)" }}>
          {STEPS.map((s) => (
            <div key={s.no} className="p-5" style={{ background: "var(--paper)" }}>
              <div className="dim mb-3" style={{ color: "var(--accent)" }}>
                {s.no}
              </div>
              <h3 className="mb-1.5 text-[15px] font-semibold">{s.title}</h3>
              <p className="text-[13.5px] leading-relaxed" style={{ color: "var(--ink-2)" }}>
                {s.body}
              </p>
            </div>
          ))}
        </div>

        <div className="mt-10 grid gap-6 sm:grid-cols-[1fr_auto] sm:items-end">
          <p className="max-w-[52ch] text-[15px] leading-relaxed" style={{ color: "var(--ink-2)" }}>
            Rooms are never dropped onto a canvas and nudged apart. Space is
            split by inserting walls, so overlapping rooms are impossible by
            construction — the same way a building is actually partitioned.
          </p>
          <TitleBlock
            title="PLANGEN"
            sheet="SHEET 00"
            rows={[
              { label: "Engine", value: "wall-graph carver" },
              { label: "Standard", value: "NBC 2016" },
              { label: "Corpus", value: "4,983 plans" },
              { label: "Output", value: "SVG · DXF" },
            ]}
            className="min-w-[240px]"
          />
        </div>

        <div className="mt-10">
          <DraftButton onClick={() => navigate("/design")}>
            Start designing
          </DraftButton>
        </div>
      </section>
    </div>
  );
}

const STEPS = [
  {
    no: "01",
    title: "Describe the plot",
    body: "Dimensions, facing, how many bedrooms. Plain sentences — no forms to fill in.",
  },
  {
    no: "02",
    title: "It asks what it needs",
    body: "Missing something structural, it asks. Nothing gets silently assumed.",
  },
  {
    no: "03",
    title: "Take the drawing",
    body: "A carved, compliant plan per floor, as a scaled SVG and a layered DXF.",
  },
];

function TowerFallback() {
  return (
    <div className="absolute inset-0 grid place-items-center">
      <span className="annot animate-pulse">Plotting structure…</span>
    </div>
  );
}

function ScrollHint({ reduced }: { reduced: boolean }) {
  if (reduced) return null;
  return (
    <div className="pointer-events-none absolute bottom-8 left-1/2 -translate-x-1/2">
      <motion.div
        className="flex flex-col items-center gap-2"
        animate={{ y: [0, 6, 0] }}
        transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
      >
        <span className="annot">Scroll</span>
        <span className="h-8 w-px" style={{ background: "var(--rule)" }} />
      </motion.div>
    </div>
  );
}
