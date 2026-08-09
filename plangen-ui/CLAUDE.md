# PlanGen — frontend

Neuro-symbolic residential floor-plan generator. This repo is the web UI only.

## What this looks like

Graphite on aged paper. An architect's drafting sheet from roughly 1910, reproduced
faithfully enough that the browser chrome feels like the anachronism.

Three adjectives: **archival, exacting, quiet.**

Reference mockups live in `docs/mockups/`. Read the relevant one before building any
page. They are the source of truth for spacing, weight, and proportion — match them,
do not reinterpret them.

## Non-negotiable rules

1. **No color.** Ever. The entire palette is paper, ink, and greys. If a design problem
   seems to need color, it needs hierarchy instead.
2. **No shadows, no gradients, no blur, no glassmorphism.** Depth comes from line weight
   and overlap, the way it does on paper.
3. **All rules are 0.5px** unless the mockup clearly shows a heavier frame line.
4. **Mono for technical text** (nav, dimensions, labels, eyebrows), **serif for prose**
   (headings, body, chat). Never mix the roles.
5. **Letterspacing is load-bearing.** Nav and eyebrow text without wide tracking looks
   wrong immediately. Use the tokens.
6. **All text is real DOM text.** Nothing readable may be baked into an image.
7. **Monuments and illustrations are assets, not code.** Never attempt to draw the Taj
   Mahal, the Brihadeeswarar gopuram, the Statue of Liberty, or the Eiffel Tower in SVG.
   They live in `public/monuments/` as transparent WebP. If an asset is missing, use a
   dashed placeholder box of the correct dimensions and say so.
8. **Dimension annotations are code, not assets.** They are the `<Dimension />`
   component, always.

## Tokens

Defined in `src/styles/tokens.css`. Never hardcode these values in a component.

```css
--paper:         #EAE6DC;   /* page background */
--paper-deep:    #E2DDD1;   /* recessed panels, chat bubbles */
--ink:           #1C1A16;   /* headings, primary text */
--ink-soft:      #3A362F;   /* body text, dimension figures */
--graphite:      #6E6859;   /* secondary text, captions */
--rule:          #9C9689;   /* standard hairlines, brackets */
--rule-faint:    #C4BEB0;   /* construction lines, grid */
--btn:           #1F1D1A;   /* SEND button, primary CTA fill */

--font-display:  'Bodoni Moda', serif;
--font-body:     'EB Garamond', serif;
--font-tech:     'IBM Plex Mono', monospace;

--track-nav:     0.18em;
--track-eyebrow: 0.32em;
--rule-hair:     0.5px;
--frame-gap:     6px;       /* distance between the two frame rules */
```

## Structure

```
src/
  components/
    chrome/     Frame, Nav, TextureLayer      -- shell, on every page
    technical/  Dimension, ConstructionGrid   -- the drafting vocabulary
    chat/       ChatPanel, Message, InputBar
    ui/         Button, IconButton
  pages/
    Hero.tsx        slide 3 -- Eiffel, isometric
    Assistant.tsx   slide 2 -- three monuments + dimensions
    Rendering.tsx   slide 4 -- deferred, animation phase
  styles/tokens.css
public/
  monuments/  taj.webp, brihadeeswarar.webp, liberty.webp, eiffel.webp
  texture/    paper.png
docs/mockups/ first_slide.png, second_slide.png, third_slide.png, fourth_slide.png
```

Slides 1 and 2 are the same page. Slide 1 is `Assistant.tsx` with `<ChatPanel />` open.
Do not build them as separate routes.

## Texture

One layer, mounted once at the app root, above pages and below nothing:
`mix-blend-mode: multiply`, `opacity: 0.35`, `pointer-events: none`.
Never apply texture per-component — it makes blacks inconsistent.

## Responsive strategy

Reflow, not scale-to-fit.

- **≥1280px** — mockup layout exactly as drawn.
- **768–1279px** — monument row keeps three across, dimension segment labels drop to
  total-only.
- **<768px** — monuments become a horizontal scroll strip, one at a time, snap-aligned.
  All dimension annotations hidden. Nav collapses to logo + menu icon.

Chat panel is a right-side drawer above 1024px and a full-screen sheet below it.

## Motion

Deferred until the pages are static-complete. When it arrives it will be
`stroke-dashoffset` line-drawing on SVG, driven by raw `requestAnimationFrame`, not
Framer Motion. Build the technical components with real SVG `<path>` elements so they
are animation-ready. Do not add page transitions or hover animations in the meantime.

Respect `prefers-reduced-motion` from the first animated component onward.

## Verification

There is a screenshot script — `npm run shot -- <route>` writes a PNG to `.shots/`.
After building any page, run it and read the output image before reporting done.
Compare it against the matching file in `docs/mockups/` and state specifically what
differs.

## Working agreement

- One component per session. Commit before moving on.
- If a mockup is ambiguous, pick the reading that requires less code and say which you
  picked. Do not ask three clarifying questions.
- Do not refactor files you were not asked to touch.
