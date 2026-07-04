# PlanGen Design System

## Theme: Architect's Drafting Studio
Dark monochrome elegance with warm gold and blueprint blue accents. The UI feels like a senior architect's digital portfolio — premium, precise, and sophisticated.

## Background
Very faint (2-3% opacity) line-art illustrations of famous architectural landmarks (Eiffel Tower, Taj Mahal, Colosseum, Petronas Towers, Burj Khalifa, Sydney Opera House, Parthenon) drawn as 2D architectural elevation drawings with single-weight thin lines, no fills. These float as a repeating panorama watermark behind all screens.

## Color Usage Rules
- Base: Dark monochrome (#0A0A0A background, #141414 surfaces, #1C1C1C elevated panels)
- Gold (#C4956A): Premium accents, active states, selected items, CTA highlights
- Blueprint Blue (#1E3A5F): Primary action buttons, links
- Green (#2D6B3F): Success, quality scores above threshold, compliance pass
- Amber (#8B6914): Warnings, quality below threshold
- Red (#8B2A2A): Errors, failed placement
- Floor plan canvas uses light paper white (#FAFAF8) background with room colors at 15% opacity

## Typography Rules
- Headlines: Playfair Display 700 for hero/display text (architectural gravitas)
- UI text: Inter for all body, labels, navigation
- Data/dimensions: JetBrains Mono for coordinates, scores, measurements
- Section labels: UPPERCASE Inter 500, letter-spacing 0.08em

## Animation Guidelines
- All transitions: smooth ease-out or spring physics
- Chat messages: slide-in from bottom (300ms spring)
- Page transitions: 400ms ease-out slide + fade
- SVG floor plans: stroke-draw animation on load (1.2s)
- Buttons: 150ms hover scale(1.02) + subtle glow
- Modals: 250ms scale(0.95->1.0) + backdrop blur

## Component Style
- Cards: 12px border-radius, subtle 1px border (#1F1F1F), no heavy shadows
- Buttons: 8px radius, filled (primary) or ghost (secondary)
- Chat bubbles: 16px radius with tail indicator
- Inputs: 8px radius, dark surface fill, subtle border that glows gold on focus
- Toggle switches: pill-shaped with spring animation, gold when ON