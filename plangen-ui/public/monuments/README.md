# Monument assets

Four **white-background PNG** files belong here. They are **not** in the repo yet —
every page that needs one currently draws a dashed placeholder box of the correct
aspect ratio and says which file is missing.

| File                   | Used by            | Aspect (w ÷ h) |
| ---------------------- | ------------------ | -------------- |
| `taj.png`              | Assistant, plate 1 | 1.52           |
| `brihadeeswarar.png`   | Assistant, plate 2 | 1.08           |
| `liberty.png`          | Assistant, plate 3 | 0.54           |
| `eiffel.png`           | Hero               | 1.06           |

Ratios are measured off `docs/mockups/` and mirrored in
`src/components/ui/Monument.tsx`. If a generated asset lands on a different shape,
update `RATIO` there so the reserved box still matches the art.

## How they composite

Every monument is placed with `mix-blend-mode: multiply`. White goes fully transparent
against the paper, the graphite line-work stays, and the paper grain reads through the
art instead of sitting on top of a cut-out. No alpha channel is needed — a flat white
background is correct, and is what the generators produce most reliably.

What that requires of the art:

- **Pure white background** (`#FFFFFF`). A tinted white multiplies into a visible box.
- **Greyscale line-work only.** Multiply keeps colour, and there is no colour on this
  sheet.
- **No baked drop shadow, vignette, or paper texture.** The sheet supplies the paper;
  anything baked in multiplies twice and reads muddy.
- **No readable text in the image** — dimension figures are the `<Dimension />`
  component (rule 8).

Drop the files in and the placeholders disappear on their own — nothing else to change.
