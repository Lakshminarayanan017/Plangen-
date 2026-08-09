# Monument assets

Four transparent-background WebP files belong here. They are **not** in the repo yet —
every page that needs one currently draws a dashed placeholder box of the correct
aspect ratio and says which file is missing.

| File                    | Used by            | Aspect (w ÷ h) |
| ----------------------- | ------------------ | -------------- |
| `taj.webp`              | Assistant, plate 1 | 1.52           |
| `brihadeeswarar.webp`   | Assistant, plate 2 | 1.08           |
| `liberty.webp`          | Assistant, plate 3 | 0.54           |
| `eiffel.webp`           | Hero               | 1.06           |

Ratios are measured off `docs/mockups/` and mirrored in
`src/components/ui/Monument.tsx`. If a generated asset lands on a different shape,
update `RATIO` there so the reserved box still matches the art.

Drop the files in and the placeholders disappear on their own — nothing else to change.
Graphite line-work on transparent background only: no paper baked in (the sheet supplies
that), no colour, no drop shadow.
