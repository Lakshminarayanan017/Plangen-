/**
 * Paper grain. Mounted exactly once, at the app root, on top of everything —
 * multiply at 0.35. Per-component texture makes blacks inconsistent, so this
 * is the only place the plate is ever used.
 */
export function TextureLayer() {
  return <div className="texture-layer" aria-hidden="true" />
}
