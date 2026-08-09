/**
 * The sheet edge. Two hairlines, `--frame-gap` apart, held off the viewport
 * edge by `--frame-inset`. It sits above the page and below the texture, and
 * it is the only thing on the page allowed to look like a border.
 */
export function Frame() {
  return (
    <div className="frame" aria-hidden="true">
      <div className="frame__rule frame__rule--outer" />
      <div className="frame__rule frame__rule--inner" />
    </div>
  )
}
