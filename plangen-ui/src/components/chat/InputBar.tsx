import { CentreMark } from '../technical/ConstructionGrid'
import { PaperclipIcon, ArrowRightIcon } from '../ui/Icons'

type Props = {
  /** `sheet` is the wide bar on the page; `panel` is the compact one in the chat. */
  variant?: 'sheet' | 'panel'
  onActivate?: () => void
}

export function InputBar({ variant = 'sheet', onActivate }: Props) {
  return (
    <form
      className={`input-bar input-bar--${variant}`}
      onSubmit={(e) => {
        e.preventDefault()
        onActivate?.()
      }}
    >
      <div className="input-bar__well">
        <CentreMark className="input-bar__mark" />
        <input
          className="input-bar__field"
          type="text"
          placeholder="Ask Plangen AI anything about architecture..."
          aria-label="Ask Plangen AI anything about architecture"
          onFocus={onActivate}
        />
        <button className="icon-btn input-bar__clip" type="button" aria-label="Attach a file">
          <PaperclipIcon className="icon" />
        </button>
        <button className="btn input-bar__send" type="submit">
          Send
          <ArrowRightIcon className="icon" />
        </button>
      </div>
    </form>
  )
}
