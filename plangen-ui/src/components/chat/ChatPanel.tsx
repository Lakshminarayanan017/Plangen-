import { CentreMark } from '../technical/ConstructionGrid'
import { MinimizeIcon, CloseIcon, FileIcon, DownloadIcon } from '../ui/Icons'
import { Message } from './Message'
import { InputBar } from './InputBar'

type Props = {
  onClose: () => void
}

/**
 * The assistant transcript, as drawn in docs/mockups/first_slide.png — a ruled
 * panel seated at the foot of the sheet, over the monument row. It keeps that
 * seat at every width, spanning what is available rather than turning into a
 * drawer or a full-screen takeover.
 */
export function ChatPanel({ onClose }: Props) {
  return (
    <section className="chat" aria-label="Plangen AI Assistant">
      <header className="chat__head">
        <CentreMark className="chat__mark" />
        <h2 className="type-nav chat__title">Plangen AI Assistant</h2>
        <div className="chat__controls">
          <button className="icon-btn" type="button" aria-label="Minimize">
            <MinimizeIcon className="icon" />
          </button>
          <button className="icon-btn" type="button" aria-label="Close" onClick={onClose}>
            <CloseIcon className="icon" />
          </button>
        </div>
      </header>

      <div className="chat__log">
        <Message from="assistant">
          Hello! I'm Plangen AI Assistant.
          <br />
          How can I help you design today?
        </Message>

        <Message from="user" time="10:34 AM" read>
          Generate a 3BHK house floor plan
        </Message>

        <Message from="assistant" time="10:35 AM">
          Sure! Here is a modern 3BHK floor plan with detailed layout, dimensions, and
          annotations.
          <a className="attach" href="/3BHK_FLOOR_PLAN.pdf" download>
            <FileIcon className="attach__kind" />
            <span className="attach__meta">
              <span className="type-body attach__name">3BHK_FLOOR_PLAN.pdf</span>
              <span className="type-dim attach__size">2.4 MB</span>
            </span>
            <DownloadIcon className="icon attach__get" />
          </a>
        </Message>
      </div>

      <InputBar variant="panel" />
    </section>
  )
}
