import type { ReactNode } from 'react'
import { CentreMark } from '../technical/ConstructionGrid'
import { UserIcon, CheckCheckIcon } from '../ui/Icons'

type Props = {
  from: 'assistant' | 'user'
  time?: string
  /** Draws the double-tick receipt after the timestamp. User messages only. */
  read?: boolean
  children: ReactNode
}

/** One exchange line: ruled avatar box, bubble, and an optional mono receipt. */
export function Message({ from, time, read, children }: Props) {
  const avatar = (
    <span className="msg__avatar">
      {from === 'assistant' ? <CentreMark /> : <UserIcon className="icon" />}
    </span>
  )

  return (
    <div className={`msg msg--${from}`}>
      {from === 'assistant' ? avatar : null}

      <div className="msg__column">
        <div className="msg__bubble type-body">{children}</div>
        {time ? (
          <p className="type-dim msg__meta">
            {time}
            {read ? <CheckCheckIcon className="msg__receipt" /> : null}
          </p>
        ) : null}
      </div>

      {from === 'user' ? avatar : null}
    </div>
  )
}
