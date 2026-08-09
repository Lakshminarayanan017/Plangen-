import { GridIcon, GearIcon, UserIcon, MenuIcon } from '../ui/Icons'

const LINKS = ['Projects', 'Design', 'Analyze', 'Documents', 'About']

/**
 * Top bar: ruled wordmark box, mono link row, three icon affordances, and the
 * hairline that closes the band. Below 768px the links collapse and only the
 * wordmark and a menu icon survive (responsive strategy).
 */
export function Nav() {
  return (
    <header className="nav">
      <a className="wordmark" href="/">
        <span className="wordmark__box">
          <span className="wordmark__text">Plangen</span>
        </span>
      </a>

      <nav className="nav__links" aria-label="Primary">
        {LINKS.map((label) => (
          <a key={label} className="type-nav nav__link" href={`/${label.toLowerCase()}`}>
            {label}
          </a>
        ))}
      </nav>

      <div className="nav__tools">
        <button className="icon-btn" type="button" aria-label="Workspaces">
          <GridIcon className="icon" />
        </button>
        <button className="icon-btn" type="button" aria-label="Settings">
          <GearIcon className="icon" />
        </button>
        <button className="icon-btn" type="button" aria-label="Account">
          <UserIcon className="icon" />
        </button>
        <button className="icon-btn nav__menu" type="button" aria-label="Menu">
          <MenuIcon className="icon" />
        </button>
      </div>
    </header>
  )
}
