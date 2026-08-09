import { Frame } from './components/chrome/Frame'
import { TextureLayer } from './components/chrome/TextureLayer'
import Hero from './pages/Hero'
import Assistant from './pages/Assistant'

/**
 * Two sheets, two paths. Slide 1 is not a route — it is `/assistant` with the
 * chat panel open, which `?chat=open` forces for screenshots.
 */
export default function App() {
  const path = window.location.pathname.replace(/\/+$/, '') || '/'
  const chatOpen = new URLSearchParams(window.location.search).get('chat') === 'open'

  return (
    <>
      {path === '/assistant' ? <Assistant chatOpen={chatOpen} /> : <Hero />}
      <Frame />
      <TextureLayer />
    </>
  )
}
