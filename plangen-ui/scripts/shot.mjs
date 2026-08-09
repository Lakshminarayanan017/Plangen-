// Screenshot harness. Boots a throwaway Vite dev server, renders one route,
// writes a PNG to .shots/ and exits.
//
//   npm run shot -- /            -> .shots/out.png + .shots/root.png
//   npm run shot -- /assistant   -> .shots/out.png + .shots/assistant.png
//   npm run shot -- / --w 1280 --h 800
//   npm run shot -- / --full     (full-page instead of viewport)
//
// Viewport defaults to 1440x960 — the 3:2 of docs/mockups/*.png.

import { spawn } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { chromium } from 'playwright'

const argv = process.argv.slice(2)
const flag = (name, fallback) => {
  const i = argv.indexOf(`--${name}`)
  return i === -1 ? fallback : argv[i + 1]
}

const route = argv.find((a) => a.startsWith('/')) ?? '/'
const width = Number(flag('w', 1440))
const height = Number(flag('h', 960))
const fullPage = argv.includes('--full')
const port = Number(flag('port', 5199))
const origin = `http://localhost:${port}`

const slug =
  route === '/'
    ? 'root'
    : route.replace(/^\/+/, '').replace(/[?=&/]+/g, '-').replace(/-+$/, '')

mkdirSync('.shots', { recursive: true })

const server = spawn(
  'npx',
  ['vite', '--port', String(port), '--strictPort', '--host', '127.0.0.1'],
  { stdio: 'ignore', shell: true },
)

const stop = () => {
  if (process.platform === 'win32' && server.pid) {
    spawn('taskkill', ['/pid', String(server.pid), '/T', '/F'], { stdio: 'ignore' })
  } else {
    server.kill('SIGTERM')
  }
}

async function waitForServer(timeoutMs = 60_000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      const res = await fetch(origin, { signal: AbortSignal.timeout(1500) })
      if (res.ok) return
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 250))
  }
  throw new Error(`dev server never came up on ${origin}`)
}

let exitCode = 0
try {
  await waitForServer()

  const browser = await chromium.launch()
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 2 })

  const problems = []
  page.on('console', (m) => m.type() === 'error' && problems.push(m.text()))
  page.on('pageerror', (e) => problems.push(String(e)))

  await page.goto(origin + route, { waitUntil: 'networkidle' })
  await page.evaluate(() => document.fonts.ready)
  await page.evaluate(() =>
    Promise.all(
      [...document.images]
        .filter((img) => !img.complete)
        .map((img) => new Promise((res) => { img.onload = img.onerror = res })),
    ),
  )
  await page.waitForTimeout(200)

  for (const out of [`.shots/out.png`, `.shots/${slug}.png`]) {
    await page.screenshot({ path: out, fullPage })
  }

  await browser.close()

  console.log(`shot ${route} @ ${width}x${height}${fullPage ? ' (full)' : ''}`)
  console.log(`  -> .shots/out.png`)
  console.log(`  -> .shots/${slug}.png`)
  if (problems.length) {
    console.log(`\n${problems.length} console error(s):`)
    for (const p of problems.slice(0, 10)) console.log(`  ! ${p}`)
  }
} catch (err) {
  console.error(err)
  exitCode = 1
} finally {
  stop()
}

process.exit(exitCode)
