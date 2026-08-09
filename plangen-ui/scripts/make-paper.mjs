// Generates public/texture/paper.png — a 512x512 tileable grain plate.
// Greyscale only, mostly near-white so `multiply` at 0.35 reads as tooth in the
// paper rather than dirt. Run once; the output is committed.
//
//   node scripts/make-paper.mjs

import { deflateSync } from 'node:zlib'
import { mkdirSync, writeFileSync } from 'node:fs'

const SIZE = 512

// Deterministic noise so re-running never changes the plate.
let seed = 0x2f6e2b1
const rand = () => {
  seed ^= seed << 13
  seed ^= seed >>> 17
  seed ^= seed << 5
  return ((seed >>> 0) % 100000) / 100000
}

// Value-noise octave, wrapped so the tile has no seam.
const lattice = (n) => {
  const g = new Float64Array(n * n)
  for (let i = 0; i < g.length; i++) g[i] = rand()
  const smooth = (x, y) => {
    const x0 = Math.floor(x), y0 = Math.floor(y)
    const fx = x - x0, fy = y - y0
    const ex = fx * fx * (3 - 2 * fx), ey = fy * fy * (3 - 2 * fy)
    const at = (a, b) => g[(((b % n) + n) % n) * n + (((a % n) + n) % n)]
    const top = at(x0, y0) * (1 - ex) + at(x0 + 1, y0) * ex
    const bot = at(x0, y0 + 1) * (1 - ex) + at(x0 + 1, y0 + 1) * ex
    return top * (1 - ey) + bot * ey
  }
  return smooth
}

const coarse = lattice(16)
const mid = lattice(64)

// One raw scanline per row: filter byte 0, then RGB triples.
const raw = Buffer.alloc(SIZE * (1 + SIZE * 3))
for (let y = 0; y < SIZE; y++) {
  const row = y * (1 + SIZE * 3)
  raw[row] = 0
  for (let x = 0; x < SIZE; x++) {
    const blotch = coarse((x / SIZE) * 16, (y / SIZE) * 16)
    const fibre = mid((x / SIZE) * 64, (y / SIZE) * 64)
    const speck = rand()
    // 232..255 — a quiet band. Multiply darkens by at most ~9%.
    let v = 255 - (blotch * 9 + fibre * 7 + speck * 5)
    if (speck > 0.9995) v -= 26 // the occasional fleck of pulp
    const p = row + 1 + x * 3
    const b = Math.max(0, Math.round(v))
    raw[p] = raw[p + 1] = raw[p + 2] = b
  }
}

const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1
  return c >>> 0
})
const crc32 = (buf) => {
  let c = 0xffffffff
  for (const byte of buf) c = crcTable[(c ^ byte) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}

const chunk = (type, data) => {
  const len = Buffer.alloc(4)
  len.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type, 'ascii'), data])
  const crc = Buffer.alloc(4)
  crc.writeUInt32BE(crc32(body))
  return Buffer.concat([len, body, crc])
}

const ihdr = Buffer.alloc(13)
ihdr.writeUInt32BE(SIZE, 0)
ihdr.writeUInt32BE(SIZE, 4)
ihdr[8] = 8 // bit depth
ihdr[9] = 2 // truecolour
ihdr[10] = ihdr[11] = ihdr[12] = 0

const png = Buffer.concat([
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
  chunk('IHDR', ihdr),
  chunk('IDAT', deflateSync(raw, { level: 9 })),
  chunk('IEND', Buffer.alloc(0)),
])

mkdirSync('public/texture', { recursive: true })
writeFileSync('public/texture/paper.png', png)
console.log(`wrote public/texture/paper.png (${SIZE}x${SIZE}, ${png.length} bytes)`)
