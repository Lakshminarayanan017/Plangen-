import { useLayoutEffect, useState, type RefObject } from 'react'

/**
 * Measured box of an element. The technical components draw in real pixels —
 * a viewBox scaled to fit would stretch tick marks and break the line weight.
 */
export function useSize(ref: RefObject<HTMLElement | null>) {
  const [size, setSize] = useState({ w: 0, h: 0 })

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return

    const ro = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect
      setSize((prev) =>
        prev.w === width && prev.h === height ? prev : { w: width, h: height },
      )
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])

  return size
}
