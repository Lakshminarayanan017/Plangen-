/**
 * ThemeToggle — ink-on-paper <-> cyanotype.
 *
 * Dark mode here is a designed variant (white lines on Prussian blue, the
 * actual second form of this medium), not an inversion. Worth offering
 * explicitly rather than only following the OS, because which one reads
 * better depends on the room the user is in.
 */

import { useEffect, useState } from "react";

type Mode = "system" | "light" | "dark";
const KEY = "plangen-theme";

export function ThemeToggle() {
  const [mode, setMode] = useState<Mode>(
    () => (localStorage.getItem(KEY) as Mode) || "system",
  );

  useEffect(() => {
    const root = document.documentElement;
    if (mode === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", mode);
    localStorage.setItem(KEY, mode);
  }, [mode]);

  const next: Record<Mode, Mode> = {
    system: "light",
    light: "dark",
    dark: "system",
  };
  const label: Record<Mode, string> = {
    system: "AUTO",
    light: "PRINT",
    dark: "BLUE",
  };

  return (
    <button
      onClick={() => setMode(next[mode])}
      className="annot fixed right-3 top-3 z-40 border px-2 py-1 transition-colors hover:opacity-100"
      style={{
        borderColor: "var(--rule)",
        background: "var(--paper)",
        opacity: 0.75,
      }}
      aria-label={`Drawing medium: ${label[mode]}. Activate to change.`}
      title="Drawing medium"
    >
      {label[mode]}
    </button>
  );
}
