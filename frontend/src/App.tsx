/**
 * App.tsx — shell, routing, and the one piece of shared state.
 *
 * The product is a linear commission: describe the plot, answer the
 * architect's questions, watch it get drawn, take the drawing. So routing is
 * a four-step hash router rather than a general-purpose one — no library,
 * no nested layouts, and the URL still survives a refresh and a paste.
 */

import { useCallback, useEffect, useState } from "react";
import Landing from "./routes/Landing";
import Chat from "./routes/Chat";
import Rendering from "./routes/Rendering";
import Output from "./routes/Output";
import { ThemeToggle } from "./components/ThemeToggle";

export type Brief = {
  plot?: string;
  facing?: string;
  floors?: string;
  bhk?: string;
  vastu?: boolean;
};

type Route =
  | { name: "landing" }
  | { name: "design" }
  | { name: "render"; runId: string }
  | { name: "sheet"; runId: string };

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\/?/, "");
  const [head, arg] = h.split("/");
  if (head === "design") return { name: "design" };
  if (head === "render" && arg) return { name: "render", runId: arg };
  if (head === "sheet" && arg) return { name: "sheet", runId: arg };
  return { name: "landing" };
}

export function navigate(to: string) {
  window.location.hash = to;
}

export default function App() {
  const [route, setRoute] = useState<Route>(parseHash);
  const [brief, setBrief] = useState<Brief>({});

  useEffect(() => {
    const onHash = () => {
      setRoute(parseHash());
      window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const onBrief = useCallback((b: Brief) => setBrief((p) => ({ ...p, ...b })), []);

  return (
    <>
      {/* Skip link — the first tab stop on every surface. */}
      <a
        href="#main"
        className="annot sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:border focus:px-3 focus:py-2"
        style={{ background: "var(--paper)", borderColor: "var(--accent)" }}
      >
        Skip to content
      </a>

      <ThemeToggle />

      <main id="main">
        {route.name === "landing" && <Landing />}
        {route.name === "design" && <Chat onBrief={onBrief} />}
        {route.name === "render" && (
          <Rendering runId={route.runId} brief={brief} />
        )}
        {route.name === "sheet" && <Output runId={route.runId} brief={brief} />}
      </main>
    </>
  );
}
