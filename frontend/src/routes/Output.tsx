/**
 * Output.tsx — the finished sheet.
 *
 * The drawing is the hero and everything else defers to it: floor tabs read
 * as sheet tabs, the file list as a drawing register, and the title block
 * carries the real metadata a drawing is identified by. No card grid, no
 * summary statistics competing with the thing the user came for.
 */

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { api, type RunFile } from "../lib/api";
import { DraftButton, SheetFrame, TitleBlock } from "../components/Sheet";
import { navigate, type Brief } from "../App";

const FLOOR_ORDER = ["ground", "first", "second", "third"];

function floorRank(name: string) {
  const i = FLOOR_ORDER.findIndex((f) => name.toLowerCase().includes(f));
  return i === -1 ? 99 : i;
}

function prettyFloor(name: string) {
  const base = name.replace(/\.svg$/i, "").replace(/_/g, " ");
  return base.replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function Output({ runId, brief }: { runId: string; brief: Brief }) {
  const [files, setFiles] = useState<RunFile[]>([]);
  const [active, setActive] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const res = await api.runFiles(runId);
        if (!alive) return;
        setFiles(res.files ?? []);
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) setLoading(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [runId]);

  const svgs = files
    .filter((f) => f.type === ".svg")
    .sort((a, b) => floorRank(a.name) - floorRank(b.name));
  const dxfs = files.filter((f) => f.type === ".dxf");
  const current = svgs[active];

  return (
    <div className="sheet-grid min-h-screen p-3 sm:p-5">
      <div className="mx-auto grid max-w-6xl gap-4 lg:grid-cols-[1fr_280px]">
        {/* ── the drawing ───────────────────────────────────────────── */}
        <div className="flex min-w-0 flex-col gap-3">
          {svgs.length > 1 && (
            <div role="tablist" aria-label="Floors" className="flex flex-wrap gap-px">
              {svgs.map((f, i) => (
                <button
                  key={f.name}
                  role="tab"
                  aria-selected={i === active}
                  aria-controls="sheet-panel"
                  onClick={() => setActive(i)}
                  className="relative border px-4 py-2 transition-colors"
                  style={{
                    borderColor: i === active ? "var(--accent)" : "var(--rule)",
                    color: i === active ? "var(--accent)" : "var(--ink-2)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    letterSpacing: "0.12em",
                    textTransform: "uppercase",
                    background: "var(--paper)",
                  }}
                >
                  {prettyFloor(f.name)}
                  {i === active && (
                    <motion.span
                      layoutId="sheet-tab"
                      className="absolute inset-x-0 -bottom-px h-0.5"
                      style={{ background: "var(--accent)" }}
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                </button>
              ))}
            </div>
          )}

          <SheetFrame className="min-h-[60vh]">
            <div id="sheet-panel" role="tabpanel" className="p-4 sm:p-6">
              {loading && <Placeholder text="Fetching the sheet…" />}
              {!loading && !current && !error && (
                <Placeholder text="No drawing was produced for this run." />
              )}
              {error && <Placeholder text={error} accent />}
              {current && (
                <motion.img
                  key={current.name}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.25 }}
                  src={api.svgUrl(runId, current.name)}
                  alt={`Floor plan drawing — ${prettyFloor(current.name)}`}
                  className="mx-auto block h-auto w-full max-w-full"
                  style={{ maxHeight: "72vh", objectFit: "contain" }}
                />
              )}
            </div>
          </SheetFrame>
        </div>

        {/* ── title block + register ────────────────────────────────── */}
        <aside className="flex flex-col gap-4">
          <TitleBlock
            title="PLANGEN"
            sheet={current ? `SHEET ${String(active + 1).padStart(2, "0")}/${String(svgs.length).padStart(2, "0")}` : "SHEET —"}
            rows={[
              { label: "Plot", value: brief.plot ?? "—" },
              { label: "Facing", value: brief.facing ?? "—" },
              { label: "Floors", value: brief.floors ?? String(svgs.length || "—") },
              { label: "Drawing", value: current ? prettyFloor(current.name) : "—" },
              { label: "Run", value: runId, live: true },
              { label: "Date", value: new Date().toISOString().slice(0, 10) },
            ]}
          />

          <div className="border" style={{ borderColor: "var(--rule)" }}>
            <p className="annot border-b px-3 py-2" style={{ borderColor: "var(--rule)" }}>
              Drawing register
            </p>
            <ul>
              {[...svgs, ...dxfs].map((f) => (
                <li
                  key={f.name}
                  className="flex items-center justify-between gap-2 border-b px-3 py-2 last:border-b-0"
                  style={{ borderColor: "var(--rule-2)" }}
                >
                  <a
                    href={api.svgUrl(runId, f.name)}
                    download={f.name}
                    className="dim truncate underline-offset-2 hover:underline"
                    style={{ color: "var(--ink)" }}
                  >
                    {f.name}
                  </a>
                  <span className="dim shrink-0" style={{ opacity: 0.7 }}>
                    {f.size_kb} KB
                  </span>
                </li>
              ))}
              {files.length === 0 && !loading && (
                <li className="dim px-3 py-2">No files.</li>
              )}
            </ul>
          </div>

          <DraftButton
            onClick={() => navigate("/design")}
            className="w-full justify-center"
          >
            Draw another
          </DraftButton>
        </aside>
      </div>
    </div>
  );
}

function Placeholder({ text, accent }: { text: string; accent?: boolean }) {
  return (
    <div className="grid min-h-[50vh] place-items-center">
      <p
        className="annot text-center"
        style={{ color: accent ? "var(--accent)" : "var(--ink-3)" }}
        role={accent ? "alert" : undefined}
      >
        {text}
      </p>
    </div>
  );
}
