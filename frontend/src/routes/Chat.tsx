/**
 * Chat.tsx — the commissioning conversation.
 *
 * Framed as talking to an architect over a drawing board rather than as a
 * messaging app: your words arrive as leader-line callouts pinned to the
 * sheet, the architect's as title-block entries. Everything gathered so far
 * accumulates in a live title block on the right, so the user can always see
 * what the building currently *is* without re-reading the transcript.
 *
 * Backdrop: Taj Mahal, Brihadeeswarar and Liberty as elevations, held at low
 * contrast so they never compete with the text they sit behind.
 *
 * The title block is also where degraded subsystems surface. Giving honesty a
 * designed home is why it can be shown without alarming anyone.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { api, type Diagnostics, type ParseResult } from "../lib/api";
import { DraftButton, SheetFrame, TitleBlock, type TitleRow } from "../components/Sheet";
import { StatueOfLiberty, TajMahal, TanjavurTemple } from "../components/Landmarks";
import { navigate, type Brief } from "../App";

type Msg = { id: number; from: "user" | "architect"; text: string };

let msgId = 0;

export default function Chat({ onBrief }: { onBrief: (b: Brief) => void }) {
  const reduced = useReducedMotion() ?? false;
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [interactive, setInteractive] = useState(false);
  const [diag, setDiag] = useState<Diagnostics | null>(null);
  const [reqs, setReqs] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const say = useCallback((from: Msg["from"], text: string) => {
    if (!text) return;
    setMessages((m) => [...m, { id: msgId++, from, text }]);
  }, []);

  /* ── boot ─────────────────────────────────────────────────────────── */
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [{ session_id }, health] = await Promise.all([
          api.createSession(),
          api.health().catch(() => null),
        ]);
        if (!alive) return;
        setSessionId(session_id);
        if (health) setDiag(health.diagnostics);
        say(
          "architect",
          "Tell me about the plot. Dimensions, which way it faces, and what you need inside it — a sentence is enough to start.",
        );
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      alive = false;
    };
  }, [say]);

  useEffect(() => {
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: reduced ? "auto" : "smooth",
    });
  }, [messages, reduced]);

  /* ── the gathering loop ───────────────────────────────────────────── */

  const absorb = useCallback(
    (r: ParseResult) => {
      if (r.data) {
        setReqs(r.data as Record<string, unknown>);
        onBrief(summarise(r.data as Record<string, unknown>));
      }
      const done = r.status === "success" || r.is_valid === true;
      setReady(done);
      return done;
    },
    [onBrief],
  );

  const pullNextQuestion = useCallback(
    async (sid: string) => {
      const action = await api.nextQuestion(sid);
      if (action.action === "ask" && action.question) {
        setInteractive(true);
        say("architect", String(action.question));
      } else {
        setInteractive(false);
        setReady(true);
        say(
          "architect",
          "That's everything structural. I can draw it whenever you're ready.",
        );
      }
    },
    [say],
  );

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || !sessionId || busy) return;
    setInput("");
    say("user", text);
    setBusy(true);
    setError(null);
    try {
      const result = interactive
        ? await api.answer(sessionId, text)
        : await api.parseText(sessionId, text);

      const complete = absorb(result);
      const note =
        (result.clarification_prompt as string | undefined) ??
        (result.message as string | undefined);
      if (note) say("architect", note);

      if (!complete) await pullNextQuestion(sessionId);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [input, sessionId, busy, interactive, say, absorb, pullNextQuestion]);

  const generate = useCallback(async () => {
    if (!sessionId) return;
    setBusy(true);
    setError(null);
    try {
      const { run_id } = await api.runPipeline(sessionId);
      navigate(`/render/${run_id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  }, [sessionId]);

  /* ── title block rows ─────────────────────────────────────────────── */
  const brief = summarise(reqs ?? {});
  const rows: TitleRow[] = [
    { label: "Plot", value: brief.plot ?? "—" },
    { label: "Facing", value: brief.facing ?? "—" },
    { label: "Floors", value: brief.floors ?? "—" },
    { label: "Programme", value: brief.bhk ?? "—" },
    { label: "Vastu", value: brief.vastu ? "Requested" : "—" },
    { label: "Status", value: ready ? "Ready to draw" : "Gathering", live: ready },
  ];

  const degraded = (diag?.checks ?? []).filter((c) => c.status !== "ok");

  return (
    <div className="sheet-grid relative min-h-screen overflow-hidden">
      <Backdrop />

      <div className="relative mx-auto grid max-w-6xl gap-4 p-3 sm:p-5 lg:grid-cols-[1fr_300px]">
        {/* ── conversation ──────────────────────────────────────────── */}
        <SheetFrame className="flex h-[calc(100vh-2.5rem)] flex-col">
          <header
            className="flex shrink-0 items-baseline justify-between border-b px-5 py-3"
            style={{ borderColor: "var(--rule)" }}
          >
            <h1
              className="text-[15px] font-semibold tracking-wide"
              style={{ fontFamily: "var(--font-display)" }}
            >
              COMMISSION
            </h1>
            <span className="annot">Sheet 01 — brief</span>
          </header>

          <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto px-5 py-5">
            <AnimatePresence initial={false}>
              {messages.map((m) => (
                <Message key={m.id} msg={m} reduced={reduced} />
              ))}
            </AnimatePresence>
            {busy && <Thinking />}
            {error && (
              <p
                className="dim mt-4 border-l-2 py-1 pl-3"
                style={{ borderColor: "var(--accent)", color: "var(--accent)" }}
                role="alert"
              >
                {error}
              </p>
            )}
          </div>

          <form
            className="shrink-0 border-t p-3"
            style={{ borderColor: "var(--rule)" }}
            onSubmit={(e) => {
              e.preventDefault();
              void send();
            }}
          >
            <div className="flex items-end gap-2">
              <label htmlFor="brief-input" className="sr-only">
                Describe your plot
              </label>
              <textarea
                id="brief-input"
                rows={2}
                value={input}
                disabled={!sessionId || busy}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void send();
                  }
                }}
                placeholder="30 by 40 feet, south facing, three bedrooms…"
                className="min-h-[46px] flex-1 resize-none border bg-transparent px-3 py-2 text-[14.5px] outline-none"
                style={{ borderColor: "var(--rule)", color: "var(--ink)" }}
              />
              <DraftButton type="submit" variant="ghost" disabled={!input.trim() || busy}>
                Send
              </DraftButton>
            </div>
            <p className="annot mt-2" style={{ opacity: 0.7 }}>
              Enter to send · Shift+Enter for a new line
            </p>
          </form>
        </SheetFrame>

        {/* ── the live title block ──────────────────────────────────── */}
        <aside className="flex flex-col gap-4">
          <TitleBlock title="PLANGEN" sheet="BRIEF" rows={rows} />

          <DraftButton
            onClick={() => void generate()}
            disabled={!ready || busy}
            className="w-full justify-center"
          >
            {busy ? "Working…" : "Draw the plan"}
          </DraftButton>

          {degraded.length > 0 && (
            <div className="border p-3" style={{ borderColor: "var(--rule)" }}>
              <p className="annot mb-2">Notes on this run</p>
              <ul className="space-y-2">
                {degraded.map((c) => (
                  <li key={c.name} className="text-[12px] leading-snug" style={{ color: "var(--ink-2)" }}>
                    <span
                      className="dim mr-1"
                      style={{
                        color: c.status === "missing" ? "var(--accent)" : "var(--ink-3)",
                      }}
                    >
                      {c.status === "missing" ? "✕" : "!"}
                    </span>
                    {c.impact || c.detail}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}

/* ── pieces ──────────────────────────────────────────────────────────── */

function Message({ msg, reduced }: { msg: Msg; reduced: boolean }) {
  const isUser = msg.from === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: reduced ? 0 : 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
      className={`mb-5 flex ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div className={`max-w-[80%] ${isUser ? "text-right" : ""}`}>
        <div className="annot mb-1.5">{isUser ? "Client" : "Architect"}</div>
        <div
          className={isUser ? "border-r-2 pr-3" : "border-l-2 pl-3"}
          style={{ borderColor: isUser ? "var(--accent)" : "var(--rule)" }}
        >
          <p className="whitespace-pre-wrap text-[14.5px] leading-relaxed">
            {msg.text}
          </p>
        </div>
      </div>
    </motion.div>
  );
}

function Thinking() {
  return (
    <div className="mb-5 flex items-center gap-2">
      <span className="annot">Architect</span>
      <span className="flex gap-1" aria-label="Thinking">
        {[0, 1, 2].map((i) => (
          <motion.span
            key={i}
            className="block h-1 w-1"
            style={{ background: "var(--ink-2)" }}
            animate={{ opacity: [0.25, 1, 0.25] }}
            transition={{ duration: 1.1, repeat: Infinity, delay: i * 0.18 }}
          />
        ))}
      </span>
    </div>
  );
}

/** Three elevations, held back so they never fight the conversation. */
function Backdrop() {
  return (
    <div
      className="pointer-events-none fixed inset-0 flex items-end justify-between px-[2vw] pb-[3vh]"
      style={{ color: "var(--ink)", opacity: 0.085 }}
      aria-hidden="true"
    >
      <TajMahal className="h-[34vh] w-auto" />
      <TanjavurTemple className="h-[46vh] w-auto" />
      <StatueOfLiberty className="h-[40vh] w-auto" />
    </div>
  );
}

/* ── brief summarisation ─────────────────────────────────────────────── */

function summarise(data: Record<string, unknown>): Brief {
  const out: Brief = {};
  const dims = data.plot_dimensions as
    | { length?: number; width?: number; unit?: string }
    | undefined;
  if (dims?.length && dims?.width) {
    out.plot = `${dims.width} × ${dims.length} ${dims.unit ?? "ft"}`;
  }
  const ctx = data.plot_context as { road_facing_sides?: string[] } | undefined;
  if (ctx?.road_facing_sides?.length) {
    out.facing = ctx.road_facing_sides.join(", ");
  }
  const floors = data.number_of_floors as number | undefined;
  if (typeof floors === "number") {
    out.floors = floors <= 1 ? "Ground" : `G+${floors - 1}`;
  }
  const rooms = data.rooms as
    | Array<{ room_type?: string; quantity?: number }>
    | undefined;
  if (rooms?.length) {
    const beds = rooms
      .filter((r) => (r.room_type ?? "").toLowerCase().includes("bedroom"))
      .reduce((n, r) => n + (r.quantity ?? 1), 0);
    out.bhk = beds ? `${beds}BHK · ${rooms.length} room types` : `${rooms.length} room types`;
  }
  if (data.vastu_compliant) out.vastu = true;
  return out;
}
