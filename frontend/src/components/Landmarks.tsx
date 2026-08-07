/**
 * Landmarks.tsx — Taj Mahal, Brihadeeswarar (Tanjavur), Statue of Liberty.
 *
 * These sit behind the chat as elevation drawings. They are SVG rather than
 * 3D on purpose: none of them ever rotates, so putting three more lattice
 * meshes on the page would cost frame budget and bundle weight to render
 * something that never moves. Drawn in the same line language as the tower,
 * they read as part of one drawing set.
 *
 * All strokes are currentColor, so they inherit the theme and need no
 * per-theme variants.
 */

const S = {
  fill: "none",
  stroke: "currentColor",
  strokeLinejoin: "round" as const,
  strokeLinecap: "round" as const,
};

/* ── Taj Mahal ───────────────────────────────────────────────────────── */

export function TajMahal({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 220 170"
      className={className}
      role="img"
      aria-label="Elevation drawing of the Taj Mahal"
    >
      <g {...S} strokeWidth={1.1}>
        {/* plinth */}
        <path d="M6 158 H214 M14 158 V146 H206 V158" />
        <path d="M24 146 V138 H196 V146" />

        {/* minarets */}
        {[32, 60, 160, 188].map((x, i) =>
          i === 1 || i === 2 ? null : (
            <g key={x}>
              <path d={`M${x - 7} 138 V64 M${x + 7} 138 V64`} />
              <path d={`M${x - 8} 118 H${x + 8} M${x - 8} 96 H${x + 8} M${x - 8} 78 H${x + 8}`} strokeWidth={0.6} />
              <path d={`M${x - 9} 64 H${x + 9}`} />
              <path d={`M${x - 7} 64 Q${x} 46 ${x + 7} 64 Z`} />
              <path d={`M${x} 46 V38`} strokeWidth={0.7} />
            </g>
          ),
        )}

        {/* main mass */}
        <path d="M62 138 V70 H158 V138" />
        {/* chamfered corners */}
        <path d="M62 70 L74 58 H146 L158 70" />

        {/* great iwan */}
        <path d="M92 138 V96 Q110 74 128 96 V138" />
        <path d="M99 138 V101 Q110 86 121 101 V138" strokeWidth={0.6} />
        {/* flanking arches */}
        <path d="M70 138 V110 Q78 98 86 110 V138" strokeWidth={0.7} />
        <path d="M134 138 V110 Q142 98 150 110 V138" strokeWidth={0.7} />

        {/* drum + onion dome */}
        <path d="M88 58 H132 V50 H88 Z" />
        <path
          d="M92 50 C88 26 100 8 110 6 C120 8 132 26 128 50"
          strokeWidth={1.3}
        />
        <path d="M110 6 V0" strokeWidth={0.8} />
        <path d="M104 50 H116" strokeWidth={0.6} />

        {/* chattris */}
        {[74, 146].map((x) => (
          <g key={x} strokeWidth={0.8}>
            <path d={`M${x - 10} 58 V48 M${x + 10} 58 V48`} />
            <path d={`M${x - 12} 48 H${x + 12}`} />
            <path d={`M${x - 9} 48 Q${x} 34 ${x + 9} 48 Z`} />
            <path d={`M${x} 34 V29`} strokeWidth={0.6} />
          </g>
        ))}
      </g>
      {/* centre line — a drawing convention, and it makes the symmetry read */}
      <path
        d="M110 -2 V166"
        stroke="currentColor"
        strokeWidth={0.5}
        strokeDasharray="6 3 1 3"
        opacity={0.5}
        fill="none"
      />
    </svg>
  );
}

/* ── Brihadeeswarar Temple, Thanjavur ────────────────────────────────── */

export function TanjavurTemple({ className = "" }: { className?: string }) {
  // The vimana is 13 diminishing storeys over a squat base — that repetition
  // IS the building, so it is generated rather than hand-drawn.
  const tiers = 13;
  const baseHalf = 62;
  const topHalf = 20;
  const yBase = 150;
  const yTop = 44;

  const rows = Array.from({ length: tiers }, (_, i) => {
    const t = i / (tiers - 1);
    const half = baseHalf + (topHalf - baseHalf) * Math.pow(t, 0.92);
    const y = yBase + (yTop - yBase) * t;
    return { half, y };
  });

  return (
    <svg
      viewBox="0 0 220 190"
      className={className}
      role="img"
      aria-label="Elevation drawing of the Brihadeeswarar Temple, Thanjavur"
    >
      <g {...S} strokeWidth={1.1}>
        {/* platform */}
        <path d="M8 178 H212 M18 178 V166 H202 V178" />
        {/* base storey */}
        <path d="M40 166 V150 H180 V166" />
        <path d="M48 166 V152 M64 166 V152 M156 166 V152 M172 166 V152" strokeWidth={0.5} />
        {/* entrance */}
        <path d="M100 166 V140 Q110 128 120 140 V166" strokeWidth={0.9} />

        {/* the diminishing storeys */}
        {rows.map((r, i) => {
          const next = rows[i + 1];
          if (!next) return null;
          return (
            <g key={i}>
              <path
                d={`M${110 - r.half} ${r.y} H${110 + r.half}`}
                strokeWidth={i % 2 === 0 ? 1 : 0.6}
              />
              <path
                d={`M${110 - r.half} ${r.y} L${110 - next.half} ${next.y}
                    M${110 + r.half} ${r.y} L${110 + next.half} ${next.y}`}
                strokeWidth={0.9}
              />
              {/* miniature shrines along each cornice */}
              {i % 2 === 0 &&
                [-0.62, -0.2, 0.2, 0.62].map((f) => {
                  const x = 110 + r.half * f;
                  const hgt = (r.y - next.y) * 0.55;
                  return (
                    <path
                      key={f}
                      d={`M${x - 4} ${r.y} V${r.y - hgt} H${x + 4} V${r.y}`}
                      strokeWidth={0.45}
                      opacity={0.85}
                    />
                  );
                })}
            </g>
          );
        })}

        {/* octagonal shikhara + kalasha */}
        <path d="M90 44 L110 44 L130 44 L124 30 H96 Z" strokeWidth={1.2} />
        <path d="M96 30 C98 18 122 18 124 30" strokeWidth={1.2} />
        <path d="M110 18 V6" strokeWidth={0.9} />
        <path d="M104 10 H116" strokeWidth={0.7} />
        <path d="M110 6 l-4 -5 h8 z" strokeWidth={0.7} />
      </g>
      <path
        d="M110 -2 V186"
        stroke="currentColor"
        strokeWidth={0.5}
        strokeDasharray="6 3 1 3"
        opacity={0.5}
        fill="none"
      />
    </svg>
  );
}

/* ── Statue of Liberty ───────────────────────────────────────────────── */

export function StatueOfLiberty({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 150 250"
      className={className}
      role="img"
      aria-label="Elevation drawing of the Statue of Liberty"
    >
      <g {...S} strokeWidth={1.1}>
        {/* star pedestal */}
        <path d="M10 244 H140 M20 244 V232 H130 V244" />
        <path d="M30 232 V196 H120 V232" />
        <path d="M38 196 V150 H112 V196" />
        <path d="M38 178 H112 M38 164 H112" strokeWidth={0.45} />
        {[52, 68, 82, 98].map((x) => (
          <path key={x} d={`M${x} 196 V150`} strokeWidth={0.4} opacity={0.8} />
        ))}
        {/* balcony */}
        <path d="M34 150 H116" strokeWidth={1.2} />

        {/* robe */}
        <path d="M60 150 C58 120 62 96 66 78 L84 78 C90 100 94 124 94 150 Z" />
        <path d="M68 150 C68 124 70 100 72 82" strokeWidth={0.45} opacity={0.9} />
        <path d="M78 150 C80 124 82 100 82 82" strokeWidth={0.45} opacity={0.9} />
        <path d="M86 150 C88 128 88 108 87 92" strokeWidth={0.45} opacity={0.9} />

        {/* tablet, left arm */}
        <path d="M60 104 L44 112 L52 128 L68 120 Z" strokeWidth={0.9} />
        <path d="M50 114 L60 122 M54 110 L64 118" strokeWidth={0.4} />

        {/* head + crown */}
        <path d="M70 78 C68 72 68 64 72 60 C78 56 84 60 84 68 C84 74 82 78 80 78 Z" />
        {/* seven rays */}
        {[-70, -46, -22, 0, 22, 46, 70].map((deg) => {
          const r = (deg * Math.PI) / 180;
          const cx = 76;
          const cy = 60;
          const x1 = cx + Math.sin(r) * 9;
          const y1 = cy - Math.cos(r) * 9;
          const x2 = cx + Math.sin(r) * 20;
          const y2 = cy - Math.cos(r) * 20;
          return (
            <path key={deg} d={`M${x1} ${y1} L${x2} ${y2}`} strokeWidth={0.8} />
          );
        })}

        {/* raised right arm + torch */}
        <path d="M86 84 L96 52 L104 26" strokeWidth={1.3} />
        <path d="M92 86 L102 54 L110 28" strokeWidth={1.3} />
        <path d="M100 26 H114 L112 18 H102 Z" strokeWidth={1} />
        <path d="M103 18 C104 8 110 8 111 18" strokeWidth={0.9} />
        <path d="M107 8 V2" strokeWidth={0.6} />
      </g>
      <path
        d="M76 -2 V246"
        stroke="currentColor"
        strokeWidth={0.5}
        strokeDasharray="6 3 1 3"
        opacity={0.5}
        fill="none"
      />
    </svg>
  );
}
