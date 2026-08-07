/**
 * TowerScene.tsx — the scroll-linked tower.
 *
 * Scroll-LINKED, not scroll-triggered: rotation and descent are tied to
 * scroll position, so the user drives the timeline and can scrub it back and
 * forth. A triggered animation that plays on its own clock while the user
 * keeps scrolling feels detached from their input.
 *
 * Nothing calls setState per frame. The scroll progress arrives as a Framer
 * MotionValue and is read inside useFrame; React never re-renders during the
 * animation. Damping is MathUtils.damp, which is frame-rate independent —
 * a naive lerp moves twice as fast on a 120Hz display.
 *
 * Accessibility: scroll-driven 3D is a documented vestibular trigger, so
 * under prefers-reduced-motion the tower holds a fixed, composed pose and
 * the canvas stops animating entirely.
 */

import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { MotionValue } from "framer-motion";
import { buildEiffel } from "./eiffelGeometry";

function Tower({
  progress,
  reduced,
}: {
  progress: MotionValue<number>;
  reduced: boolean;
}) {
  const group = useRef<THREE.Group>(null!);
  const { primary, secondary } = useMemo(
    () => buildEiffel({ height: 10, rings: 56, braces: 2 }),
    [],
  );

  // One material instance per line class, shared by every segment.
  const inkColor = useCssColor("--ink");
  const matPrimary = useMemo(
    () => new THREE.LineBasicMaterial({ transparent: true, opacity: 0.92 }),
    [],
  );
  const matSecondary = useMemo(
    () => new THREE.LineBasicMaterial({ transparent: true, opacity: 0.34 }),
    [],
  );
  matPrimary.color = inkColor;
  matSecondary.color = inkColor;

  useFrame((_, delta) => {
    if (!group.current) return;
    if (reduced) {
      group.current.rotation.y = -0.5;
      group.current.position.y = 0;
      return;
    }
    const p = progress.get(); // 0 at top of page, 1 at the end of the hero

    // Rotate a little over half a turn across the whole scroll, and sink the
    // tower as the page advances — it descends past the reader rather than
    // the camera flying, which keeps the horizon stable and the motion legible.
    const targetRotation = -0.5 + p * Math.PI * 1.15;
    const targetY = -p * 7.5;

    group.current.rotation.y = THREE.MathUtils.damp(
      group.current.rotation.y,
      targetRotation,
      5,
      delta,
    );
    group.current.position.y = THREE.MathUtils.damp(
      group.current.position.y,
      targetY,
      5,
      delta,
    );
  });

  return (
    <group ref={group}>
      <lineSegments geometry={primary} material={matPrimary} />
      <lineSegments geometry={secondary} material={matSecondary} />
    </group>
  );
}

/** Read a themed CSS colour into a THREE.Color, reacting to theme changes. */
function useCssColor(varName: string): THREE.Color {
  const { invalidate } = useThree();
  const color = useMemo(() => new THREE.Color("#111111"), []);
  // useEffect, not useMemo: this subscribes to two listeners and must be able
  // to tear them down. useMemo discards the returned function, so the
  // MutationObserver and the media-query listener would leak on every unmount.
  useEffect(() => {
    const read = () => {
      const raw = getComputedStyle(document.documentElement)
        .getPropertyValue(varName)
        .trim();
      if (raw) {
        color.set(raw);
        invalidate();
      }
    };
    read();
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", read);
    const obs = new MutationObserver(read);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => {
      mq.removeEventListener("change", read);
      obs.disconnect();
    };
  }, [varName, color, invalidate]);
  return color;
}

export default function TowerScene({
  progress,
  reduced = false,
}: {
  progress: MotionValue<number>;
  reduced?: boolean;
}) {
  return (
    <Canvas
      // dpr clamp is the single most important 3D performance line: a 3x
      // retina phone otherwise renders nine times the pixels of a 1x display.
      dpr={[1, 2]}
      camera={{ position: [0, 0, 16], fov: 32 }}
      gl={{ antialias: true, powerPreference: "high-performance", alpha: true }}
      // Under reduced motion nothing animates, so render on demand only.
      frameloop={reduced ? "demand" : "always"}
      style={{ background: "transparent" }}
    >
      <Tower progress={progress} reduced={reduced} />
    </Canvas>
  );
}
