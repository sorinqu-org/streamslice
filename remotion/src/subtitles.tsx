import React from "react";
import {spring, useCurrentFrame, useVideoConfig} from "remotion";
import type {CaptionWord, ClipProps} from "./types";

export const KineticSubtitles: React.FC<{
  words: CaptionWord[];
  style: ClipProps["subtitleStyle"];
}> = ({words, style}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const time = frame / fps;
  const activeIndex = words.findIndex((word) => word.start <= time && time <= word.end + 0.15);
  if (activeIndex < 0) {
    return null;
  }
  const currentWord = words[activeIndex];

  return (
    <div
      style={{
        position: "absolute",
        zIndex: 20,
        left: 20,
        right: 20,
        bottom: 240,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        textAlign: "center",
        pointerEvents: "none",
      }}
    >
      <span
        style={{
          fontFamily: `${style.font_family}, Impact, "Arial Black", sans-serif`,
          fontSize: 92,
          fontWeight: 900,
          lineHeight: 1.0,
          textTransform: "uppercase",
          letterSpacing: "0.02em",
          color: "#FFE815",
          WebkitTextStroke: "14px #000000",
          paintOrder: "stroke fill",
          textShadow: "0 10px 20px rgba(0,0,0,0.95)",
          transformOrigin: "center",
          transform: `scale(${spring({
            frame: frame - currentWord.start * fps,
            fps,
            config: {damping: 12, stiffness: 280}
          })})`,
        }}
      >
        {currentWord.text}
      </span>
    </div>
  );
};
