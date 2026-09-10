import React from "react";
import {Composition} from "remotion";
import {ShortClip} from "./ShortClip";
import type {ClipProps} from "./types";

const defaultProps: ClipProps = {
  source: "placeholder.mp4",
  durationInSeconds: 30,
  renderFps: 60,
  emotionScore: 5,
  layoutRecommendation: {layout: "split", focus_x: 0.5, focus_y: 0.5},
  words: [],
  overlayMasks: [],
  overlayBlurPx: 28,
  layout: {
    webcam: {x: 0.738, y: 0.739, width: 0.262, height: 0.261},
    webcam_full: {x: 0.78, y: 0.74, width: 0.22, height: 0.26},
    gameplay: {x: 0, y: 0, width: 0.738, height: 1},
    split_webcam_height: 0.38,
    split_gameplay_height: 0.62,
    fullscreen_emotion_threshold: 8.5,
    zoom_scale: 1.28,
    default_focus_x: 0.5,
    default_focus_y: 0.5
  },
  subtitleStyle: {
    font_family: "Montserrat",
    font_size: 92,
    primary_color: "#FFE815",
    active_color: "#FFE815",
    outline_color: "#000000",
    outline_width: 14,
    shadow_blur: 18,
    words_per_group: 1
  }
};

export const RemotionRoot: React.FC = () => (
  <Composition
    id="ShortClip"
    component={ShortClip}
    durationInFrames={1800}
    fps={60}
    width={1080}
    height={1920}
    defaultProps={defaultProps}
    calculateMetadata={({props}) => {
      const renderFps = props.renderFps ?? 60;
      return {
        fps: renderFps,
        durationInFrames: Math.max(1, Math.ceil(props.durationInSeconds * renderFps))
      };
    }}
  />
);
