import React from "react";
import {
  AbsoluteFill,
  Audio,
  interpolate,
  staticFile,
  useCurrentFrame,
  useVideoConfig
} from "remotion";
import {Video} from "@remotion/media";
import type {ClipProps} from "./types";
import {KineticSubtitles} from "./subtitles";

const clamp = (value: number) => Math.max(0, Math.min(1, value));
const finite = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

type Crop = {x: number; y: number; width: number; height: number};

const validCrop = (value: unknown): value is Crop => {
  if (!value || typeof value !== "object") return false;
  const crop = value as Partial<Crop>;
  return (
    finite(crop.x) &&
    finite(crop.y) &&
    finite(crop.width) &&
    finite(crop.height) &&
    crop.x >= 0 &&
    crop.y >= 0 &&
    crop.width > 0 &&
    crop.height > 0 &&
    crop.x + crop.width <= 1 &&
    crop.y + crop.height <= 1
  );
};

const CropVideo: React.FC<{
  src: string;
  crop: Crop;
  width: number;
  height: number;
  zoom?: number;
  focusX?: number;
  focusY?: number;
  panProgress?: number;
  panFocusX?: number;
  panFocusY?: number;
  muted?: boolean;
  centerCrop?: boolean;
}> = ({
  src,
  crop,
  width,
  height,
  zoom = 1,
  focusX = 0.5,
  focusY = 0.5,
  panProgress = 0,
  panFocusX = 0.5,
  panFocusY = 0.5,
  muted,
  centerCrop = false
}) => {
  // The source chunks are 16:9. Size the inner video from that aspect ratio,
  // then expose only the requested source rectangle through the viewport.
  const sourceAspect = 16 / 9;
  const viewportAspect = width / Math.max(1, height);
  const centeredVisibleHeight = Math.max(
    0.001,
    Math.min(crop.height, (crop.width * sourceAspect) / viewportAspect)
  );
  const centeredVisibleWidth = Math.max(
    0.001,
    (centeredVisibleHeight * viewportAspect) / sourceAspect
  );
  const centerX = crop.x + crop.width / 2;
  const centerY = crop.y + crop.height / 2;
  const visibleHeight = centerCrop ? centeredVisibleHeight : Math.max(crop.height, 0.001);
  const visibleWidth = centerCrop
    ? centeredVisibleWidth
    : Math.max((visibleHeight * viewportAspect) / sourceAspect, 0.001);
  
  // Safe bounds: ensure [targetX, targetX + visibleWidth] stays strictly within source bounds [0, 1]
  const maxAllowedX = Math.max(0, 1 - visibleWidth);
  const targetX = Math.max(0, Math.min(maxAllowedX, panFocusX - visibleWidth / 2));
  const maxAllowedY = Math.max(0, 1 - visibleHeight);
  const targetY = Math.max(0, Math.min(maxAllowedY, panFocusY - visibleHeight / 2));
  
  // Default base position centers on the focal point (gameplay action/crosshair) rather than raw left 0.0
  const baseCenterX = Math.max(0, Math.min(maxAllowedX, panFocusX - visibleWidth / 2));
  const baseCenterY = Math.max(0, Math.min(maxAllowedY, panFocusY - visibleHeight / 2));

  const visibleX = centerCrop
    ? Math.max(0, Math.min(maxAllowedX, centerX - visibleWidth / 2))
    : interpolate(panProgress, [0, 1], [baseCenterX, targetX], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp"
      });
  const visibleY = centerCrop
    ? Math.max(0, Math.min(maxAllowedY, centerY - visibleHeight / 2))
    : interpolate(panProgress, [0, 1], [baseCenterY, targetY], {
        extrapolateLeft: "clamp",
        extrapolateRight: "clamp"
      });
  const innerHeight = 100 / visibleHeight;
  const innerWidth = 100 / visibleWidth;
  const innerLeft = -visibleX * innerWidth;
  const innerTop = -visibleY * innerHeight;
  return (
    <div style={{position: "absolute", inset: 0, overflow: "hidden"}}>
      <Video
        muted={muted}
        src={staticFile(src)}
        objectFit="fill"
        style={{
          position: "absolute",
          left: `${innerLeft}%`,
          top: `${innerTop}%`,
          width: `${innerWidth}%`,
          height: `${innerHeight}%`,
          transform: `scale(${zoom})`,
          transformOrigin: `${clamp(focusX) * 100}% ${clamp(focusY) * 100}%`
        }}
      />
    </div>
  );
};

const GameplayOverlayMasks: React.FC<{
  masks: NonNullable<ClipProps["overlayMasks"]>;
  crop: Crop;
  webcamExclusion?: Crop;
  width: number;
  height: number;
  blurPx: number;
  panProgress?: number;
  panFocusX?: number;
  panFocusY?: number;
}> = ({masks, crop, width, height, blurPx, panProgress = 0, panFocusX = 0.5, panFocusY = 0.5}) => {
  const sourceAspect = 16 / 9;
  const viewportAspect = width / Math.max(1, height);
  const visibleHeight = Math.max(crop.height, 0.001);
  const visibleWidth = Math.max((visibleHeight * viewportAspect) / sourceAspect, 0.001);
  const maxAllowedX = Math.max(0, 1 - visibleWidth);
  const targetX = Math.max(0, Math.min(maxAllowedX, panFocusX - visibleWidth / 2));
  const maxAllowedY = Math.max(0, 1 - visibleHeight);
  const targetY = Math.max(0, Math.min(maxAllowedY, panFocusY - visibleHeight / 2));
  const baseCenterX = Math.max(0, Math.min(maxAllowedX, panFocusX - visibleWidth / 2));
  const baseCenterY = Math.max(0, Math.min(maxAllowedY, panFocusY - visibleHeight / 2));

  const visibleLeft = interpolate(
    panProgress,
    [0, 1],
    [baseCenterX, targetX],
    {extrapolateLeft: "clamp", extrapolateRight: "clamp"}
  );
  const visibleTop = interpolate(
    panProgress,
    [0, 1],
    [baseCenterY, targetY],
    {extrapolateLeft: "clamp", extrapolateRight: "clamp"}
  );
  const effectiveMasks = masks;
  return (
    <>
      {effectiveMasks.map((mask, index) => {
        const left = Math.max(mask.x, visibleLeft);
        const top = Math.max(mask.y, visibleTop);
        const right = Math.min(mask.x + mask.width, visibleLeft + visibleWidth);
        const bottom = Math.min(mask.y + mask.height, visibleTop + visibleHeight);
        if (right <= left || bottom <= top) return null;
        return (
          <div
            key={`${mask.kind}-${index}`}
            style={{
              position: "absolute",
              zIndex: 6,
              left: `${((left - visibleLeft) / visibleWidth) * 100}%`,
              top: `${((top - visibleTop) / visibleHeight) * 100}%`,
              width: `${((right - left) / visibleWidth) * 100}%`,
              height: `${((bottom - top) / visibleHeight) * 100}%`,
              backgroundColor: "rgba(5,5,7,0.78)",
              backdropFilter: `blur(${blurPx}px)`,
              WebkitBackdropFilter: `blur(${blurPx}px)`,
              boxShadow: "0 0 18px rgba(5,5,7,0.75)"
            }}
          />
        );
      })}
    </>
  );
};

export const ShortClip: React.FC<ClipProps> = (props) => {
  const frame = useCurrentFrame();
  const {fps, width, height} = useVideoConfig();
  const webcamHeight = height * props.layout.split_webcam_height;
  const gameplayHeight = height * props.layout.split_gameplay_height;
  const requestedLayout = props.layoutRecommendation.layout ?? "split";
  const recommendation = props.layoutRecommendation;
  const rawEventTime = recommendation.event_time;
  const rawEventEnd = recommendation.event_end;
  const hasValidatedGameplayEvent =
    recommendation.gameplay_event_validated === true &&
    finite(rawEventTime) &&
    finite(rawEventEnd) &&
    rawEventTime >= 0 &&
    rawEventEnd > rawEventTime &&
    rawEventEnd <= props.durationInSeconds;
  const eventTime = hasValidatedGameplayEvent ? rawEventTime : 0;
  const eventEnd = hasValidatedGameplayEvent ? rawEventEnd : 0;
    const targetZoom = hasValidatedGameplayEvent
    ? Math.max(1, Math.min(1.18, recommendation.gameplay_zoom ?? props.layout.zoom_scale))
    : 1;
  const zoom = hasValidatedGameplayEvent
    ? interpolate(
        frame,
        [
          eventTime * fps - fps * 0.25,
          eventTime * fps,
          eventEnd * fps,
          props.durationInSeconds * fps
        ],
        [1, targetZoom, targetZoom, targetZoom],
        {extrapolateLeft: "clamp", extrapolateRight: "clamp"}
      )
    : 1;
  const panProgress =
    !hasValidatedGameplayEvent
      ? 0
      : interpolate(
          frame,
          [
            eventTime * fps - fps * 0.25,
            eventTime * fps,
            eventEnd * fps,
            props.durationInSeconds * fps
          ],
          [0, 1, 1, 1],
          {extrapolateLeft: "clamp", extrapolateRight: "clamp"}
        );
  const focusX = clamp(recommendation.focus_x ?? props.layout.default_focus_x);
  const focusY = clamp(recommendation.focus_y ?? props.layout.default_focus_y);
  const detectedWebcamCrop =
    recommendation.webcam_box_validated === true
      ? recommendation.webcam_crop ?? recommendation.webcam_box
      : undefined;
  const webcamCrop = validCrop(detectedWebcamCrop) ? detectedWebcamCrop : props.layout.webcam;
  const configuredWebcamFullCrop = props.layout.webcam_full ?? webcamCrop;
  const webcamFullCrop = validCrop(detectedWebcamCrop)
    ? detectedWebcamCrop
    : configuredWebcamFullCrop;
  const gameplayCrop = props.layout.gameplay ?? {
    x: 0,
    y: 0,
    width: webcamCrop.x,
    height: 1
  };
  const gameplayCropWidth = Math.max(
    gameplayCrop.width,
    hasValidatedGameplayEvent
      ? recommendation.gameplay_crop_width ?? gameplayCrop.width
      : gameplayCrop.width
  );
  const gameplayCropForFrame = {
    ...gameplayCrop,
    width: interpolate(
      panProgress,
      [0, 1],
      [gameplayCrop.width, gameplayCropWidth],
      {extrapolateLeft: "clamp", extrapolateRight: "clamp"}
    )
  };
  const gameplayFocusX = clamp((focusX - gameplayCrop.x) / Math.max(gameplayCrop.width, 0.001));
  const gameplayFocusY = clamp((focusY - gameplayCrop.y) / Math.max(gameplayCrop.height, 0.001));

  const gameplayFull = requestedLayout === "gameplay_full";
  const webcamFull = requestedLayout === "webcam_full";
  const gameplayTop = gameplayFull ? 0 : webcamHeight;
  const gameplayLayerHeight = gameplayFull ? height : gameplayHeight;
  const webcamLayerHeight = webcamFull ? height : webcamHeight;

  return (
    <AbsoluteFill style={{backgroundColor: "#050505", overflow: "hidden"}}>
      {!webcamFull && (
        <div
          style={{
            position: "absolute",
            left: 0,
            top: gameplayTop,
            width,
            height: gameplayLayerHeight,
            overflow: "hidden"
          }}
        >
          <CropVideo
            src={props.source}
            crop={gameplayCropForFrame}
            width={width}
            height={gameplayLayerHeight}
            zoom={zoom}
            focusX={gameplayFocusX}
            focusY={gameplayFocusY}
            panProgress={panProgress}
            panFocusX={focusX}
            panFocusY={focusY}
            muted
          />
          <GameplayOverlayMasks
            masks={props.overlayMasks ?? []}
            crop={gameplayCropForFrame}
            webcamExclusion={webcamCrop}
            width={width}
            height={gameplayLayerHeight}
            blurPx={props.overlayBlurPx ?? 28}
            panProgress={panProgress}
            panFocusX={focusX}
            panFocusY={focusY}
          />
        </div>
      )}

      {!gameplayFull && (
        <div
          style={{
            position: "absolute",
            zIndex: 2,
            left: 0,
            top: 0,
            width,
            height: webcamLayerHeight,
            overflow: "hidden",
            borderBottom: !webcamFull ? "8px solid #000000" : "none"
          }}
        >
          <CropVideo
            src={props.source}
            crop={webcamFull ? webcamFullCrop : webcamCrop}
            width={width}
            height={webcamLayerHeight}
            muted
            centerCrop
          />
        </div>
      )}

      {/* Hook / Title between webcam and gameplay */}
      {props.title && !webcamFull && (
        <div
          style={{
            position: "absolute",
            zIndex: 15,
            left: 30,
            right: 30,
            top: webcamHeight - 42,
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            textAlign: "center",
            pointerEvents: "none",
          }}
        >
          <span
            style={{
              fontFamily: `${props.subtitleStyle.font_family}, Impact, "Rubik", "Arial Black", sans-serif`,
              fontSize: 48,
              fontWeight: 900,
              lineHeight: 1.05,
              textTransform: "uppercase",
              letterSpacing: "0.02em",
              color: "#FFE815",
              WebkitTextStroke: "10px #000000",
              paintOrder: "stroke fill",
              textShadow: "0 8px 16px rgba(0,0,0,0.95)",
            }}
          >
            {props.title}
          </span>
        </div>
      )}

      <Audio src={staticFile(props.source)} />
      <KineticSubtitles words={props.words} style={props.subtitleStyle} />
    </AbsoluteFill>
  );
};
