import cv2
import numpy as np


# Colours (BGR)
_BG_COLOR     = (0, 0, 0)
_TEXT_COLOR   = (255, 255, 255)
_SHADOW_COLOR = (30, 30, 30)

# Event-type accent colours (BGR) shown as a left border stripe
_ACCENT = {
    "possession_change": (0, 200, 255),   # amber
    "sprint":            (0, 255, 100),   # green
    "fast_ball":         (0, 80,  255),   # red-orange
}
_ACCENT_DEFAULT = (200, 200, 200)


class SubtitleOverlay:
    """
    Burns commentary text onto video frames as styled subtitles.

    Parameters
    ----------
    display_duration_sec : float
        How many seconds each subtitle stays on screen (default 4 s).
    fps : int
        Video frame rate used to convert seconds → frame count.
    font_scale : float
        OpenCV font scale for the subtitle text.
    position : str
        'bottom' (default) or 'top'.
    """

    def __init__(
        self,
        display_duration_sec: float = 4.0,
        fps: int = 24,
        font_scale: float = 0.85,
        position: str = "bottom",
    ):
        self.display_frames  = max(1, int(display_duration_sec * fps))
        self.font            = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale      = font_scale
        self.font_thickness  = 2
        self.base_font_scale = font_scale
        self.base_thickness  = 2
        self.position        = position
    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, frames: list, events_with_commentary: list) -> list:
        """
        Overlay subtitles on *frames* at the timestamps defined by
        *events_with_commentary* (each must have 'frame_num' and 'commentary').

        Returns a new list of annotated frames.

        FIX: Always reads from 'commentary' key consistently — same source
        used by TTS, so subtitle text and audio always match.
        """
        frame_info: dict = {}
        for event in events_with_commentary:
            start = event["frame_num"]
            end   = min(start + self.display_frames, len(frames) - 1)
            # FIXED: use 'commentary' first (same key TTS uses), only
            # fall back to 'description' if commentary is missing/empty
            text  = event.get("commentary") or event.get("description", "")
            etype = event.get("event_type", "")

            if not text.strip():
                continue  # skip empty commentary — don't show blank subtitle

            for f in range(start, end + 1):
                # If two events overlap, keep the newer (higher priority) one
                if f not in frame_info:
                    frame_info[f] = (text, etype)

        output_frames = []
        for frame_num, frame in enumerate(frames):
            if frame_num in frame_info:
                text, etype = frame_info[frame_num]
                frame = self._draw_subtitle(frame.copy(), text, etype)
            output_frames.append(frame)

        return output_frames

    # ------------------------------------------------------------------
    # Internal drawing
    # ------------------------------------------------------------------

    def _draw_subtitle(self, frame: np.ndarray, text: str, event_type: str) -> np.ndarray:
        h, w = frame.shape[:2]
        scale_factor = w / 1920.0
        font_scale = self.base_font_scale * scale_factor
        font_thickness = max(1, int(self.base_thickness * scale_factor))
        max_text_w = w - int(60 * scale_factor)

        lines  = self._wrap_text(text, max_text_w, font_scale, font_thickness)
        line_h = int(font_scale * 40)
        pad    = 12
        accent_w = 6
        total_h  = line_h * len(lines) + pad * 2

        if self.position == "bottom":
            y_box_top = h - total_h - 30
        else:
            y_box_top = 30

        x1, y1 = 20, y_box_top
        x2, y2 = w - 20, y_box_top + total_h

        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), _BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        # Accent stripe on left edge
        accent_color = _ACCENT.get(event_type, _ACCENT_DEFAULT)
        cv2.rectangle(frame, (x1, y1), (x1 + accent_w, y2), accent_color, -1)

        # Text lines
        for i, line in enumerate(lines):
            text_x = x1 + accent_w + 10
            text_y = y1 + pad + (i + 1) * line_h - 4

            # Drop shadow
            cv2.putText(
                frame, line,
                (text_x + 2, text_y + 2),
                self.font, font_scale,
                _SHADOW_COLOR, font_thickness + 1,
                cv2.LINE_AA,
            )
            # Main text
            cv2.putText(
                frame, line,
                (text_x, text_y),
                self.font, self.font_scale,
                _TEXT_COLOR, self.font_thickness,
                cv2.LINE_AA,
            )

        return frame

    def _wrap_text(self, text: str, max_width: int, font_scale=None, font_thickness=None) -> list:
        font_scale = font_scale or self.font_scale
        font_thickness = font_thickness or self.font_thickness
        words   = text.split()
        lines: list = []
        current = ""

        for word in words:
            candidate = (current + " " + word).strip()
            (tw, _), _ = cv2.getTextSize(
                candidate, self.font, font_scale, font_thickness
            )
            if tw <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word

        if current:
            lines.append(current)

        return lines or [text]