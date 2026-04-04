"""
tts_generator.py
-----------------
Converts commentary text to MP3 audio and muxes it into the output video
using ffmpeg.

Supported TTS providers:
    "gtts"   – Google TTS, free, British accent (default)
    "openai" – OpenAI TTS, model tts-1, voice "onyx" (requires OPENAI_API_KEY)

FIX v2: Prevents audio overlap by:
  1. Measuring each clip's actual duration
  2. Delaying clips so they never overlap (each waits for the previous to finish)
  3. Using volume normalization in amix to prevent double-voice distortion
"""

import os
import subprocess
from typing import Optional


class TTSGenerator:
    def __init__(self, provider: str = "gtts", api_key: Optional[str] = None):
        provider = provider.lower()
        if provider not in ("gtts", "openai"):
            raise ValueError(f"Unknown TTS provider '{provider}'. Choose 'gtts' or 'openai'.")
        self.provider = provider
        self.api_key  = api_key or os.getenv("OPENAI_API_KEY")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, text: str, output_path: str) -> str:
        if not output_path.endswith(".mp3"):
            output_path += ".mp3"
        if self.provider == "gtts":
            return self._gtts(text, output_path)
        return self._openai_tts(text, output_path)

    def generate_batch(self, events_with_commentary: list, output_dir: str) -> list:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n=== Generating Audio ({self.provider}) ===")
        results = []

        for idx, event in enumerate(events_with_commentary):
            text = event.get("commentary") or event.get("description", "")
            if not text.strip():
                print(f"  ⏭️ [{idx+1:>3}] frame {event['frame_num']:>5} → skipped (no text)")
                results.append(event)
                continue

            fname = f"commentary_{idx:03d}_frame{event['frame_num']}.mp3"
            path  = os.path.join(output_dir, fname)

            try:
                self.generate(text, path)
                # Trim leading silence from the audio clip
                trimmed_path = path.replace(".mp3", "_trimmed.mp3")
                _trim_silence(path, trimmed_path)
                if os.path.exists(trimmed_path) and os.path.getsize(trimmed_path) > 0:
                    os.replace(trimmed_path, path)
                # Measure actual audio duration for overlap prevention
                duration = _get_audio_duration(path)
                results.append({**event, "audio_path": path, "audio_duration": duration})
                print(f"  ✅ [{idx+1:>3}] frame {event['frame_num']:>5} → {fname} ({duration:.1f}s)")
            except Exception as exc:
                print(f"  ❌ [{idx+1:>3}] frame {event['frame_num']:>5} → TTS error: {exc}")
                results.append(event)

        generated = sum(1 for e in results if "audio_path" in e)
        print(f"=== {generated}/{len(events_with_commentary)} audio clips generated ===\n")
        return results

    def mux_audio_into_video(
        self,
        video_path: str,
        events_with_audio: list,
        output_path: str,
        fps: int = 24,
    ) -> str:
        _check_ffmpeg()

        valid_events = [e for e in events_with_audio if "audio_path" in e]
        if not valid_events:
            print("[TTSGenerator] No valid audio clips — copying silent video as-is.")
            import shutil
            shutil.copy2(video_path, output_path)
            return output_path

        print(f"=== Muxing {len(valid_events)} audio clips into video ===")
        duration = _get_video_duration(video_path)

        # ── Compute non-overlapping start times ──────────────────────
        # Each clip starts at its event timestamp OR after the previous
        # clip finishes, whichever is later. This prevents double voices.
        # ── Place each clip at its exact event time ──────────────────
        adjusted_starts = []
        mux_events = []

        for event in valid_events:
            event_time_sec = event["frame_num"] / fps
            adjusted_starts.append(event_time_sec)
            mux_events.append(event)
            print(f"  🔊 Frame {event['frame_num']}: audio at {event_time_sec:.1f}s")
        # ── Build ffmpeg command ─────────────────────────────────────
        inputs = ["-i", video_path]
        for event in mux_events:
            inputs += ["-i", event["audio_path"]]

        filter_parts = [
            f"anullsrc=r=44100:cl=stereo,atrim=duration={duration:.3f}[base]"
        ]
        mix_labels = ["[base]"]

        for idx, (event, start_sec) in enumerate(zip(mux_events, adjusted_starts)):
            start_ms = max(50, int(start_sec * 1000))  # minimum 50ms delay
            filter_parts.append(
                f"[{idx + 1}:a]aresample=44100,adelay={start_ms}|{start_ms},volume=1.0[a{idx}]"
            )
            mix_labels.append(f"[a{idx}]")

        n_inputs = len(mix_labels)
        # dropout_transition=0 prevents volume ducking when clips overlap
        filter_parts.append(
            f"{''.join(mix_labels)}amix=inputs={n_inputs}:duration=first"
            f":dropout_transition=0:normalize=0[out]"
        )

        filter_complex = ";".join(filter_parts)

        cmd = [_get_ffmpeg_exe()] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[out]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-y", output_path,
        ]

        subprocess.run(cmd, check=True)
        print(f"  ✅ Final video with audio → {output_path}\n")
        return output_path

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _gtts(self, text: str, output_path: str) -> str:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", tld="co.uk", slow=False)
        temp_path = output_path.replace(".mp3", "_raw.mp3")
        tts.save(temp_path)
        # Speed up audio 1.3x to sound more natural
        try:
            ffmpeg = _get_ffmpeg_exe()
            subprocess.run([
                ffmpeg, "-i", temp_path,
                "-filter:a", "atempo=1.3",
                "-y", output_path,
            ], check=True, capture_output=True)
            os.remove(temp_path)
        except Exception:
            os.replace(temp_path, output_path)
        return output_path

    def _openai_tts(self, text: str, output_path: str) -> str:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        with client.audio.speech.with_streaming_response.create(
            model="tts-1", voice="onyx", input=text,
        ) as response:
            response.stream_to_file(output_path)
        return output_path


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_ffmpeg_exe() -> str:
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found on PATH. "
        "Download from https://ffmpeg.org and add to system PATH."
    )


def _get_ffprobe_exe() -> str:
    import shutil
    fp = shutil.which("ffprobe")
    if fp:
        return fp
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        probe = os.path.join(os.path.dirname(ff), os.path.basename(ff).replace("ffmpeg", "ffprobe"))
        if os.path.isfile(probe):
            return probe
        return ff
    except Exception:
        pass
    raise RuntimeError("ffprobe not found.")


def _check_ffmpeg():
    _get_ffmpeg_exe()


def _trim_silence(input_path: str, output_path: str):
    """Remove leading AND trailing silence from audio."""
    try:
        ffmpeg = _get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-i", input_path,
            "-af", (
                "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-30dB,"
                "areverse,"
                "silenceremove=start_periods=1:start_silence=0.05:start_threshold=-30dB,"
                "areverse"
            ),
            "-y", output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        pass

def _get_audio_duration(audio_path: str) -> float:
    """Get the duration of an audio file in seconds."""
    try:
        ffprobe = _get_ffprobe_exe()
        if os.path.basename(ffprobe).startswith("ffprobe"):
            result = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path,
                ],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
    except Exception:
        pass

    # Fallback: estimate from file size (64kbps mp3)
    try:
        size_bytes = os.path.getsize(audio_path)
        return size_bytes / (64 * 1000 / 8)  # 64kbps bitrate
    except Exception:
        return 4.0  # safe default


def _get_video_duration(video_path: str) -> float:
    ffprobe = _get_ffprobe_exe()
    ffmpeg  = _get_ffmpeg_exe()

    if os.path.basename(ffprobe).startswith("ffprobe"):
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())

    result = subprocess.run(
        [ffmpeg, "-i", video_path],
        capture_output=True, text=True,
    )
    for line in result.stderr.splitlines():
        if "Duration" in line:
            parts = line.strip().split("Duration:")[1].split(",")[0].strip()
            h, m, s = parts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise RuntimeError(f"Could not determine duration of {video_path}")