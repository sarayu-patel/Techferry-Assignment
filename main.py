import os
import cv2
import numpy as np

# ── Load .env BEFORE anything else ────────────────────────────────────
from dotenv import load_dotenv

_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(_env_path, override=True)

# Verify API keys are loaded
print(f"[ENV] Loaded .env from: {_env_path}")
print(f"[ENV] OPENAI_API_KEY: {'SET' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
print(f"[ENV] GEMINI_API_KEY: {'SET' if os.environ.get('GEMINI_API_KEY') else 'NOT SET'}")

from utils import read_video, save_video
from trackers import Tracker
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from commentary import EventDetector, CommentaryGenerator, SubtitleOverlay, TTSGenerator


# ── Read config from .env ─────────────────────────────────────────────
_PROVIDER       = os.environ.get("COMMENTARY_PROVIDER", "openai").lower()
_MODEL          = os.environ.get("COMMENTARY_MODEL", "")
_MAX_EVENTS     = int(os.environ.get("COMMENTARY_MAX_EVENTS", 15))
_MIN_GAP_SEC    = float(os.environ.get("COMMENTARY_MIN_GAP_SEC", 6.30))
_MODE           = os.environ.get("COMMENTARY_MODE", "both").lower()
_TTS_PROVIDER   = os.environ.get("COMMENTARY_TTS_PROVIDER", "gtts").lower()
_SUB_DURATION   = float(os.environ.get("SUBTITLE_DURATION_SEC", 4.5))
_SUB_POSITION   = os.environ.get("SUBTITLE_POSITION", "bottom").lower()

# Default models per provider
_DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5",
    "groq":      "llama-3.3-70b-versatile",
}

# API key env var names
_KEY_VARS = {
    "openai":    "OPENAI_API_KEY",
    "gemini":    "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
}


def _build_provider_chain():
    model = _MODEL or _DEFAULT_MODELS.get(_PROVIDER, "gpt-4o-mini")
    api_key = os.environ.get(_KEY_VARS.get(_PROVIDER, ""), "")

    providers = [{"provider": _PROVIDER, "model": model, "api_key": api_key}]

    for p in ["openai", "gemini", "groq", "anthropic"]:
        if p == _PROVIDER:
            continue
        key = os.environ.get(_KEY_VARS.get(p, ""), "")
        if key:
            providers.append({
                "provider": p,
                "model": _DEFAULT_MODELS[p],
                "api_key": key,
            })

    return providers


def main():
    print(f"[Config] Provider: {_PROVIDER} | Mode: {_MODE} | TTS: {_TTS_PROVIDER}")
    print(f"[Config] Max events: {_MAX_EVENTS} | Min gap: {_MIN_GAP_SEC}s | Subtitle: {_SUB_DURATION}s")

    if _MODE == "off":
        print("[Config] Commentary is disabled (COMMENTARY_MODE=off). Skipping.")

    # ── 1. Read video ──────────────────────────────────────────────────────
    video_path = 'input_videos/chelsea_arsenal_3min_1080p.mp4'
    video_frames = read_video(video_path)
    
    # Limit frames to avoid memory issues
    MAX_FRAMES = 3000
    if len(video_frames) > MAX_FRAMES:
        print(f"[Info] Limiting from {len(video_frames)} to {MAX_FRAMES} frames")
        video_frames = video_frames[:MAX_FRAMES]
        
    h, w = video_frames[0].shape[:2]
    if w < 1280:
        print(f"[Info] Upscaling video from {w}x{h} to 1280x720...")
        video_frames = [cv2.resize(f, (1280, 720)) for f in video_frames]

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    cap.release()
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # ── 2. Track objects  [CACHED → stubs/track_stubs.pkl] ────────────────
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=f'stubs/{video_name}_track.pkl'
    )
    tracker.add_position_to_tracks(tracks)
    
     # Trim tracks to match limited frames
    for obj_type in ["players", "ball", "referees"]:
        if obj_type in tracks:
            tracks[obj_type] = tracks[obj_type][:len(video_frames)]

    # ── 3. Camera movement  [CACHED → stubs/camera_movement_stub.pkl] ─────
    # ── 3. Camera movement  [CACHED → stubs/camera_movement_stub.pkl] ─────
    try:
        camera_estimator = CameraMovementEstimator(video_frames[0])
        camera_movement_per_frame = camera_estimator.get_camera_movement(
            video_frames,
            read_from_stub=True,
            stub_path=f'stubs/{video_name}_camera.pkl'
        )
        camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)
    except Exception as e:
        print(f"[Warning] Camera movement estimation failed: {e}")
        print("[Warning] Using zero camera movement — continuing without adjustment.")
        camera_movement_per_frame = [[0, 0]] * len(video_frames)
        # Add position_adjusted = position since no camera adjustment
        for obj_type in ["players", "ball", "referees"]:
            if obj_type not in tracks:
                continue
            for frame_data in tracks[obj_type]:
                for track_id, info in frame_data.items():
                    if "position" in info:
                        info["position_adjusted"] = info["position"]
    # ── 4. View transformer — pixels → real metres ─────────────────────────
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # ── 5. Speed & distance ────────────────────────────────────────────────
    speed_estimator = SpeedAndDistance_Estimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # ── 6. Assign teams ────────────────────────────────────────────────────
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], track["bbox"], player_id
            )
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = \
                team_assigner.team_colors[team]

    # ── 7. Assign ball possession ──────────────────────────────────────────
    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", None)
        if ball_bbox:
            assigned_player = player_assigner.assign_ball_to_player(
                player_track, ball_bbox
            )
            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                team_ball_control.append(
                    tracks["players"][frame_num][assigned_player]["team"]
                )
            else:
                team_ball_control.append(
                    team_ball_control[-1] if team_ball_control else 1
                )
        else:
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else 1
            )

    team_ball_control = np.array(team_ball_control)
    # ── Free memory before drawing ────────────────────────────────────────
    import gc
    gc.collect()
    print(f"[Info] Processing {len(video_frames)} frames at {video_frames[0].shape}")
    # ── Free memory before drawing ────────────────────────────────────────
    import gc
    del camera_movement_per_frame
    gc.collect()


    # ── 8. Draw base annotations ───────────────────────────────────────────
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )
    try:
        output_video_frames = camera_estimator.draw_camera_movement(
            output_video_frames, camera_movement_per_frame
        )
    except Exception:
        print("[Warning] Skipping camera movement overlay.")
        
    output_video_frames = speed_estimator.draw_speed_and_distance(
        output_video_frames, tracks
    )

    # ── 9. Save annotated video (always saved) ────────────────────────────
    os.makedirs('output_videos', exist_ok=True)
    annotated_path = 'output_videos/output_video.mp4'
    save_video(output_video_frames, annotated_path)
    print(f"✅ Annotated video saved → {annotated_path}")

    # ── Skip commentary if mode is off ────────────────────────────────────
    if _MODE == "off":
        print("🎬 Done! Commentary disabled.")
        return

    # ── 10. Detect events ─────────────────────────────────────────────────
    detector = EventDetector()
    total_frames = len(video_frames)
    video_duration_sec = total_frames / fps

    events = detector.detect_events(
        tracks, team_ball_control,
        fps=fps,
        max_events=_MAX_EVENTS,
        min_gap_sec=_MIN_GAP_SEC,
    ) 

   # Create a short opening event at 1 second if first event is too late
   # Create a real event at 1 second if first event is too late
    if events and events[0]["frame_num"] > fps * 2:
        first_team = int(team_ball_control[fps]) if len(team_ball_control) > fps else 1
        early_event = {
            "frame_num":  fps,
            "event_type": "opening",
            "priority":   3,
            "description": f"Team {first_team} has the ball as the game begins",
            "context":     {"team": first_team},
            "short": True,
        }
        events.insert(0, early_event)

    print(f"[Events] {len(events)} events for {video_duration_sec:.0f}s video:")
    for e in events:
        print(f"  frame={e['frame_num']:>4} time={e['frame_num']/fps:.1f}s type={e['event_type']}")

    # ── 11. Generate AI commentary ────────────────────────────────────────
    providers = _build_provider_chain()
    chain_str = ", ".join(f"{p['provider']}/{p['model']}" for p in providers)
    print(f"\n[Commentary] Provider chain: {chain_str}")

    commentary_gen = CommentaryGenerator(providers=providers, fps=fps)
    events_with_commentary = commentary_gen.generate_batch(events)

    # ── 12. Apply subtitles (if mode includes subtitle) ───────────────────
    if _MODE in ("subtitle", "both"):
        subtitle_overlay = SubtitleOverlay(
            display_duration_sec=_SUB_DURATION,
            fps=fps,
            font_scale=1.2,
            position=_SUB_POSITION,
        )
        output_video_frames = subtitle_overlay.apply(
            output_video_frames, events_with_commentary
        )

    # Save video WITHOUT audio
    no_audio_path = 'output_videos/output_video_final.mp4'
    save_video(output_video_frames, no_audio_path)
    print(f"✅ Video WITHOUT audio → {no_audio_path}")

    # ── 13-14. Generate TTS audio and mux (if mode includes audio) ────────
    if _MODE in ("audio", "both"):
        silent_path = 'output_videos/_silent_temp.mp4'
        save_video(output_video_frames, silent_path)

        tts_gen = TTSGenerator(provider=_TTS_PROVIDER)
        events_with_audio = tts_gen.generate_batch(
            events_with_commentary,
            output_dir='output_videos/audio_clips'
        )

        try:
            tts_gen.mux_audio_into_video(
                video_path=silent_path,
                events_with_audio=events_with_audio,
                output_path='output_videos/output_video_with_audio.mp4',
                fps=fps
            )
            print("✅ Video WITH audio → output_videos/output_video_with_audio.mp4")
        except RuntimeError as e:
            print(f"\n[Warning] Audio mux failed: {e}")

        # Clean up temp file
        if os.path.exists(silent_path):
            os.remove(silent_path)

    print(f"\n🎬 Done! Output videos in output_videos/")
    if _MODE in ("subtitle", "both"):
        print(f"   1. No audio  → {no_audio_path}")
    if _MODE in ("audio", "both"):
        print(f"   2. With audio → output_videos/output_video_with_audio.mp4")


if __name__ == "__main__":
    main()