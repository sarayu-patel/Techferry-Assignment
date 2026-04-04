import math
import os


_PRIORITY = {
    "possession_change": 3,
    "fast_ball":         2,
    "sprint":            1,
}


class EventDetector:
    """Detects football incidents from tracking data to trigger commentary."""

    SPRINT_THRESHOLD_KMH       = 25.0
    FAST_BALL_PX_PER_FRAME     = 35.0
    MIN_POSSESSION_FRAMES      = 30
    SPRINT_COOLDOWN_FRAMES     = 72
    FAST_BALL_COOLDOWN_FRAMES  = 48
    POSSESSION_COOLDOWN_FRAMES = 48

    def detect_events(
        self,
        tracks,
        team_ball_control,
        fps: int = 24,
        max_events: int = None,
        min_gap_sec: float = None,
    ) -> list:
        max_events  = max_events  or int(os.getenv("COMMENTARY_MAX_EVENTS", 8))
        min_gap_sec = min_gap_sec or float(os.getenv("COMMENTARY_MIN_GAP_SEC", 8))
        min_gap_frames = int(min_gap_sec * fps)

        raw: list = []
        raw += self._detect_possession_changes(team_ball_control)
        raw += self._detect_sprints(tracks)
        raw += self._detect_fast_ball(tracks)

        return self._prioritize(raw, max_events=max_events, min_gap_frames=min_gap_frames)

    def _prioritize(self, events: list, max_events: int, min_gap_frames: int) -> list:
        if not events:
            return []

        for e in events:
            e.setdefault("priority", _PRIORITY.get(e["event_type"], 0))

        sorted_by_priority = sorted(events, key=lambda e: (-e["priority"], e["frame_num"]))

        selected: list = []
        for candidate in sorted_by_priority:
            if len(selected) >= max_events:
                break
            too_close = any(
                abs(candidate["frame_num"] - kept["frame_num"]) < min_gap_frames
                for kept in selected
            )
            if not too_close:
                selected.append(candidate)

        selected.sort(key=lambda e: e["frame_num"])
        print(
            f"[EventDetector] {len(events)} raw events → {len(selected)} selected "
            f"(max={max_events}, min_gap={min_gap_frames} frames)"
        )
        return selected

    def _detect_possession_changes(self, team_ball_control):
        events = []
        prev_team  = None
        run_start  = 0
        last_event_frame = -self.POSSESSION_COOLDOWN_FRAMES

        for frame_num, team in enumerate(team_ball_control):
            if team != prev_team:
                run_length = frame_num - run_start
                if (
                    prev_team is not None
                    and run_length >= self.MIN_POSSESSION_FRAMES
                    and frame_num - last_event_frame >= self.POSSESSION_COOLDOWN_FRAMES
                ):
                    events.append({
                        "frame_num":  frame_num,
                        "event_type": "possession_change",
                        "description": f"Team {int(team)} won the ball from Team {int(prev_team)}",
                        "context": {"new_team": int(team), "old_team": int(prev_team)},
                    })
                    last_event_frame = frame_num
                run_start = frame_num
                prev_team = team

        return events

    def _detect_sprints(self, tracks):
        events = []
        last_sprint: dict = {}

        for frame_num, player_tracks in enumerate(tracks["players"]):
            for player_id, info in player_tracks.items():
                speed = info.get("speed", 0) or 0
                if speed < self.SPRINT_THRESHOLD_KMH:
                    continue
                if (
                    frame_num - last_sprint.get(player_id, -self.SPRINT_COOLDOWN_FRAMES)
                    < self.SPRINT_COOLDOWN_FRAMES
                ):
                    continue

                team = info.get("team", "?")
                events.append({
                    "frame_num":  frame_num,
                    "event_type": "sprint",
                    "description": f"Player {player_id} (Team {team}) sprinting at {speed:.1f} km/h",
                    "context": {"player_id": player_id, "team": team, "speed": round(speed, 1)},
                })
                last_sprint[player_id] = frame_num

        return events

    def _detect_fast_ball(self, tracks):
        events = []
        prev_pos = None
        last_event_frame = -self.FAST_BALL_COOLDOWN_FRAMES

        for frame_num, ball_frame in enumerate(tracks["ball"]):
            pos = ball_frame.get(1, {}).get("position")

            if pos and prev_pos:
                dist = math.dist(pos, prev_pos)
                if (
                    dist >= self.FAST_BALL_PX_PER_FRAME
                    and frame_num - last_event_frame >= self.FAST_BALL_COOLDOWN_FRAMES
                ):
                    events.append({
                        "frame_num":  frame_num,
                        "event_type": "fast_ball",
                        "description": "Fast ball movement – possible shot or long pass!",
                        "context": {"ball_speed_px_per_frame": round(dist, 1)},
                    })
                    last_event_frame = frame_num

            prev_pos = pos

        return events