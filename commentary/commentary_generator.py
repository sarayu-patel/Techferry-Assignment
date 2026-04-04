import os
import random

_SYSTEM_PROMPT = (
    "You are a world-class football (soccer) TV commentator known for vivid, "
    "dramatic, and varied live commentary — think Peter Drury or Martin Tyler.\n\n"
    "RULES:\n"
    "1. Generate ONE sentence of natural, broadcast-style commentary.\n"
    "2. Keep it between 10-20 words. Not too short, not too long.\n"
    "3. NEVER repeat the same phrasing across events — vary vocabulary, rhythm, "
    "   and energy level every single time.\n"
    "4. Use the match context (time, score, momentum) to shape your tone:\n"
    "   - Early game → build-up, setting the scene\n"
    "   - Mid game → tactical observations, shifts in momentum\n"
    "   - Late game → urgency, drama, tension\n"
    "5. For possession changes: describe HOW (interception, tackle, misplaced pass).\n"
    "6. For sprints: paint a picture — counter-attack, recovery run, overlapping run.\n"
    "7. For fast ball: describe it as a shot, cross, long ball, or switch of play.\n"
    "8. Add subtle filler that real commentators use: 'And it's...', "
    "   'Oh, what a...', 'Here we go...'\n"
    "9. Do NOT reference player IDs like 'Player 5' — say 'the midfielder', "
    "   'the winger', 'the defender' instead.\n"
    "10. No quotation marks. No asterisks.\n"
)

# Multiple prompt variants per event type to force variety
_EVENT_PROMPTS = {
    "intro": [
        (
            "The match has just kicked off! Two teams are battling it out on the pitch. "
            "Generate an exciting opening line as if the game just started. "
            "Set the scene with energy and anticipation."
        ),
        (
            "We are LIVE! The whistle has blown and the game is underway. "
            "Generate a thrilling opening commentary to kick things off."
        ),
        (
            "Welcome to the match! The players are on the pitch and the action begins. "
            "Create a dramatic, welcoming opening line for the broadcast."
        ),
    ],
    "atmosphere": [
        (
            "Match time: {match_time}. Team {team} is in possession, working the ball around. "
            "Generate natural commentary about the flow of play — tactical observations, "
            "player movement, or building tension. Keep it conversational like a real broadcast."
        ),
        (
            "At {match_time}, the game flows on. Team {team} has the ball. "
            "Describe what's happening — are they probing? Keeping shape? "
            "Looking for an opening? Sound like a real commentator filling natural match time."
        ),
        (
            "{match_time} on the clock. Team {team} controlling possession. "
            "Add color commentary — talk about tempo, formations, crowd atmosphere, "
            "or tactical battle. Make it feel like a live broadcast."
        ),
    ],
    "possession_change": [
        (
            "Match time: {match_time}. Team {new_team} has just won possession "
            "from Team {old_team}. They've had {possession_pct}% of recent possession. "
            "This is the {change_number}{ordinal} turnover. "
            "Describe this moment with fresh, varied commentary."
        ),
        (
            "At {match_time}, Team {old_team} loses the ball to Team {new_team}! "
            "Momentum shift — Team {new_team} now looking to build. "
            "Recent ball control was {possession_pct}% for the losing side. "
            "Give dramatic, unique commentary for this turnover."
        ),
        (
            "{match_time} on the clock. The ball changes hands — Team {new_team} "
            "takes it from Team {old_team}. This is turnover #{change_number}. "
            "Create vivid commentary that sounds different from any previous line."
        ),
    ],
    "sprint": [
        (
            "Match time: {match_time}. A player from Team {team} is surging "
            "forward at {speed} km/h — that's a serious burst of pace! "
            "Generate exciting, unique commentary for this run."
        ),
        (
            "At {match_time}, explosive acceleration from Team {team} — "
            "clocking {speed} km/h! Is this a counter-attack or a recovery run? "
            "Describe it vividly with commentary we haven't heard before."
        ),
        (
            "{match_time}: Pure pace from Team {team}, hitting {speed} km/h! "
            "The crowd reacts. Give fresh, energetic commentary."
        ),
    ],
    "fast_ball": [
        (
            "Match time: {match_time}. The ball rockets across the pitch at "
            "incredible speed — could be a shot, a cross, or a long diagonal. "
            "Generate dramatic commentary for this moment."
        ),
        (
            "At {match_time}, the ball flies through the air at blistering pace! "
            "Is it a strike at goal or a defence-splitting pass? "
            "Give vivid, unique commentary."
        ),
        (
            "{match_time} — WHOOSH! The ball zips across the field. "
            "Shot? Cross? Long ball? Build the tension with fresh commentary."
        ),
    ],
}

_ORDINALS = {1: "st", 2: "nd", 3: "rd"}

_DEFAULT_MODELS = {
    "gemini":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
    "anthropic": "claude-haiku-4-5",
    "openai":    "gpt-4o-mini",
}

_KEY_ENV = {
    "gemini":    "GEMINI_API_KEY",
    "groq":      "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
}


class CommentaryGenerator:
    """
    Generates varied, realistic football commentary using a cascade of LLM providers.

    Key improvements over v1:
    - Multiple prompt templates per event type (randomly selected) to avoid repetition
    - Richer context: match time, possession %, turnover count
    - History-aware: sends recent commentary lines to the LLM so it avoids repeating
    - Higher max_tokens to avoid truncation
    - Consistent commentary stored in 'commentary' key for both subtitle and TTS
    """

    def __init__(self, providers: list, fps: int = 24):
        if not providers:
            raise ValueError("providers list must contain at least one entry.")
        self._providers = []
        self._fps = fps
        self._history: list[str] = []  # track recent lines to avoid repetition

        for cfg in providers:
            p = cfg["provider"].lower()
            if p not in _DEFAULT_MODELS:
                raise ValueError(
                    f"Unsupported provider '{p}'. "
                    f"Choose one of: {', '.join(_DEFAULT_MODELS)}."
                )
            self._providers.append({
                "provider": p,
                "model":    cfg.get("model") or _DEFAULT_MODELS[p],
                "api_key":  cfg.get("api_key") or os.getenv(_KEY_ENV[p]),
                "_client":  None,
            })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, event: dict) -> str:
        """
        Try each provider in cascade order.
        Returns the first successful LLM response.
        Falls back to the event's own description only if every provider fails.
        """
        prompt = self._build_prompt(event)
        if event.get("short"):
            prompt += "\n\nCRITICAL: Keep this VERY short — maximum 8 words only!"
        for cfg in self._providers:
            try:
                text = self._call_llm(cfg, prompt)
                text = text.strip().strip('"').strip("'").replace("**", "").replace("*", "")
                if text and len(text) > 10:
                    self._history.append(text)
                    if len(self._history) > 10:
                        self._history.pop(0)
                    return text
            except Exception as exc:
                print(
                    f"[Commentary] {cfg['provider']}/{cfg['model']} failed "
                    f"({exc}), trying next provider..."
                )
        print("[Commentary] All providers failed. Using event description.")
        desc = event.get("description", "")
        self._history.append(desc)
        return desc

    def generate_batch(self, events: list) -> list:
        """
        Generate commentary for every event in *events*.
        Returns a new list of events with a 'commentary' key added.
        Adds enriched context (match_time, possession %, turnover count).
        """
        chain = ", ".join(
            f"{c['provider']}/{c['model']}" for c in self._providers
        )
        print(f"\n[Commentary] Provider chain: {chain}")

        self._history.clear()
        possession_change_count = 0

        results = []
        for event in events:
            # Enrich context before generating
            enriched = self._enrich_context(event, possession_change_count)
            if event["event_type"] == "possession_change":
                possession_change_count += 1

            commentary = self.generate(enriched)
            results.append({**event, "commentary": commentary})
            print(f"  [Frame {event['frame_num']:4d}] [{event['event_type']}] {commentary}")

        print(f"[Commentary] {len(results)} lines generated.\n")
        return results

    # ------------------------------------------------------------------
    # Context enrichment
    # ------------------------------------------------------------------

    def _enrich_context(self, event: dict, poss_change_count: int) -> dict:
        """Add match_time, possession stats, and turnover count to event context."""
        enriched = {**event}
        ctx = {**event.get("context", {})}

        # Calculate match time from frame number
        frame = event.get("frame_num", 0)
        total_seconds = frame / self._fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        ctx["match_time"] = f"{minutes}:{seconds:02d}"

        if event["event_type"] == "possession_change":
            ctx["change_number"] = poss_change_count + 1
            ordinal_suffix = _ORDINALS.get(
                (poss_change_count + 1) % 10
                if (poss_change_count + 1) % 100 not in (11, 12, 13)
                else 0,
                "th"
            )
            ctx["ordinal"] = ordinal_suffix
            ctx.setdefault("possession_pct", random.randint(40, 65))

        enriched["context"] = ctx
        return enriched

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, event: dict) -> str:
        templates = _EVENT_PROMPTS.get(event["event_type"])
        if not templates:
            return event.get("description", "")

        # Pick a random template to force variety
        template = random.choice(templates)
        ctx = {**event.get("context", {}), "description": event.get("description", "")}

        try:
            prompt = template.format(**ctx)
        except KeyError:
            prompt = event.get("description", "")

        # Append recent history so the LLM actively avoids repetition
        if self._history:
            recent = self._history[-5:]
            history_block = "\n".join(f"  - {line}" for line in recent)
            prompt += (
                f"\n\nIMPORTANT — Here are the last {len(recent)} commentary lines "
                f"already used. You MUST say something completely different:\n"
                f"{history_block}"
            )

        return prompt

    # ------------------------------------------------------------------
    # LLM clients
    # ------------------------------------------------------------------

    def _get_client(self, cfg: dict):
        if cfg["_client"] is not None:
            return cfg["_client"]

        provider = cfg["provider"]
        api_key  = cfg["api_key"]

        if provider == "gemini":
            from google import genai
            cfg["_client"] = genai.Client(api_key=api_key)

        elif provider == "groq":
            import openai
            cfg["_client"] = openai.OpenAI(
                api_key=api_key or "no-key",
                base_url="https://api.groq.com/openai/v1",
            )

        elif provider == "anthropic":
            import anthropic
            cfg["_client"] = anthropic.Anthropic(api_key=api_key)

        elif provider == "openai":
            import openai
            cfg["_client"] = openai.OpenAI(api_key=api_key)

        return cfg["_client"]

    def _call_llm(self, cfg: dict, prompt: str) -> str:
        client   = self._get_client(cfg)
        provider = cfg["provider"]
        model    = cfg["model"]

        if provider == "gemini":
            from google.genai import types
            response = client.models.generate_content(
                model=model,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    max_output_tokens=80,
                    temperature=0.9,
                ),
                
                contents=prompt,
            )
            return response.text

        if provider in ("groq", "openai"):
            response = client.chat.completions.create(
                model=model,
                max_tokens=80,
                temperature=0.9,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return response.choices[0].message.content

        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=80,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        raise ValueError(f"Unknown provider: {provider}")