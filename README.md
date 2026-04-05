# Football Analysis & AI Commentary System

An end-to-end football (soccer) video analysis pipeline that uses **computer vision** to track players, referees, and the ball, then generates **AI-powered live-style commentary** with text-to-speech audio — all from a single input video.

## Demo

| Input Video | Output (with Commentary + Audio) |
|---|---|
| Raw match footage | Annotated video with player tracking, team assignment, speed/distance stats, AI commentary subtitles, and spoken audio |

## Features

- **Object Detection & Tracking** — YOLOv5 detects players, referees, and the ball in every frame
- **Team Assignment** — K-Means clustering on jersey colors to assign players to teams
- **Ball Possession** — Determines which team controls the ball frame-by-frame
- **Speed & Distance** — Calculates real-world player speed (km/h) and distance covered using perspective transformation
- **Camera Movement Estimation** — OpenCV optical flow compensates for camera panning
- **AI Commentary Generation** — LLM-powered commentary (OpenAI, Gemini, Groq, Anthropic) that reacts to real match events
- **Text-to-Speech Audio** — Commentary converted to spoken audio (Google TTS or OpenAI TTS)
- **Subtitle Overlay** — Auto-scaling commentary subtitles burned into the video
- **Multi-Provider Cascade** — Falls back to next LLM provider if one fails
- **Configurable via `.env`** — All settings (provider, max events, gap, mode) controlled from a single file

## Architecture

```
Input Video
    │
    ▼
┌─────────────────┐
│  YOLO Detection  │ ← models/best.pt (trained on Roboflow dataset)
│  & Tracking      │
└────────┬────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐ ┌──────────┐
│ Team   │ │ Ball     │
│ Assign │ │ Possess  │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌─────────────────────┐
│ Speed & Distance    │ ← View Transformer (pixel → meters)
│ Camera Movement     │ ← Optical Flow
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Event Detection     │ ← Possession changes, sprints, fast ball
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ LLM Commentary      │ ← OpenAI / Gemini / Groq / Anthropic
└────────┬────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐ ┌──────────┐
│Subtitle│ │ TTS      │ ← gTTS / OpenAI TTS
│Overlay │ │ Audio    │
└───┬────┘ └────┬─────┘
    │           │
    ▼           ▼
┌─────────────────────┐
│ ffmpeg Mux          │
└────────┬────────────┘
         │
         ▼
    Output Videos
    ├── output_video_final.mp4      (subtitles, no audio)
    └── output_video_with_audio.mp4 (subtitles + spoken commentary)
```

## Project Structure

```
Techferry-Assignment/
├── main.py                          # Main pipeline orchestrator
├── .env                             # API keys & config (not committed)
├── .env.example                     # Template for .env
├── requirements.txt                 # Python dependencies
│
├── models/
│   └── best.pt                      # Trained YOLOv5 weights
│
├── input_videos/                    # Place input videos here
│   └── 08fd33_4.mp4
│
├── output_videos/                   # Generated outputs
│   ├── output_video.mp4             # Annotated (no commentary)
│   ├── output_video_final.mp4       # With subtitles
│   └── output_video_with_audio.mp4  # With subtitles + audio
│
├── stubs/                           # Cached tracking data per video
│   ├── {video_name}_track.pkl
│   └── {video_name}_camera.pkl
│
├── commentary/                      # AI Commentary module
│   ├── __init__.py
│   ├── commentary_generator.py      # LLM-based commentary generation
│   ├── event_detector.py            # Detects match events from tracking data
│   ├── subtitle_overlay.py          # Burns subtitles onto video frames
│   └── tts_generator.py             # Text-to-speech + ffmpeg audio muxing
│
├── trackers/                        # YOLO object tracking
│   ├── __init__.py
│   └── tracker.py
│
├── team_assigner/                   # Jersey color-based team assignment
│   ├── __init__.py
│   └── team_assigner.py
│
├── player_ball_assigner/            # Ball possession detection
│   ├── __init__.py
│   └── player_ball_assigner.py
│
├── camera_movement_estimator/       # Optical flow camera compensation
│   ├── __init__.py
│   └── camera_movement_estimator.py
│
├── view_transformer/                # Pixel to real-world coordinate mapping
│   ├── __init__.py
│   └── view_transformer.py
│
├── speed_and_distance_estimator/    # Player speed & distance calculation
│   ├── __init__.py
│   └── speed_and_distance_estimator.py
│
├── utils/                           # Video I/O utilities
│   ├── __init__.py
│   ├── video_utils.py
│   └── bbox_utils.py
│
├── training/                        # YOLO training notebook & dataset
│   └── football_training_yolo_v5.ipynb
│
└── development_and_analysis/        # Development notebooks
    └── color_assignment.ipynb
```

## Files Not Included in Repository

The following files are **not pushed to Git** due to their large size or sensitivity:

| Folder / File | Reason | How to Obtain |
|---|---|---|
| `models/best.pt` | Trained YOLO weights (186MB, exceeds GitHub 100MB limit) | Download from [`models/MODEL_LINKS.md`](models/MODEL_LINKS.md) |
| `.env` | Contains secret API keys | Copy `.env.example` → `.env` and add your API keys |
| `.venv/` | Python virtual environment | Created during setup with `python -m venv .venv` |
| Output videos | 165MB+ each, exceeds GitHub limit | Download links in [`output_videos/VIDEO_LINKS.md`](output_videos/VIDEO_LINKS.md) |


> **Note:** The `stubs/` folder contains pre-cached tracking data for the demo video (`chelsea_arsenal_3min_1080p`). This only works with that specific video. For any new input video, set `read_from_stub=False` on first run to generate fresh tracking data.

> **Note:** Output demo videos are too large for GitHub (100MB+ each). Download links are available in [`output_videos/VIDEO_LINKS.md`](output_videos/VIDEO_LINKS.md).

## Setup

### Prerequisites

- Python 3.10+
- ffmpeg (for audio muxing)
- GPU recommended for faster YOLO inference (works on CPU too)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/sarayu-patel/Techferry-Assignment.git
   cd Techferry-Assignment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/Mac
   .venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install python-dotenv gtts openai google-genai
   ```

4. **Install ffmpeg**
   - Download from https://ffmpeg.org/download.html
   - Add to system PATH
   - Or install via: `pip install imageio-ffmpeg`

5. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=sk-proj-...
   GEMINI_API_KEY=AIza...
   ```

6. **Add input video**
   ```bash
   mkdir input_videos
   ```
   Place a football match video (broadcast camera angle, MP4) in `input_videos/` and update `video_path` in `main.py`.

7. **Directories auto-created on first run**
   - `stubs/` — cached tracking data (auto-generated)
   - `output_videos/` — output videos (auto-generated)
   - `output_videos/audio_clips/` — TTS audio clips (auto-generated)

## Configuration

All settings are in `.env`:

| Variable | Default | Description |
|---|---|---|
| `COMMENTARY_PROVIDER` | `openai` | Primary LLM: `openai`, `gemini`, `anthropic`, `groq` |
| `COMMENTARY_MODEL` | (auto) | Override model name |
| `COMMENTARY_MAX_EVENTS` | `15` | Max commentary lines per video |
| `COMMENTARY_MIN_GAP_SEC` | `6` | Min seconds between commentary |
| `COMMENTARY_MODE` | `both` | `subtitle`, `audio`, `both`, or `off` |
| `COMMENTARY_TTS_PROVIDER` | `gtts` | TTS engine: `gtts` (free) or `openai` |
| `SUBTITLE_DURATION_SEC` | `4.5` | How long subtitles stay on screen |
| `SUBTITLE_POSITION` | `bottom` | `bottom` or `top` |

## Usage

### Basic Usage

1. Place your football video in `input_videos/`
2. Update `video_path` in `main.py`:
   ```python
   video_path = 'input_videos/your_video.mp4'
   ```
3. Run:
   ```bash
   python main.py
   ```
4. Outputs will be in `output_videos/`

### First Run vs Subsequent Runs

**First run** with a new video — set `read_from_stub=False`:
```python
tracks = tracker.get_object_tracks(
    video_frames,
    read_from_stub=False,
    stub_path=f'stubs/{video_name}_track.pkl'
)
```
This runs YOLO detection on every frame (slow, but saves results).

**Subsequent runs** — set `read_from_stub=True`:
```python
read_from_stub=True,
```
Loads cached results instantly.

### Using Google Colab (Recommended for GPU)

For faster processing, use Google Colab with T4 GPU:

1. Upload project to Google Drive
2. Open a new Colab notebook
3. Set runtime to **GPU (T4)**
4. Mount Drive and run `python main.py`


## Model Performance

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv5x (97.2M parameters) |
| Image Size | 640×640 |
| Epochs | 100 |
| Batch Size | 16 |
| Optimizer | AdamW (lr=0.00125) |
| GPU | Tesla T4 (14.9 GB) |
| Training Time | 2.06 hours |
| Dataset | 612 train images, 38 validation images |
| Classes | 4 (ball, goalkeeper, player, referee) |

### Best Model Validation Results

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **All (Overall)** | **0.882** | **0.829** | **0.860** | **0.612** |
| Player | 0.967 | 0.983 | 0.989 | 0.794 |
| Goalkeeper | 0.891 | 0.909 | 0.977 | 0.750 |
| Referee | 0.893 | 0.940 | 0.959 | 0.676 |
| Ball | 0.777 | 0.486 | 0.513 | 0.227 |

### Key Metrics Explained

- **Precision (0.882)** — 88.2% of detections are correct (low false positives)
- **Recall (0.829)** — 82.9% of actual objects are detected (low false negatives)
- **mAP@50 (0.860)** — 86.0% mean Average Precision at 50% IoU threshold
- **mAP@50-95 (0.612)** — 61.2% mAP averaged across IoU thresholds 50%-95%

### Per-Class Observations

- **Player detection** is excellent (98.9% mAP@50) — the model reliably tracks all outfield players
- **Goalkeeper & Referee** detection is strong (95%+ mAP@50) — distinct jersey colors help identification
- **Ball detection** is the most challenging (51.3% mAP@50) — the ball is small, fast-moving, and often occluded by players

### Training Loss Progression

| Metric | Epoch 1 | Epoch 50 | Epoch 100 |
|--------|---------|----------|-----------|
| Box Loss | 1.348 | 0.902 | 0.647 |
| Class Loss | 1.666 | 0.396 | 0.299 |
| DFL Loss | 0.805 | 0.758 | 0.754 |

> **Inference Speed:** 17.2ms per image on Tesla T4 GPU (~58 FPS)


## Event Detection

The system detects three types of real match events from tracking data:

| Event | Detection Method | Priority |
|---|---|---|
| **Possession Change** | Ball switches between teams (min 30 frames of sustained possession) | 3 (highest) |
| **Fast Ball** | Ball moves > 35 px/frame (possible shot or long pass) | 2 |
| **Sprint** | Player exceeds 25 km/h | 1 |

Events are ranked by priority, spaced by `COMMENTARY_MIN_GAP_SEC`, and capped at `COMMENTARY_MAX_EVENTS`.

## Commentary Generation

The LLM receives structured event data enriched with:
- **Match time** (calculated from frame number and FPS)
- **Possession statistics** (turnover count, possession %)
- **Recent commentary history** (to avoid repetition)

Multiple prompt templates per event type ensure varied, natural-sounding commentary. The system uses a **provider cascade** — if the primary LLM fails (rate limit, network error), it automatically tries the next provider.

### Supported LLM Providers

| Provider | Model | Cost |
|---|---|---|
| OpenAI | gpt-4o-mini | Low |
| Google Gemini | gemini-2.0-flash | Free tier available |
| Groq | llama-3.3-70b-versatile | Free tier available |
| Anthropic | claude-haiku-4-5 | Low |

## Tech Stack

- **Object Detection**: YOLOv5 (Ultralytics)
- **Tracking**: ByteTrack
- **Computer Vision**: OpenCV
- **Team Clustering**: K-Means (scikit-learn)
- **Perspective Transform**: OpenCV
- **LLM Commentary**: OpenAI / Gemini / Groq / Anthropic APIs
- **Text-to-Speech**: gTTS / OpenAI TTS
- **Audio Processing**: ffmpeg
- **Configuration**: python-dotenv

## Limitations

- Camera movement estimation may fail on videos with frequent camera angle switches or heavy compression
- Player identification is by jersey color (no individual player names)
- Commentary is based on tracking events (possession, speed, ball movement) — not visual understanding of specific plays like tackles or headers
- Long videos (3+ minutes) at 1080p require significant RAM; use Google Colab or limit frames

## Author

**Sarayu Patel** — [@sarayu-patel](https://github.com/sarayu-patel)