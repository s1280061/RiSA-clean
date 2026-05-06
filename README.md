# RiSA: Risk-aware Situational Assistant

RiSA (Risk-aware Situation Assessment) is a research prototype that integrates object detection, tracking, and risk prediction for driving assistance AI.
It provides an open and reproducible framework for interpretable safety reasoning in naturalistic driving scenarios, combining environment perception, trajectory forecasting, and multimodal reasoning to assist drivers with context-aware safety advice.

## Demo

![RiSA Demo](assets/output_4panel_3s.gif)

*Four-panel visualization showing driving risk zones, vehicle intentions, and predicted trajectories.*

## Evaluation Gallery

https://s1280061.github.io/RiSA-clean/eval_gallery/

---

## Features

- **Multi-stage Environment Recognition**: Scene understanding with LLaVA-based visual reasoning
- **Vehicle Detection & Tracking**: YOLOv8 + ByteTrack for robust multi-object tracking
- **Trajectory Prediction**: Seq2Seq (GRU-based) model for forecasting vehicle movements
- **Intent Classification**: Turn signal and brake light detection using custom classifiers
- **BEV Visualization**: Bird's-eye view rendering with lane detection and risk zones
- **Latency Profiling**: Built-in per-module performance measurement
- **Risk Assessment**: Real-time yellow-to-red zone transition detection

---

## System Architecture

![RiSA Architecture](./assets/RiSA-02_architecture_v2.drawio.png)

The multi-stage pipeline includes:

1. **Perception** – YOLOv8-based detection and ByteTrack tracking
2. **Trajectory Prediction** – GRU/Seq2Seq forecasting of future motion
3. **Intent Classification** – Turn and brake signal recognition
4. **Multimodal Reasoning** – LLaVA-based risk assessment
5. **Visualization** – BEV risk-zone rendering and latency profiling

---

## Quick Start

### Option A: Docker (Recommended)

Docker provides the easiest reproducible environment. No manual dependency installation required.

**Prerequisites**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- NVIDIA GPU + drivers (CPU-only fallback is supported)
- For GPU acceleration: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**1. Clone and build**

```bash
git clone https://github.com/s1280061/RiSA-clean.git
cd RiSA-clean
docker compose build
```

Build takes approximately 2–3 minutes (base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`).

**2. Configure data paths**

Edit the `volumes` section in `docker-compose.yml`:

```yaml
volumes:
  - /path/to/your/videos:/data    # input video and CSV files
  - ./outputs:/app/outputs        # results will be written here
  - ./models:/app/models          # model weight files
```

**3. Run**

```bash
docker compose run --rm risa python run_demo.py \
  --video /data/scene_020.mp4 \
  --csv   /data/scene_020.csv \
  --folder /app/outputs
```

> **LLaVA note**: Risk assessment requires a running [Ollama](https://ollama.com/) instance.
> Without it, LLaVA warnings are suppressed and the rest of the pipeline continues normally.
> CPU-only mode processes approximately 449 frames in ~54 seconds.

---

### Option B: Manual Setup

**1. Clone the repository**

```bash
git clone https://github.com/s1280061/RiSA-clean.git
cd RiSA-clean
```

**2. Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

**3. Install PyTorch**

Choose the command that matches your CUDA version. Visit [pytorch.org](https://pytorch.org/get-started/locally/) for other configurations.

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio
```

**4. Install ByteTrack**

```bash
git clone https://github.com/ifzhang/ByteTrack.git
pip install -e ./ByteTrack
pip install cython-bbox lap loguru
```

> **Windows only**: Before installing, open `ByteTrack/setup.py` and change:
> ```python
> # Before
> with open("README.md") as f:
> # After
> with open("README.md", encoding="utf-8") as f:
> ```

**5. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

**6. Set Python paths**

Either add to the top of `run_demo.py`:

```python
import sys
sys.path.insert(0, 'src/train')
sys.path.insert(0, 'src/perception')
sys.path.insert(0, 'src/utils')
```

Or export as an environment variable:

```bash
# Linux / macOS
export PYTHONPATH="src/train:src/perception:src/utils:$PYTHONPATH"

# Windows (Command Prompt)
set PYTHONPATH=src/train;src/perception;src/utils;%PYTHONPATH%
```

**7. Prepare model weights**

Download your trained checkpoints and place them as follows:

```
RiSA-clean/
├── models/
│   └── classify/
│       ├── turn_best.pt      # Turn signal classifier
│       └── brake_best.pt     # Brake light classifier
└── 26x/
    └── checkpoints_traj_px_best15/
        └── best_ade_px.pt    # Trajectory predictor
```

If your paths differ, update the corresponding variables at the top of `run_demo.py`.

---

## Usage

```bash
python run_demo.py \
  --video  path/to/scene_020.mp4 \
  --csv    path/to/scene_020.csv \
  --folder path/to/output/
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--video` | Yes | Path to input `.mp4` video |
| `--csv`   | No  | CSV file with per-frame speed data |
| `--folder`| Yes | Directory where outputs will be saved |

### Input CSV format

```
Frame,Speed
0,45.2
1,45.5
2,45.8
```

The `SPEED_COL_UNIT` parameter in `run_demo.py` controls the unit (`"mph"`, `"mps"`, or `"kmh"`).

### Output files

| File | Description |
|------|-------------|
| `scene_020_with_bev.mp4` | Annotated video with BEV overlay and risk zones |
| `scene_020_with_bev_traj.csv` | Per-frame vehicle trajectories and metadata |
| `scene_020_context.json` | Full scene context per frame |
| `scene_020_llava.json` | LLaVA risk assessment results |
| `scene_020_latency.csv` / `.jsonl` | Per-module latency profiling |
| `scene_020_risk_transition.csv` | Yellow-to-red zone transition events |
| `scene_020_risk_transition_series.csv` | Frame-by-frame data during each transition |

---

## Configuration

Key parameters are defined at the top of `run_demo.py`:

```python
# Speed unit for input CSV
SPEED_COL_UNIT = "mph"       # "mph" | "mps" | "kmh"

# BEV viewport
BEV_FRONT_M = 35             # Forward view distance (meters)
BEV_RIGHT_M = 3.5            # Right lane half-width (meters)
BEV_LEFT_M  = 3.5            # Left lane half-width (meters)

# Intent classifier smoothing windows
turn_window_frames  = 15     # Turn signal voting window
brake_window_frames = 10     # Brake light voting window

# Trajectory prediction horizon
H_PAST = 30                  # Input history length (frames)
H_FUT  = 45                  # Prediction horizon (frames)
```

---

## Risk Transition Analysis

RiSA automatically detects transitions from yellow to red risk zones.

**Transition criteria**

| Zone | Condition |
|------|-----------|
| Yellow | LLaVA assesses risk as `high` or `very_high` |
| Red | Predicted trajectory enters stopping distance **and** ego speed ≥ 40 km/h |

**Output files**

| File | Description |
|------|-------------|
| `scene_020_risk_transition.csv` | Summary of each transition event |
| `scene_020_risk_transition_series.csv` | Frame-by-frame data during transitions |
| `risk_transition_summary.json` | Aggregated statistics across multiple scenes |

---

## Performance Profiling

The built-in latency tracer records execution time per module:

| Key | Module |
|-----|--------|
| `yolo_detect` | Object detection |
| `byte_track` | Multi-object tracking |
| `seq2seq_pred` | Trajectory prediction |
| `turn_cls` / `brake_cls` | Intent classification |
| `stage1_env` | Environment recognition |
| `llava_assess` | LLaVA risk assessment |
| `visualization` | Frame rendering |
| `video_write` | Video encoding |

Each record includes start/end timestamps, frame index, and GPU synchronization markers.

---

## Troubleshooting

**CUDA out of memory**
- Reduce frame resolution or process a shorter video segment.
- Set `device = "cpu"` in `run_demo.py`.

**ByteTrack install fails on Windows**
- Apply the `encoding="utf-8"` fix to `ByteTrack/setup.py` described in step 4 above.
- Alternatively, copy a pre-built `yolox/` folder from an existing environment.

**Missing model weights**
- Verify that all `.pt` files exist at the paths defined in `run_demo.py`.
- Check that the `models/` and `26x/` directories follow the structure shown in step 7.

**LLaVA: connection refused**
- Start the Ollama server: `ollama serve`
- Pull the required model: `ollama pull llava`
- Or set `OLLAMA_HOST` to your Ollama server address in `docker-compose.yml`.
- If Ollama is unavailable, risk assessment is skipped automatically and processing continues.

**CSV frame offset warning**
- Verify that the CSV contains a `Frame` column starting from the correct index (0-indexed).

---

## Dataset

RiSA was developed using the **SHRP2 Naturalistic Driving Study (NDS)** dataset, which contains privacy-sensitive real-world driving data collected under restricted research agreements and **cannot be publicly distributed**.

To reproduce experiments, apply for access at:  
🔗 [SHRP2 NDS Data Access Portal](https://insight.shrp2nds.us/)

Once approved, organize data as:

```
videos/
└── scene_000.mp4
csv_divided/
└── scene_000.csv
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@inproceedings{asai2025risa,
  title     = {RiSA: Risk-aware Situational Assistant: From Risk Forecasting to Actionable Driver Advice},
  author    = {Kaito Asai and Bin Zhou and Jianyu Huang and Yutaka Arakawa and Tsunenori Mine},
  booktitle = {Proceedings of the IEEE International Conference on Big Data (IEEE BigData)},
  year      = {2025},
  note      = {Accepted (to appear)},
}
```

---

## Acknowledgments

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **ByteTrack**: [ByteTrack Repository](https://github.com/ifzhang/ByteTrack)
- **LLaVA**: [LLaVA Model](https://github.com/haotian-liu/LLaVA)
- **SHRP2 Dataset**: [SHRP2 NDS](https://insight.shrp2nds.us/)

---

## Contact

- Email: asai.kaito@arakawa-lab.com
- GitHub Issues: [Create an issue](https://github.com/s1280061/RiSA-clean/issues)

> This is a research prototype. It is not intended for production use.