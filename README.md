# RiSA: Risk-aware Situational Assistant

RiSA (Risk-aware Situation Assessment) is a research prototype that integrates object detection, tracking, and risk prediction for driving assistance AI.  
It is an open and reproducible research framework for interpretable safety reasoning in naturalistic driving scenarios, combining environment perception, trajectory forecasting, and multimodal reasoning to assist drivers with context-aware safety advice.

## Demo

Below is a short demonstration of the RiSA system in action:

![RiSA Demo](assets/output_4panel_3s.gif)

*(Four-panel visualization showing driving risk zones, vehicle intentions, and predicted trajectories.)*

## Evaluation Gallery

https://s1280061.github.io/RiSA-clean/eval_gallery/

## Features

- **Multi-stage Environment Recognition**: Scene understanding with LLaVA-based visual reasoning
- **Vehicle Detection & Tracking**: YOLOv8 + ByteTrack for robust multi-object tracking
- **Trajectory Prediction**: Seq2Seq model for forecasting vehicle movements
- **Intent Classification**: Turn signal and brake light detection using custom classifiers
- **BEV Visualization**: Bird's-eye view rendering with lane detection and risk zones
- **Latency Profiling**: Built-in performance measurement for each processing module
- **Risk Assessment**: Real-time yellow-to-red zone transition detection

## System Architecture

![RiSA Architecture](./assets/RiSA-02_architecture_v2.drawio.png)

The multi-stage pipeline includes:
1. **Perception** – YOLOv8-based detection and ByteTrack tracking
2. **Trajectory Prediction** – GRU/Seq2Seq forecasting of future motion
3. **Intent Classification** – Turn and brake signal recognition
4. **Multimodal Reasoning** – LLaVA-based risk assessment
5. **Visualization** – BEV risk-zone rendering and latency profiling

---

## 🐳 Quick Start with Docker (Recommended)

Docker is the easiest way to run RiSA with a reproducible environment. No manual dependency installation required.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- NVIDIA GPU + drivers (CPU fallback also works without additional setup)
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### 1. Clone and build

```bash
git clone https://github.com/s1280061/RiSA-clean.git
cd RiSA-clean
docker compose build
```

Build time is approximately 2–3 minutes (uses `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` as base).

### 2. Configure data paths

Edit the `volumes` section in `docker-compose.yml` to point to your data:

```yaml
volumes:
  - /path/to/your/videos:/data        # input video/csv directory
  - ./outputs:/app/outputs            # output results
  - ./models:/app/models              # model weights
```

### 3. Run

```bash
docker compose run --rm risa python run_demo.py \
  --video /data/path/to/scene_020.mp4 \
  --csv /data/path/to/scene_020.csv \
  --folder /app/outputs
```

### Notes

- GPU support requires NVIDIA Container Toolkit. See [setup guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
- CPU-only mode works out of the box and has been verified to process 449 frames in ~54 seconds.
- LLaVA-based risk assessment requires a running [Ollama](https://ollama.com/) instance. Without it, the system continues processing with LLaVA warnings suppressed.

---

## ⚙️ Manual Setup (Alternative)

### 1. Clone the repository

```bash
git clone https://github.com/s1280061/RiSA-clean.git
cd RiSA-clean
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/macOS
```

### 3. Install PyTorch (GPU)

```bash
# CUDA 12.1 example
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install ByteTrack

ByteTrack requires a local editable install:

```bash
git clone https://github.com/ifzhang/ByteTrack.git
# Fix setup.py encoding (Windows):
# Open ByteTrack/setup.py and change:
#   with open("README.md") as f:
# to:
#   with open("README.md", encoding="utf-8") as f:
pip install -e ./ByteTrack
```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
pip install cython-bbox lap loguru
```

### 6. Add source paths

Add the following to the top of `run_demo.py`:

```python
import sys
sys.path.insert(0, 'src/train')
sys.path.insert(0, 'src/perception')
sys.path.insert(0, 'src/utils')
```

Or set the environment variable:

```bash
export PYTHONPATH="src/train:src/perception:src/utils:$PYTHONPATH"
```

### 7. Prepare model weights

Place your model checkpoints in the following structure:

```
models/
├── classify/
│   ├── turn_best.pt     # Turn signal classifier
│   └── brake_best.pt    # Brake light classifier
26x/
└── checkpoints_traj_px_best15/
    └── best_ade_px.pt   # Trajectory predictor
```

Update paths in `run_demo.py` if your directory structure differs.

---

## Usage

### Basic Execution

```bash
python run_demo.py \
  --video path/to/scene_020.mp4 \
  --csv path/to/scene_020.csv \
  --folder path/to/output/
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--video` | Path to input video file (`.mp4`) |
| `--csv` | Path to CSV file with frame and speed data (optional) |
| `--folder` | Output directory for results |

### Output Files

| File | Description |
|------|-------------|
| `scene_020_with_bev.mp4` | Annotated video with BEV and risk zones |
| `scene_020_with_bev_traj.csv` | Vehicle trajectories and metadata |
| `scene_020_context.json` | Full scene context per frame |
| `scene_020_llava.json` | LLaVA risk assessment results |
| `scene_020_latency.csv` / `.jsonl` | Per-module latency profiling |
| `scene_020_risk_transition.csv` | Yellow-to-red zone transition events |
| `scene_020_risk_transition_series.csv` | Frame-by-frame data during transitions |

---

## 🔧 Configuration

Key parameters in `run_demo.py`:

```python
# Speed unit
SPEED_COL_UNIT = "mph"       # Options: "mph", "mps", "kmh"

# BEV view
BEV_FRONT_M = 35             # Forward view distance (meters)
BEV_RIGHT_M = 3.5            # Right lane width (meters)
BEV_LEFT_M  = 3.5            # Left lane width (meters)

# Tracking smoothing
turn_window_frames  = 15     # Turn signal window
brake_window_frames = 10     # Brake light window

# Trajectory prediction
H_PAST = 30                  # History frames
H_FUT  = 45                  # Future frames to predict
```

---

## Performance Profiling

The built-in latency tracer logs execution time per module:

| Module | Description |
|--------|-------------|
| `yolo_detect` | Object detection |
| `byte_track` | Multi-object tracking |
| `seq2seq_pred` | Trajectory prediction |
| `turn_cls` / `brake_cls` | Intent classification |
| `stage1_env` | Environment recognition |
| `llava_assess` | Risk assessment |
| `visualization` | Frame rendering |
| `video_write` | Output encoding |

---

## Risk Transition Analysis

RiSA automatically detects yellow-to-red risk zone transitions:

**Transition Criteria:**
- **Yellow Zone**: LLaVA detects high/very_high risk
- **Red Zone**: Predicted trajectory enters stopping distance AND ego speed ≥ 40 km/h

**Output:**
- `scene_020_risk_transition.csv` – Per-event summary
- `scene_020_risk_transition_series.csv` – Frame-by-frame transition data
- `risk_transition_summary.json` – Cross-scene aggregation

---

## Dataset Access

RiSA was developed and evaluated using the **SHRP2 Naturalistic Driving Study (NDS)** dataset. This dataset contains privacy-sensitive real-world driving data collected under restricted research agreements and **cannot be publicly distributed**.

Researchers may apply for access at:  
🔗 [SHRP2 NDS Data Access Portal](https://insight.shrp2nds.us/)

Once approved, organize data as:
```
videos/
└── scene_000.mp4
csv_divided/
└── scene_000.csv
```

---

## Troubleshooting

**CUDA Out of Memory**
- Reduce frame resolution or process shorter clips
- Set device to `"cpu"` in `run_demo.py`

**ByteTrack install fails on Windows**
- Fix the `setup.py` encoding issue (see Manual Setup step 4 above)
- Or copy the pre-built `yolox/` folder from an existing environment

**Missing model weights**
- Ensure `.pt` files exist at the paths defined in `run_demo.py`

**LLaVA connection refused**
- Start Ollama: `ollama serve` and pull the model: `ollama pull llava`
- Or set `OLLAMA_HOST` to your Ollama server address in `docker-compose.yml`

**CSV frame offset warning**
- Verify the CSV has a `Frame` column starting from the correct index

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

> **Note**: This is a research prototype. Use at your own risk in production environments.