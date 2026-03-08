# 🎬 paddle-sttn (STTN / ProPainter / LaMa)

A high-performance, AI-driven video tool that automatically detects subtitles and removes them using state-of-the-art inpainting. This version uses **STTN** (Spatial-Temporal Transformer Network) to ensure temporal consistency and eliminate "shivering" artifacts.

---

## ✨ Key Features

* **🔍 Smart Detection**: Automatically identifies subtitle regions across video frames.
* **🎥 Temporal Consistency**: Powered by **STTN** and **ProPainter** to reconstruct backgrounds using surrounding frames for a flicker-free result.
* **🖼️ Versatile Modes**: Supports multiple algorithms including STTN (fast/balanced), LaMa (clean static), and ProPainter (high-end motion).
* **🚀 GPU Optimized**: Fully configured for **NVIDIA GPUs** (CUDA 11.8) for rapid processing on Google Colab or local machines.
* **📦 Modern Stack**: Managed with `uv` for lightning-fast setup and reproducible environments.

---

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for ultra-fast, reliable dependency management.

1. **Clone the Repository**
2. **Setup Environment**:

   ```bash
   pip install uv  # If you don't have it
   uv sync         # Automatically creates .venv and installs everything
   ```

---

## 🚀 Usage

### Local Run

1. Place your video in the `videos/` directory.
2. Update the `VIDEO_PATH` and `OUTPUT_VIDEO` in `src/main.py` if needed.
3. Run the script:

   ```bash
   chmod +x run.sh
   ./run.sh
   ```

### Google Colab (Recommended for Free T4 GPU)

1. Open a new Colab notebook with a **T4 GPU Runtime**.
2. Run the following commands:

   ```bash
   !git clone <your-repo-url>
   %cd <your-repo-name>
   !pip install uv
   !uv sync
   !./run.sh
   ```

---

## 🧠 How It Works

1. **Scene Detection**: Optionally splits video into scenes to process motion more accurately.
2. **Detection**: Runs a character region awareness model to find exactly where subtitles are.
3. **Inpainting**:
   * **STTN/ProPainter**: Uses a transformer-based approach to "look" at previous and future frames to fill in the text area seamlessly.
   * **LaMa**: Uses large-scale fast inpainting for clean results on a per-frame basis.
4. **Merge**: Re-attaches original audio and exports the final high-quality video.

---

## 📝 License

This project is open-source and available under the MIT License.
