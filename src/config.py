import warnings
from enum import Enum, unique
warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import onnxruntime as ort

# Project version number
VERSION = "1.1.1"
# ******************** [DO NOT MODIFY] start ********************
logging.disable(logging.DEBUG)  # Disable DEBUG logs
logging.disable(logging.WARNING)  # Disable WARNING logs
try:
    import torch_directml
    device = torch_directml.device(torch_directml.default_device())
    USE_DML = True
except:
    USE_DML = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
VIDEO_INPAINT_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'video')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

# Check if the full model file exists in this path, if not, merge small files to generate full file
if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

if 'ProPainter.pth' not in os.listdir(VIDEO_INPAINT_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=VIDEO_INPAINT_MODEL_PATH)

if 'infer_model.pth' not in os.listdir(os.path.dirname(STTN_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=os.path.dirname(STTN_MODEL_PATH))

# Specify ffmpeg executable path
sys_str = platform.system()
if sys_str == "Windows":
  ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
  ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
  ffmpeg_bin = os.path.join('macos', 'ffmpeg')
_bundled_ffmpeg = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if sys_str == "Windows" and 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
  fs = Filesplit()
  fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))

if sys_str == "Linux" and 'ffmpeg' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'linux_x64')):
  fs = Filesplit()
  fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'linux_x64'))

if sys_str == "Darwin" and 'ffmpeg' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'macos')):
  fs = Filesplit()
  fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'macos'))

# Make bundled binary executable if it exists, then fall back to system ffmpeg
import shutil as _shutil
if os.path.isfile(_bundled_ffmpeg):
  os.chmod(_bundled_ffmpeg, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
  FFMPEG_PATH = _bundled_ffmpeg
else:
  _system_ffmpeg = _shutil.which('ffmpeg')
  if _system_ffmpeg:
    FFMPEG_PATH = _system_ffmpeg
    print(f'[Info] Bundled ffmpeg not found, using system ffmpeg: {FFMPEG_PATH}')
  else:
    FFMPEG_PATH = _bundled_ffmpeg  # Let it fail loudly later
    print('[Warning] ffmpeg not found — audio merging will be unavailable.')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Use ONNX (DirectML/AMD/Intel)?
ONNX_PROVIDERS = []
available_providers = ort.get_available_providers()
for provider in available_providers:
    if provider in [
        "CPUExecutionProvider"
    ]:
        continue
    if provider not in [
        "DmlExecutionProvider",         # DirectML, for Windows GPU
        "ROCMExecutionProvider",        # AMD ROCm
        "MIGraphXExecutionProvider",    # AMD MIGraphX
        "VitisAIExecutionProvider",     # AMD VitisAI, for RyzenAI & Windows, performance similar to DirectML
        "OpenVINOExecutionProvider",    # Intel GPU
        "MetalExecutionProvider",       # Apple macOS
        "CoreMLExecutionProvider",      # Apple macOS
        # NOTE: CUDAExecutionProvider intentionally excluded — PaddlePaddle handles
        # NVIDIA CUDA natively and does NOT need the ONNX conversion path.
    ]:
        continue
    ONNX_PROVIDERS.append(provider)
# ******************** [DO NOT MODIFY] end ********************


@unique
class InpaintMode(Enum):
    """
    Inpaint algorithm enumeration
    """
    STTN = 'sttn'
    LAMA = 'lama'
    PROPAINTER = 'propainter'


# ******************** [MODIFIABLE] start ********************
# Whether to use h264 encoding, if you need to share generated video on Android phones, please enable this option
USE_H264 = True

# ********** General Settings start **********
"""
MODE optional algorithm types:
- InpaintMode.STTN algorithm: Better for live video, fast, can skip subtitle detection
- InpaintMode.LAMA algorithm: Better for animation, average speed, cannot skip subtitle detection
- InpaintMode.PROPAINTER algorithm: Requires large VRAM, slow, good for very high motion video
"""
# [Set inpaint algorithm]
MODE = InpaintMode.STTN
# [Set pixel deviation]
# Used to determine if it is a non-subtitle area (generally subtitle text boxes are wider than they are high)
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
# Used to expand mask size, prevents text borders or residue in inpaint stage
SUBTITLE_AREA_DEVIATION_PIXEL = 25  # Increased for cleaner coverage
# Used to determine if two text boxes are on the same line (within specified vertical pixel difference)
THRESHOLD_HEIGHT_DIFFERENCE = 20
# Used to determine if two subtitle text boxes are similar (within specified X/Y axis threshold)
PIXEL_TOLERANCE_Y = 20  # Allowed vertical deviation in pixels
PIXEL_TOLERANCE_X = 20  # Allowed horizontal deviation in pixels
# ********** General Settings end **********

# ********** InpaintMode.STTN algorithm settings start **********
# Parameters below only take effect when using STTN algorithm
STTN_SKIP_DETECTION = False  # Set to False to enable automatic subtitle detection
# Reference frame stride
STTN_NEIGHBOR_STRIDE = 5
# Reference frame length (quantity)
STTN_REFERENCE_LENGTH = 20  # Increased for better temporal consistency
# Set maximum number of frames processed simultaneously by STTN algorithm
STTN_MAX_LOAD_NUM = 100  # Increased for better results on T4 GPU
if STTN_MAX_LOAD_NUM < STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE:
    STTN_MAX_LOAD_NUM = STTN_REFERENCE_LENGTH * STTN_NEIGHBOR_STRIDE
# ********** InpaintMode.STTN algorithm settings end **********

# ********** InpaintMode.PROPAINTER algorithm settings start **********
# [Set according to your GPU VRAM size] Maximum number of images processed simultaneously
# 1280x720p: 80 frames requires 25GB VRAM, 50 frames requires 19GB VRAM
# 720x480p: 80 frames requires 8GB VRAM, 50 frames requires 7GB VRAM
PROPAINTER_MAX_LOAD_NUM = 70
# ********** InpaintMode.PROPAINTER algorithm settings end **********

# ********** InpaintMode.LAMA algorithm settings start **********
# Whether to enable high-speed mode, does not guarantee inpaint quality
LAMA_SUPER_FAST = False
# ********** InpaintMode.LAMA algorithm settings end **********
# ******************** [MODIFIABLE] end ********************
