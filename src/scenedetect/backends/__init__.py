# -*- coding: utf-8 -*-
#
#            PySceneDetect: Python-Based Video Scene Detector
#   -------------------------------------------------------------------
#     [  Site:    https://scenedetect.com                           ]
#     [  Docs:    https://scenedetect.com/docs/                     ]
#     [  Github:  https://github.com/Breakthrough/PySceneDetect/    ]
#
# Copyright (C) 2014-2023 Brandon Castellano <http://www.bcastell.com>.
# PySceneDetect is licensed under the BSD 3-Clause License; see the
# included LICENSE file, or visit one of the above pages for details.
#
"""``scenedetect.srcs`` Module

This module contains :class:`VideoStream <scenedetect.video_stream.VideoStream>` implementations
backed by various Python multimedia libraries. In addition to creating src objects directly,
:func:`scenedetect.open_video` can be used to open a video with a specified src, falling
back to OpenCV if not available.

All srcs available on the current system can be found via :data:`AVAILABLE_BACKENDS`.

If you already have a `cv2.VideoCapture` object you want to use for scene detection, you can
use a :class:`VideoCaptureAdapter <scenedetect.srcs.opencv.VideoCaptureAdapter>` instead
of a src. This is useful when working with devices or streams, for example.

===============================================================
Video Files
===============================================================

Assuming we have a file `video.mp4` in our working directory, we can load it and perform scene
detection on it using :func:`open_video`:

.. code:: python

    from scenedetect import open_video
    video = open_video('video.mp4')

An optional src from :data:`AVAILABLE_BACKENDS` can be passed to :func:`open_video`
(e.g. `src='opencv'`). Additional keyword arguments passed to :func:`open_video`
will be forwarded to the src constructor. If the specified src is unavailable, or
loading the video fails, ``opencv`` will be tried as a fallback.

Lastly, to use a specific src directly:

.. code:: python

    # Manually importing and constructing a src:
    from scenedetect.srcs.opencv import VideoStreamCv2
    video = VideoStreamCv2('video.mp4')

In both examples above, the resulting ``video`` can be used with
:meth:`SceneManager.detect_scenes() <scenedetect.scene_manager.SceneManager.detect_scenes>`.

===============================================================
Devices / Cameras / Pipes
===============================================================

You can use an existing `cv2.VideoCapture` object with the PySceneDetect API using a
:class:`VideoCaptureAdapter <scenedetect.srcs.opencv.VideoCaptureAdapter>`. For example,
to use a :class:`SceneManager <scenedetect.scene_manager.SceneManager>` with a webcam device:

.. code:: python

    from scenedetect import SceneManager, ContentDetector
    from scenedetect.srcs import VideoCaptureAdapter
    # Open device ID 2.
    cap = cv2.VideoCapture(2)
    video = VideoCaptureAdapter(cap)
    total_frames = 1000
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(video=video, duration=total_frames)

When working with live inputs, note that you can pass a callback to
:meth:`detect_scenes() <scenedetect.scene_manager.SceneManager.detect_scenes>` to be
called on every scene detection event. See the :mod:`SceneManager <scenedetect.scene_manager>`
examples for details.
"""

# TODO(v1.0): Consider removing and making this a namespace package so that additional srcs can
# be dynamically added. The preferred approach for this should probably be:
# https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/#using-namespace-packages

# TODO: Future VideoStream implementations under consideration:
#  - Nvidia VPF: https://developer.nvidia.com/blog/vpf-hardware-accelerated-video-processing-framework-in-python/

from typing import Dict, Type

# OpenCV must be available at minimum.
from src.scenedetect.srcs.opencv import VideoStreamCv2, VideoCaptureAdapter

try:
    from scenedetect.srcs.pyav import VideoStreamAv
except ImportError:
    VideoStreamAv = None

try:
    from scenedetect.srcs.moviepy import VideoStreamMoviePy
except ImportError:
    VideoStreamMoviePy = None

# TODO(0.6.3): Replace this with a function named `get_available_srcs`.
AVAILABLE_BACKENDS: Dict[str, Type] = {
    src.BACKEND_NAME: src for src in filter(None, [
        VideoStreamCv2,
        VideoStreamAv,
        VideoStreamMoviePy,
    ])
}
"""All available srcs that :func:`scenedetect.open_video` can consider for the `src`
parameter. These srcs must support construction with the following signature:

    BackendType(path: str, framerate: Optional[float])
"""
