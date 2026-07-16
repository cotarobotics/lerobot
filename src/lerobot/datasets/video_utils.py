#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import glob
import importlib
import logging
import os
import queue
import shutil
import tempfile
import threading
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import av
import fsspec
import numpy as np
import pyarrow as pa
import torch
from datasets.features.features import register_feature
from PIL import Image

from lerobot.configs import (
    VideoEncoderConfig,
    camera_encoder_defaults,
)
from lerobot.utils.import_utils import get_safe_default_video_backend

logger = logging.getLogger(__name__)


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
    return_uint8: bool = False,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available
            in the platform; otherwise, defaults to "pyav". The legacy value "video_reader" is
            accepted for one release as an alias for "pyav" and will be removed in a future version.
        return_uint8 (bool): If True, return raw uint8 frames without float32 normalization.
            This reduces memory for DataLoader IPC; normalization can be done on GPU afterward.

    Returns:
        torch.Tensor: Decoded frames (float32 in [0,1] by default, or uint8 if return_uint8=True).

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_video_backend()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    elif backend == "pyav":
        return decode_video_frames_pyav(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    elif backend == "video_reader":
        logger.warning("backend='video_reader' is deprecated and now aliases to 'pyav'.")
        return decode_video_frames_pyav(video_path, timestamps, tolerance_s, return_uint8=return_uint8)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_pyav(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    return_uint8: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video using PyAV.

    This is the fallback decoder for platforms where torchcodec has no wheel (currently macOS
    x86_64 and linux armv7l — see the torchcodec block in pyproject.toml for the full matrix).
    On supported platforms, prefer `decode_video_frames_torchcodec`, which is faster and supports
    accurate seek.

    PyAV doesn't support accurate seek: we seek to the nearest preceding keyframe and decode
    forward until we have covered the requested timestamp range. The number of key frames in a
    video can be adjusted at encoding time to trade off decoding speed against file size.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps (in seconds) to extract frames for.
        tolerance_s: Allowed deviation in seconds between a queried timestamp and the closest
            decoded frame.
        log_loaded_timestamps: When True, log every decoded frame's timestamp at INFO level.
        return_uint8: When True, return raw uint8 frames (C, H, W). Otherwise, return float32 in
            [0, 1] range.

    Returns:
        torch.Tensor of shape (len(timestamps), C, H, W).
    """
    # TODO(rcadene): also load audio stream at the same time
    video_path = str(video_path)

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    loaded_frames: list[torch.Tensor] = []
    loaded_ts: list[float] = []

    # Seek + decode. `container.seek(offset)` with no `stream` argument expects the offset in
    # av.time_base units (microseconds). `backward=True` lands us on the nearest keyframe at or
    # before `first_ts`, so we can then decode forward until we cover `last_ts`. See:
    # https://pyav.basswood-io.com/docs/stable/api/container.html#av.container.InputContainer.seek
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(int(first_ts * av.time_base), backward=True)

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * stream.time_base)
            if log_loaded_timestamps:
                logger.info(f"frame loaded at timestamp={current_ts:.4f}")
            # Convert to CHW uint8 to match torchcodec's output layout.
            arr = frame.to_ndarray(format="rgb24")  # H, W, 3
            loaded_frames.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

    if not loaded_frames:
        raise FrameTimestampError(
            f"No frames could be decoded from {video_path} in the timestamp range [{first_ts}, {last_ts}]."
        )

    query_ts = torch.tensor(timestamps)
    loaded_ts_t = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts_t[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts_t}"
            f"\nvideo: {video_path}"
            f"\nbackend: pyav"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts_t[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if len(timestamps) != len(closest_frames):
        raise FrameTimestampError(
            f"Number of retrieved frames ({len(closest_frames)}) does not match "
            f"number of queried timestamps ({len(timestamps)})"
        )

    if return_uint8:
        return closest_frames

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255
    return closest_frames


DEFAULT_DECODER_CACHE_SIZE = 100
"""Default LRU capacity for :class:`VideoDecoderCache`.

Sized to comfortably hold a small rolling window of episodes worth of decoders
(typical recipes: 2-4 cameras per episode × tens of episodes in flight) while
bounding host RAM. Each cached entry retains a torchcodec ``VideoDecoder`` plus
an open ``fsspec`` file handle — on the order of a few MB per entry. Override
via the ``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` env var or by passing ``max_size``
to the constructor (``None`` restores the legacy unbounded behaviour).
"""


def _default_max_cache_size() -> int | None:
    raw = os.environ.get("LEROBOT_VIDEO_DECODER_CACHE_SIZE")
    if raw is None:
        return DEFAULT_DECODER_CACHE_SIZE
    raw = raw.strip().lower()
    if raw in ("", "none", "unbounded", "-1"):
        return None
    try:
        value = int(raw)
    except ValueError as e:
        raise ValueError(
            f"LEROBOT_VIDEO_DECODER_CACHE_SIZE must be an integer, 'none', or '-1'; got {raw!r}"
        ) from e
    if value <= 0:
        raise ValueError(f"LEROBOT_VIDEO_DECODER_CACHE_SIZE must be positive; got {value}")
    return value


class VideoDecoderCache:
    """Thread-safe LRU cache for torchcodec ``VideoDecoder`` instances.

    Cached entries hold a ``VideoDecoder`` plus the open ``fsspec`` file handle
    backing it. When the cache is full and a new path is requested, the
    least-recently-used entry is evicted and its file handle is closed. This
    bounds host-RAM growth when iterating over datasets with many distinct
    video files (otherwise each ``DataLoader`` worker pins every decoder it has
    ever opened until the process exits).

    Args:
        max_size: Maximum number of decoders to retain. ``None`` disables
            eviction and restores legacy unbounded behaviour. Defaults to the
            value of ``LEROBOT_VIDEO_DECODER_CACHE_SIZE`` if set, otherwise
            :data:`DEFAULT_DECODER_CACHE_SIZE`.
    """

    _SENTINEL: ClassVar[object] = object()

    def __init__(self, max_size: int | None | object = _SENTINEL):
        if max_size is VideoDecoderCache._SENTINEL:
            max_size = _default_max_cache_size()
        if max_size is not None and max_size <= 0:
            raise ValueError(f"max_size must be positive or None; got {max_size}")
        self.max_size: int | None = max_size  # type: ignore[assignment]
        self._cache: OrderedDict[str, tuple[Any, Any]] = OrderedDict()
        self._lock = Lock()

    def __contains__(self, video_path: object) -> bool:
        with self._lock:
            return str(video_path) in self._cache

    def get_decoder(self, video_path: str):
        """Get a cached decoder or create a new one, evicting LRU if at capacity."""
        if importlib.util.find_spec("torchcodec"):
            from torchcodec.decoders import VideoDecoder
        else:
            raise ImportError(
                "'torchcodec' is required but not installed. "
                "Install it with: pip install 'lerobot[dataset]' (or uv pip install 'lerobot[dataset]')"
            )

        video_path = str(video_path)

        with self._lock:
            entry = self._cache.get(video_path)
            if entry is not None:
                self._cache.move_to_end(video_path)
                return entry[0]

            file_handle = fsspec.open(video_path).__enter__()
            try:
                decoder = VideoDecoder(file_handle, seek_mode="approximate")
            except Exception:
                file_handle.close()
                raise
            self._cache[video_path] = (decoder, file_handle)

            # Evict LRU entries until we are back under the cap. We close
            # evicted file handles immediately; the associated ``VideoDecoder``
            # is released to the GC when its last reference goes away.
            if self.max_size is not None:
                while len(self._cache) > self.max_size:
                    _evicted_path, (_evicted_decoder, evicted_handle) = self._cache.popitem(last=False)
                    with contextlib.suppress(Exception):
                        evicted_handle.close()

            return decoder

    def clear(self):
        """Clear the cache and close all file handles."""
        with self._lock:
            for _, file_handle in self._cache.values():
                with contextlib.suppress(Exception):
                    file_handle.close()
            self._cache.clear()

    def size(self) -> int:
        """Return the number of cached decoders."""
        with self._lock:
            return len(self._cache)


class FrameTimestampError(ValueError):
    """Helper error to indicate the retrieved timestamps exceed the queried ones"""

    pass


_default_decoder_cache = VideoDecoderCache()


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    log_loaded_timestamps: bool = False,
    decoder_cache: VideoDecoderCache | None = None,
    return_uint8: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Args:
        video_path: Path to the video file.
        timestamps: List of timestamps to extract frames.
        tolerance_s: Allowed deviation in seconds for frame retrieval.
        log_loaded_timestamps: Whether to log loaded timestamps.
        decoder_cache: Optional decoder cache instance. Uses default if None.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    if decoder_cache is None:
        decoder_cache = _default_decoder_cache

    # Use cached decoder instead of creating new one each time
    decoder = decoder_cache.get_decoder(str(video_path))

    loaded_ts = []
    loaded_frames = []

    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps
    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]
    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=True):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logger.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    if not is_within_tol.all():
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
            " It means that the closest frame that can be loaded from the video is too far away in time."
            " This might be due to synchronization issues with timestamps during data collection."
            " To be safe, we advise to ignore this item during training."
            f"\nqueried timestamps: {query_ts}"
            f"\nloaded timestamps: {loaded_ts}"
            f"\nvideo: {video_path}"
        )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logger.info(f"{closest_ts=}")

    if not len(timestamps) == len(closest_frames):
        raise FrameTimestampError(
            f"Retrieved timestamps differ from queried {set(closest_frames) - set(timestamps)}"
        )

    if return_uint8:
        return closest_frames

    # convert to float32 in [0,1] range
    closest_frames = (closest_frames / 255.0).type(torch.float32)
    return closest_frames


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    camera_encoder: VideoEncoderConfig | None = None,
    encoder_threads: int | None = None,
    *,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""
    if camera_encoder is None:
        camera_encoder = camera_encoder_defaults()
    vcodec = camera_encoder.vcodec
    pix_fmt = camera_encoder.pix_fmt

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    if video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {video_path}. Skipping encoding.")
        return

    video_path.parent.mkdir(parents=True, exist_ok=True)

    # Get input frames
    template = "frame-" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("-")[-1].split(".")[0])
    )

    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    with Image.open(input_list[0]) as dummy_image:
        width, height = dummy_image.size

    video_options = camera_encoder.get_codec_options(encoder_threads, as_strings=True)

    # Set logging level
    if log_level is not None:
        # "While less efficient, it is generally preferable to modify logging with Python's logging"
        logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            with Image.open(input_data) as input_image:
                input_image = input_image.convert("RGB")
                input_frame = av.VideoFrame.from_image(input_image)
                packet = output_stream.encode(input_frame)
                if packet:
                    output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    if log_level is not None:
        av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


def reencode_video(
    input_video_path: Path | str,
    output_video_path: Path | str,
    camera_encoder: VideoEncoderConfig | None = None,
    encoder_threads: int | None = None,
    log_level: int | None = av.logging.WARNING,
    overwrite: bool = False,
) -> None:
    """Re-encode a video file using the given encoder configuration.

    Args:
        input_video_path: Existing video file to read.
        output_video_path: Path for the re-encoded file.
        camera_encoder: Encoder configuration. Defaults to :func:`camera_encoder_defaults`.
        encoder_threads: Optional thread count forwarded to :meth:`VideoEncoderConfig.get_codec_options`.
        log_level: libav log level while encoding, or ``None`` to leave logging unchanged. Defaults to WARNING.
        overwrite: When ``False`` and ``output_video_path`` already exists, skip and log a warning.
    """

    camera_encoder = camera_encoder or camera_encoder_defaults()

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {output_video_path}. Skipping re-encode.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    video_options = camera_encoder.get_codec_options(encoder_threads, as_strings=True)
    vcodec = camera_encoder.vcodec
    pix_fmt = camera_encoder.pix_fmt

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)

    try:
        with av.open(input_video_path, mode="r") as src:
            try:
                in_stream = src.streams.video[0]
            except IndexError as e:
                raise ValueError(f"No video stream in {input_video_path}") from e

            fps = (
                in_stream.base_rate
            )  # We allow fractional fps though LeRobotDataset only supports integer fps
            width = int(in_stream.width)
            height = int(in_stream.height)

            with av.open(
                tmp_output_video_path,
                mode="w",
                options={
                    "movflags": "faststart"
                },  # faststart is to move the metadata to the beginning of the file to speed up loading
            ) as dst:
                out_stream = dst.add_stream(vcodec, fps, options=video_options)
                out_stream.pix_fmt = pix_fmt
                out_stream.width = width
                out_stream.height = height

                for frame in src.decode(in_stream):
                    frame = frame.reformat(width=width, height=height, format=pix_fmt)
                    packet = out_stream.encode(frame)
                    if packet:
                        dst.mux(packet)

                packet = out_stream.encode()
                if packet:
                    dst.mux(packet)

        shutil.move(tmp_output_video_path, output_video_path)
    except Exception:
        Path(tmp_output_video_path).unlink(missing_ok=True)
        raise
    finally:
        if log_level is not None:
            av.logging.restore_default_callback()

    if not output_video_path.exists():
        raise OSError(f"Video re-encoding did not work. File not found: {output_video_path}.")


def concatenate_video_files(
    input_video_paths: list[Path | str],
    output_video_path: Path,
    overwrite: bool = True,
    compatibility_check: bool = False,
):
    """
    Concatenate multiple video files into a single video file using pyav.

    This function takes a list of video input file paths and concatenates them into a single
    output video file. It uses ffmpeg's concat demuxer with stream copy mode for fast
    concatenation without re-encoding.

    Args:
        input_video_paths: Ordered list of input video file paths to concatenate.
        output_video_path: Path to the output video file.
        overwrite: Whether to overwrite the output video file if it already exists. Default is True.
        compatibility_check: Whether to check if the input videos are compatible. Default is False.

    Note:
        - Creates a temporary directory for intermediate files that is cleaned up after use.
        - Uses ffmpeg's concat demuxer which requires all input videos to have the same
          codec, resolution, and frame rate for proper concatenation.
    """

    output_video_path = Path(output_video_path)

    if output_video_path.exists() and not overwrite:
        logger.warning(f"Video file already exists: {output_video_path}. Skipping concatenation.")
        return

    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    if len(input_video_paths) == 0:
        raise FileNotFoundError("No input video paths provided.")

    # This check may be skipped at recording time as videos are encoded with the same encoder config.
    if compatibility_check:
        reference_video_info = get_video_info(input_video_paths[0])
        for input_path in input_video_paths[1:]:
            video_info = get_video_info(input_path)
            if (
                video_info["video.height"] != reference_video_info["video.height"]
                or video_info["video.width"] != reference_video_info["video.width"]
                or video_info["video.fps"] != reference_video_info["video.fps"]
                or video_info["video.codec"] != reference_video_info["video.codec"]
                or video_info["video.pix_fmt"] != reference_video_info["video.pix_fmt"]
            ):
                raise ValueError(
                    f"Input video {input_path} is not compatible with the reference video {input_video_paths[0]}."
                )

    # Create a temporary .ffconcat file to list the input video paths
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffconcat", delete=False) as tmp_concatenate_file:
        tmp_concatenate_file.write("ffconcat version 1.0\n")
        for input_path in input_video_paths:
            tmp_concatenate_file.write(f"file '{str(input_path.resolve())}'\n")
        tmp_concatenate_file.flush()
        tmp_concatenate_path = tmp_concatenate_file.name

    # Create input and output containers
    input_container = av.open(
        tmp_concatenate_path, mode="r", format="concat", options={"safe": "0"}
    )  # safe = 0 allows absolute paths as well as relative paths

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_named_file:
        tmp_output_video_path = tmp_named_file.name

    output_container = av.open(
        tmp_output_video_path, mode="w", options={"movflags": "faststart"}
    )  # faststart is to move the metadata to the beginning of the file to speed up loading

    # Replicate input streams in output container
    stream_map = {}
    for input_stream in input_container.streams:
        if input_stream.type in ("video", "audio", "subtitle"):  # only copy compatible streams
            stream_map[input_stream.index] = output_container.add_stream_from_template(
                template=input_stream, opaque=True
            )

            # set the time base to the input stream time base (missing in the codec context)
            stream_map[input_stream.index].time_base = input_stream.time_base

    # Demux + remux packets (no re-encode)
    for packet in input_container.demux():
        # Skip packets from un-mapped streams
        if packet.stream.index not in stream_map:
            continue

        # Skip demux flushing packets
        if packet.dts is None:
            continue

        output_stream = stream_map[packet.stream.index]
        packet.stream = output_stream
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    shutil.move(tmp_output_video_path, output_video_path)
    Path(tmp_concatenate_path).unlink()


class _CameraEncoderThread(threading.Thread):
    """A thread that encodes video frames streamed via a queue into an MP4 file.

    One instance is created per camera per episode. Frames are received as numpy arrays
    from the main thread, encoded in real-time using PyAV (which releases the GIL during
    encoding), and written to disk. Stats are computed incrementally using
    RunningQuantileStats and returned via result_queue.
    """

    def __init__(
        self,
        video_path: Path,
        fps: int,
        vcodec: str,
        pix_fmt: str,
        codec_options: dict[str, str],
        frame_queue: queue.Queue,
        result_queue: queue.Queue,
        stop_event: threading.Event,
    ):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.codec_options = codec_options
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.stop_event = stop_event

    def run(self) -> None:
        from .compute_stats import RunningQuantileStats, auto_downsample_height_width

        container = None
        output_stream = None
        stats_tracker = RunningQuantileStats()
        frame_count = 0

        try:
            logging.getLogger("libav").setLevel(av.logging.WARNING)

            while True:
                try:
                    frame_data = self.frame_queue.get(timeout=1)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                    continue

                if frame_data is None:
                    # Sentinel: flush and close
                    break

                # Ensure HWC uint8 numpy array
                if isinstance(frame_data, np.ndarray):
                    if frame_data.ndim == 3 and frame_data.shape[0] == 3:
                        # CHW -> HWC
                        frame_data = frame_data.transpose(1, 2, 0)
                    if frame_data.dtype != np.uint8:
                        frame_data = (frame_data * 255).astype(np.uint8)

                # Open container on first frame (to get width/height)
                if container is None:
                    height, width = frame_data.shape[:2]
                    Path(self.video_path).parent.mkdir(parents=True, exist_ok=True)
                    container = av.open(str(self.video_path), "w")
                    output_stream = container.add_stream(self.vcodec, self.fps, options=self.codec_options)
                    output_stream.pix_fmt = self.pix_fmt
                    output_stream.width = width
                    output_stream.height = height
                    output_stream.time_base = Fraction(1, self.fps)

                # Encode frame with explicit timestamps
                pil_img = Image.fromarray(frame_data)
                video_frame = av.VideoFrame.from_image(pil_img)
                video_frame.pts = frame_count
                video_frame.time_base = Fraction(1, self.fps)
                packet = output_stream.encode(video_frame)
                if packet:
                    container.mux(packet)

                # Update stats with downsampled frame (per-channel stats like compute_episode_stats)
                img_chw = frame_data.transpose(2, 0, 1)  # HWC -> CHW
                img_downsampled = auto_downsample_height_width(img_chw)
                # Reshape CHW to (H*W, C) for per-channel stats
                channels = img_downsampled.shape[0]
                img_for_stats = img_downsampled.transpose(1, 2, 0).reshape(-1, channels)
                stats_tracker.update(img_for_stats)

                frame_count += 1

            # Flush encoder
            if output_stream is not None:
                packet = output_stream.encode()
                if packet:
                    container.mux(packet)

            if container is not None:
                container.close()

            av.logging.restore_default_callback()

            # Get stats and put on result queue
            if frame_count >= 2:
                stats = stats_tracker.get_statistics()
                self.result_queue.put(("ok", stats))
            else:
                self.result_queue.put(("ok", None))

        except Exception as e:
            logger.error(f"Encoder thread error: {e}")
            if container is not None:
                with contextlib.suppress(Exception):
                    container.close()
            self.result_queue.put(("error", str(e)))


class StreamingVideoEncoder:
    """Manages per-camera encoder threads for real-time video encoding during recording.

    Instead of writing frames as PNG images and then encoding to MP4 at episode end,
    this class streams frames directly to encoder threads, eliminating the
    PNG round-trip and making save_episode() near-instant.

    Uses threading instead of multiprocessing to avoid the overhead of pickling large
    numpy arrays through multiprocessing.Queue. PyAV's encode() releases the GIL,
    so encoding runs in parallel with the main recording loop.
    """

    def __init__(
        self,
        fps: int,
        camera_encoder: VideoEncoderConfig | None = None,
        queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ):
        """
        Args:
            fps: Frames per second for the output videos.
            camera_encoder: Video encoder settings applied to all cameras.
                When ``None``, :func:`camera_encoder_defaults` is used.
            encoder_threads: Number of encoder threads (global setting).
                ``None`` lets the codec decide.
            queue_maxsize: Max frames to buffer per camera before
                back-pressure drops frames.
        """
        self.fps = fps
        self._camera_encoder = camera_encoder or camera_encoder_defaults()
        self._encoder_threads = encoder_threads
        self.queue_maxsize = queue_maxsize

        self._frame_queues: dict[str, queue.Queue] = {}
        self._result_queues: dict[str, queue.Queue] = {}
        self._threads: dict[str, _CameraEncoderThread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._video_paths: dict[str, Path] = {}
        self._dropped_frames: dict[str, int] = {}
        self._episode_active = False
        self._closed = False

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        """Start encoder threads for a new episode.

        Args:
            video_keys: List of video feature keys (e.g. ["observation.images.laptop"])
            temp_dir: Base directory for temporary MP4 files
        """
        if self._episode_active:
            self.cancel_episode()

        self._dropped_frames.clear()

        for video_key in video_keys:
            frame_queue: queue.Queue = queue.Queue(maxsize=self.queue_maxsize)
            result_queue: queue.Queue = queue.Queue(maxsize=1)
            stop_event = threading.Event()

            temp_video_dir = Path(tempfile.mkdtemp(dir=temp_dir))
            video_path = temp_video_dir / f"{video_key.replace('/', '_')}_streaming.mp4"

            vcodec = self._camera_encoder.vcodec
            codec_options = self._camera_encoder.get_codec_options(self._encoder_threads, as_strings=True)
            encoder_thread = _CameraEncoderThread(
                video_path=video_path,
                fps=self.fps,
                vcodec=vcodec,
                pix_fmt=self._camera_encoder.pix_fmt,
                codec_options=codec_options,
                frame_queue=frame_queue,
                result_queue=result_queue,
                stop_event=stop_event,
            )
            encoder_thread.start()

            self._frame_queues[video_key] = frame_queue
            self._result_queues[video_key] = result_queue
            self._threads[video_key] = encoder_thread
            self._stop_events[video_key] = stop_event
            self._video_paths[video_key] = video_path

        self._episode_active = True

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """Feed a frame to the encoder for a specific camera.

        A copy of the image is made before enqueueing to prevent race conditions
        with camera drivers that may reuse buffers. If the encoder queue is full
        (encoder can't keep up), the frame is dropped with a warning instead of
        crashing the recording session.

        Args:
            video_key: The video feature key
            image: numpy array in (H,W,C) or (C,H,W) format, uint8 or float

        Raises:
            RuntimeError: If the encoder thread has crashed
        """
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")

        thread = self._threads[video_key]
        if not thread.is_alive():
            # Check for error
            try:
                status, msg = self._result_queues[video_key].get_nowait()
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} crashed: {msg}")
            except queue.Empty:
                pass
            raise RuntimeError(f"Encoder thread for {video_key} is not alive")

        try:
            self._frame_queues[video_key].put(image.copy(), timeout=0.1)
        except queue.Full:
            self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1
            count = self._dropped_frames[video_key]
            # Log periodically to avoid spam (1st, then every 10th)
            if count == 1 or count % 10 == 0:
                logger.warning(
                    f"Encoder queue full for {video_key}, dropped {count} frame(s). "
                    f"Consider using vcodec='auto' for hardware encoding or increasing encoder_queue_maxsize."
                )

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """Finish encoding the current episode.

        Sends sentinel values, waits for encoder threads to complete,
        and collects results.

        Returns:
            Dict mapping video_key to (mp4_path, stats_dict_or_None)
        """
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")

        results = {}

        # Report dropped frames
        for video_key, count in self._dropped_frames.items():
            if count > 0:
                logger.warning(f"Episode finished with {count} dropped frame(s) for {video_key}.")

        # Send sentinel to all queues
        for video_key in self._frame_queues:
            self._frame_queues[video_key].put(None)

        # Wait for all threads and collect results
        for video_key in self._threads:
            self._threads[video_key].join(timeout=120)
            if self._threads[video_key].is_alive():
                logger.error(f"Encoder thread for {video_key} did not finish in time")
                self._stop_events[video_key].set()
                self._threads[video_key].join(timeout=5)
                results[video_key] = (self._video_paths[video_key], None)
                continue

            try:
                status, data = self._result_queues[video_key].get(timeout=5)
                if status == "error":
                    raise RuntimeError(f"Encoder thread for {video_key} failed: {data}")
                results[video_key] = (self._video_paths[video_key], data)
            except queue.Empty:
                logger.error(f"No result from encoder thread for {video_key}")
                results[video_key] = (self._video_paths[video_key], None)

        self._cleanup()
        self._episode_active = False
        return results

    def cancel_episode(self) -> None:
        """Cancel the current episode, stopping encoder threads and cleaning up."""
        if not self._episode_active:
            return

        # Signal all threads to stop
        for video_key in self._stop_events:
            self._stop_events[video_key].set()

        # Wait for threads to finish
        for video_key in self._threads:
            self._threads[video_key].join(timeout=5)

            # Clean up temp MP4 files
            video_path = self._video_paths.get(video_key)
            if video_path is not None and video_path.exists():
                shutil.rmtree(str(video_path.parent), ignore_errors=True)

        self._cleanup()
        self._episode_active = False

    def close(self) -> None:
        """Close the encoder, canceling any in-progress episode."""
        if self._closed:
            return
        if self._episode_active:
            self.cancel_episode()
        self._closed = True

    def _cleanup(self) -> None:
        """Clean up queues and thread tracking dicts."""
        for q in self._frame_queues.values():
            with contextlib.suppress(Exception):
                while not q.empty():
                    q.get_nowait()
        self._frame_queues.clear()
        self._result_queues.clear()
        self._threads.clear()
        self._stop_events.clear()
        self._video_paths.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess streaming encoder
#
# Why this exists: with an nvenc vcodec, the FIRST encode() call on a
# _CameraEncoderThread lazily opens the NVENC session (CUDA context + per-stream
# encoder). PyAV holds the GIL through that open — measured at ~137 ms per camera
# even with a warm CUDA context (~300 ms cold) — and it recurs every episode, per
# camera. In-process, that GIL hold starves the recording process's 30 Hz control
# loop: the follower's Present_Position sync_read times out ("no status packet"),
# the episode aborts, and the follower jumps.
#
# Moving the whole streaming encoder into a spawned child process takes NVENC (and
# its GIL-holding session open + all encode() calls) off the parent entirely. The
# child hosts an ordinary in-process StreamingVideoEncoder (thread-per-camera); the
# parent ships raw frames to it over a small pool of shared-memory slots and a
# command queue. The parent's control loop never touches CUDA and never blocks on
# the encoder — a full shared-memory slot pool just drops frames (as the in-process
# encoder already does when its queue is full).
#
# spawn, not fork: CUDA cannot be initialized across a fork, and the parent must
# never initialize CUDA. spawn also means the child re-imports this module fresh
# WITHOUT collect.py's get_codec_options monkeypatch (which stringifies the nvenc
# `rc` option so PyAV's add_stream accepts it) — so the child re-applies that same
# fix itself in _subprocess_encoder_main before building the encoder.
# ─────────────────────────────────────────────────────────────────────────────

# Per-camera shared-memory ring depth. This must cover the burst of frames the parent produces while
# the child can't drain slots — chiefly the per-episode NVENC session open, which holds the CHILD's
# GIL for ~137 ms/camera (serialized => N_cameras × ~137 ms). The parent keeps recording through that
# window (that's the whole point — the robot no longer freezes), so those frames must buffer here or
# they drop. 24 slots ≈ 800 ms at 30 fps, comfortably covering a several-camera cold start; raise it
# via OPENARM_ENCODE_SHM_BUFFERS if you still see startup drops. Memory = slots × cameras × frame bytes.
_SHM_BUFFERS_PER_KEY = int(os.environ.get("OPENARM_ENCODE_SHM_BUFFERS", "24"))


def _patch_nvenc_codec_options() -> None:
    """Re-apply collect.py's get_codec_options string-coercion fix in this (child) process.

    lerobot's VideoEncoderConfig.get_codec_options builds the nvenc branch with an int `rc`
    even when as_strings=True; PyAV's add_stream requires all option VALUES to be strings and
    raises TypeError otherwise. Idempotent (guarded by a sentinel)."""
    if getattr(VideoEncoderConfig.get_codec_options, "_nvenc_str_safe", False):
        return
    _orig = VideoEncoderConfig.get_codec_options

    def _str_safe(self, encoder_threads=None, as_strings=False):
        opts = _orig(self, encoder_threads, as_strings)
        if as_strings:
            opts = {k: (v if isinstance(v, str) else str(v)) for k, v in opts.items()}
        return opts

    _str_safe._nvenc_str_safe = True
    VideoEncoderConfig.get_codec_options = _str_safe


def _warm_nvenc_session(camera_encoder: VideoEncoderConfig, fps: int, encoder_threads: int | None) -> None:
    """Create the CUDA context + a throwaway NVENC session once, in THIS process.

    Only meaningful in the child, where it pays the one-time (process-global) CUDA-context cost at
    startup so the first real episode's per-camera session opens are cheaper — and, being in the
    child, never touch the parent's control-loop GIL. No-op for software codecs. Best-effort."""
    if "nvenc" not in camera_encoder.vcodec:
        return
    try:
        options = camera_encoder.get_codec_options(encoder_threads, as_strings=True)
        tmp = Path(tempfile.mkdtemp(prefix="nvenc_warm_")) / "warm.mp4"
        container = av.open(str(tmp), "w")
        stream = container.add_stream(camera_encoder.vcodec, fps, options=options)
        stream.pix_fmt = camera_encoder.pix_fmt
        stream.width = stream.height = 256
        stream.time_base = Fraction(1, fps)
        dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))
        for i in range(2):
            vf = av.VideoFrame.from_image(dummy)
            vf.pts = i
            vf.time_base = Fraction(1, fps)
            pkt = stream.encode(vf)
            if pkt:
                container.mux(pkt)
        pkt = stream.encode()
        if pkt:
            container.mux(pkt)
        container.close()
        shutil.rmtree(tmp.parent, ignore_errors=True)
    except Exception as e:  # noqa: BLE001 — warm-up is an optimization; never fail the child on it
        logger.warning(f"NVENC warm-up in encoder subprocess failed (non-fatal): {e}")


def _subprocess_encoder_main(cmd_q, resp_q, free_q, enc_kwargs: dict) -> None:
    """Child-process entry point: host an in-process StreamingVideoEncoder and drive it from the
    parent's command queue. Frames arrive via named shared-memory slots (see
    SubprocessStreamingVideoEncoder). Runs in a spawned process, so it re-applies the nvenc option
    fix and re-initializes everything from scratch."""
    from multiprocessing import shared_memory

    logging.getLogger("libav").setLevel(av.logging.WARNING)
    _patch_nvenc_codec_options()

    try:
        encoder = StreamingVideoEncoder(**enc_kwargs)
        _warm_nvenc_session(encoder._camera_encoder, encoder.fps, encoder._encoder_threads)
        resp_q.put(("ready", None))
    except Exception as e:  # noqa: BLE001 — report so the parent can fall back to in-process
        resp_q.put(("ready_error", repr(e)))
        return

    shms: dict[str, list] = {}          # video_key -> [SharedMemory, ...]
    views: dict[str, list] = {}         # video_key -> [np.ndarray view into each slot]
    epoch = 0                           # current episode epoch (tags frees back to the parent)

    while True:
        op, *rest = cmd_q.get()

        if op == "alloc":
            key, names, shape, dtype = rest
            slots, slot_views = [], []
            for name in names:
                # Attach only — the PARENT created these blocks and is the sole owner that unlinks
                # them (in close()). The child just close()s its handles; attaching re-registers the
                # name in the shared resource_tracker set idempotently, and the parent's unlink()
                # balances it, so no premature unlink and no double-unregister KeyError.
                shm = shared_memory.SharedMemory(name=name)
                slots.append(shm)
                slot_views.append(np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf))
            shms[key], views[key] = slots, slot_views

        elif op == "start":
            video_keys, temp_dir, epoch = rest
            try:
                encoder.start_episode(video_keys, Path(temp_dir))
            except Exception as e:  # noqa: BLE001
                resp_q.put(("error", f"start_episode failed: {e!r}"))

        elif op == "feed":
            key, idx = rest
            frame = views[key][idx].copy()   # copy OUT of shared memory before releasing the slot
            free_q.put((epoch, key, idx))    # release the slot immediately (never blocked by encode)
            try:
                encoder.feed_frame(key, frame)
            except Exception as e:  # noqa: BLE001 — surface an encoder-thread crash to the parent
                resp_q.put(("error", f"{key}: {e}"))

        elif op == "finish":
            try:
                results = encoder.finish_episode()
                resp_q.put(("finish_ok", {k: (str(p), stats) for k, (p, stats) in results.items()}))
            except Exception as e:  # noqa: BLE001
                resp_q.put(("finish_error", repr(e)))

        elif op == "cancel":
            with contextlib.suppress(Exception):
                encoder.cancel_episode()
            resp_q.put(("cancel_ok", None))

        elif op == "close":
            with contextlib.suppress(Exception):
                encoder.close()
            for slots in shms.values():
                for shm in slots:
                    with contextlib.suppress(Exception):
                        shm.close()
            resp_q.put(("close_ok", None))
            return


class SubprocessStreamingVideoEncoder:
    """Drop-in replacement for :class:`StreamingVideoEncoder` that runs the actual encoder in a
    spawned child process, so nvenc's GIL-holding session open never hitches the parent's control
    loop. Same public API: start_episode / feed_frame / finish_episode / cancel_episode / close.

    Frame transport: a small per-camera ring of shared-memory slots (``OPENARM_ENCODE_SHM_BUFFERS``,
    default 8). feed_frame copies the frame into a free slot and sends a tiny descriptor; the child
    copies it out and frees the slot. If no slot is free (encoder behind), the frame is dropped with
    a warning — same back-pressure behavior as the in-process encoder's full queue."""

    def __init__(
        self,
        fps: int,
        camera_encoder: VideoEncoderConfig | None = None,
        queue_maxsize: int = 30,
        encoder_threads: int | None = None,
        n_buffers: int = _SHM_BUFFERS_PER_KEY,
    ):
        import multiprocessing as mp

        self.fps = fps
        self._n_buffers = max(2, n_buffers)
        self._ctx = mp.get_context("spawn")
        self._cmd_q = self._ctx.Queue()
        self._resp_q = self._ctx.Queue()
        self._free_q = self._ctx.Queue()

        enc_kwargs = {
            "fps": fps,
            "camera_encoder": camera_encoder,
            "queue_maxsize": queue_maxsize,
            "encoder_threads": encoder_threads,
        }
        self._proc = self._ctx.Process(
            target=_subprocess_encoder_main,
            args=(self._cmd_q, self._resp_q, self._free_q, enc_kwargs),
            daemon=True,
        )
        self._proc.start()

        # Per-key shared-memory pools, created lazily on the first frame of each key (sized exactly
        # to that key's frame). Reused across episodes — frame shape/dtype per key is constant.
        self._pools: dict[str, list] = {}
        self._views: dict[str, list] = {}
        self._free_slots: dict[str, list[int]] = {}
        self._slot_nbytes: dict[str, int] = {}
        self._dropped_frames: dict[str, int] = {}
        self._epoch = 0   # bumped per episode; tags frees so stale ones can't cross an episode boundary

        self._episode_active = False
        self._closed = False

        # Wait for the child to build + warm the encoder. A failure here propagates so the caller
        # (_build_streaming_encoder) can fall back to the in-process encoder. Bounded + liveness-checked:
        # spawn re-imports the parent's __main__ (heavy: torch/cv2/lerobot), and if the child dies during
        # that import it never posts a reply — so poll for the reply OR the process exiting, with a cap.
        import time as _time

        startup_timeout = float(os.environ.get("OPENARM_ENCODE_SUBPROCESS_TIMEOUT", "120"))
        deadline = _time.monotonic() + startup_timeout
        tag = payload = None
        while _time.monotonic() < deadline:
            try:
                tag, payload = self._resp_q.get(timeout=1.0)
                break
            except queue.Empty:
                if not self._proc.is_alive():
                    self.close()
                    raise RuntimeError(
                        f"encoder subprocess exited during startup (code {self._proc.exitcode})"
                    )
        if tag != "ready":
            self.close()
            raise RuntimeError(f"encoder subprocess failed to start: {payload}")
        logger.info(
            f"Streaming video encoder running in subprocess (pid={self._proc.pid}, "
            f"{self._n_buffers} shm buffers/camera) — NVENC init + encode are off the control-loop GIL."
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _drain_frees(self, discard: bool = False) -> None:
        """Reclaim slots the child has finished copying out. Frees are tagged with the episode epoch
        they belong to; a free from a prior episode (rare feeder-thread lag past the finish barrier)
        is discarded so a slot can never be handed out twice within an episode. ``discard=True`` drops
        everything (used at episode start to flush any straggler frees)."""
        while True:
            try:
                epoch, key, idx = self._free_q.get_nowait()
            except queue.Empty:
                break
            if discard or epoch != self._epoch:
                continue
            if key in self._free_slots:
                self._free_slots[key].append(idx)

    def _raise_if_child_error(self) -> None:
        """Non-blocking check for an async error posted by the child (e.g. an encoder-thread crash).
        Only ``error`` is posted asynchronously; all other replies are synchronous request/response,
        so anything on the queue mid-episode is a genuine error."""
        try:
            tag, payload = self._resp_q.get_nowait()
        except queue.Empty:
            return
        if tag == "error":
            raise RuntimeError(f"Encoder subprocess error: {payload}")

    def _alloc_pool(self, video_key: str, image: np.ndarray) -> None:
        from multiprocessing import shared_memory

        slots, views, names = [], [], []
        for _ in range(self._n_buffers):
            shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
            slots.append(shm)
            views.append(np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf))
            names.append(shm.name)
        self._pools[video_key] = slots
        self._views[video_key] = views
        self._free_slots[video_key] = list(range(self._n_buffers))
        self._slot_nbytes[video_key] = image.nbytes
        self._cmd_q.put(("alloc", video_key, names, tuple(image.shape), np.dtype(image.dtype).str))

    # ── public API (mirrors StreamingVideoEncoder) ─────────────────────────────

    def start_episode(self, video_keys: list[str], temp_dir: Path) -> None:
        if self._episode_active:
            self.cancel_episode()
        self._epoch += 1
        self._drain_frees(discard=True)   # flush any straggler frees from the previous episode
        self._dropped_frames.clear()
        # All slots are free again once the previous episode finished/cancelled (the child frees each
        # slot as it copies out, and finish/cancel are barriers). Reset bookkeeping defensively.
        for key in self._pools:
            self._free_slots[key] = list(range(self._n_buffers))
        self._cmd_q.put(("start", list(video_keys), str(temp_dir), self._epoch))
        self._episode_active = True

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")
        self._raise_if_child_error()
        self._drain_frees()

        image = np.ascontiguousarray(image)
        if video_key not in self._pools:
            self._alloc_pool(video_key, image)
        elif image.nbytes != self._slot_nbytes[video_key]:
            # Frame size changed unexpectedly for this key — can't fit the fixed slot; drop it.
            self._note_drop(video_key, reason="frame size changed")
            return

        free = self._free_slots[video_key]
        if not free:
            self._note_drop(video_key)
            return
        idx = free.pop()
        self._views[video_key][idx][...] = image
        self._cmd_q.put(("feed", video_key, idx))

    def _note_drop(self, video_key: str, reason: str = "encoder behind") -> None:
        self._dropped_frames[video_key] = self._dropped_frames.get(video_key, 0) + 1
        count = self._dropped_frames[video_key]
        if count == 1 or count % 10 == 0:
            logger.warning(
                f"Encoder subprocess {reason} for {video_key}, dropped {count} frame(s). "
                f"Increase OPENARM_ENCODE_SHM_BUFFERS or encoder_queue_maxsize if this persists."
            )

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")
        for video_key, count in self._dropped_frames.items():
            if count > 0:
                logger.warning(f"Episode finished with {count} dropped frame(s) for {video_key}.")
        self._cmd_q.put(("finish",))
        # Block until the child has flushed + joined its encoder threads. Ignore any late async
        # 'error' that raced ahead of the finish reply — the finish reply is authoritative.
        while True:
            tag, payload = self._resp_q.get()
            if tag == "finish_ok":
                self._episode_active = False
                return {k: (Path(p), stats) for k, (p, stats) in payload.items()}
            if tag == "finish_error":
                self._episode_active = False
                raise RuntimeError(f"Encoder subprocess finish failed: {payload}")
            if tag == "error":
                self._episode_active = False
                raise RuntimeError(f"Encoder subprocess error: {payload}")

    def cancel_episode(self) -> None:
        if not self._episode_active:
            return
        self._cmd_q.put(("cancel",))
        with contextlib.suppress(Exception):
            self._drain_pending_until("cancel_ok", timeout=10)
        self._episode_active = False

    def _drain_pending_until(self, expected_tag: str, timeout: float) -> None:
        import time as _time

        deadline = _time.monotonic() + timeout
        while _time.monotonic() < deadline:
            try:
                tag, _ = self._resp_q.get(timeout=max(0.0, deadline - _time.monotonic()))
            except queue.Empty:
                return
            if tag == expected_tag:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._episode_active:
            with contextlib.suppress(Exception):
                self.cancel_episode()
        with contextlib.suppress(Exception):
            self._cmd_q.put(("close",))
            self._drain_pending_until("close_ok", timeout=10)
        with contextlib.suppress(Exception):
            self._proc.join(timeout=10)
        if self._proc.is_alive():
            with contextlib.suppress(Exception):
                self._proc.terminate()
        # Parent owns the shared-memory blocks: close AND unlink them here.
        for slots in self._pools.values():
            for shm in slots:
                with contextlib.suppress(Exception):
                    shm.close()
                with contextlib.suppress(Exception):
                    shm.unlink()
        self._pools.clear()
        self._views.clear()
        self._free_slots.clear()
        for q in (self._cmd_q, self._resp_q, self._free_q):
            with contextlib.suppress(Exception):
                q.close()


@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(
    video_path: Path | str,
    camera_encoder: VideoEncoderConfig | None = None,
) -> dict:
    """Build the ``video.*`` / ``audio.*`` info dict persisted in ``info.json``.

    Args:
        video_path: Path to the encoded video file to probe.
        camera_encoder: If provided, record the exact encoder settings used to encode this
            video. Stream-derived values take precedence — encoder fields are only written for keys
            not already populated from the video file itself.
    """
    logging.getLogger("libav").setLevel(av.logging.WARNING)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    # Add additional encoder configuration if provided
    if camera_encoder is not None:
        for field_name, field_value in asdict(camera_encoder).items():
            # vcodec is already populated from the video stream
            if field_name == "vcodec":
                continue
            video_info.setdefault(f"video.{field_name}", field_value)

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_video_duration_in_s(video_path: Path | str) -> float:
    """
    Get the duration of a video file in seconds using PyAV.

    Args:
        video_path: Path to the video file.

    Returns:
        Duration of the video in seconds.
    """
    with av.open(str(video_path)) as container:
        # Get the first video stream
        video_stream = container.streams.video[0]
        # Calculate duration: stream.duration * stream.time_base gives duration in seconds
        if video_stream.duration is not None:
            duration = float(video_stream.duration * video_stream.time_base)
        else:
            # Fallback to container duration if stream duration is not available
            duration = float(container.duration / av.time_base)
    return duration


class VideoEncodingManager:
    """
    Context manager that ensures proper video encoding and data cleanup even if exceptions occur.

    This manager handles:
    - Batch encoding for any remaining episodes when recording interrupted
    - Cleaning up temporary image files from interrupted episodes
    - Removing empty image directories

    Args:
        dataset: The LeRobotDataset instance
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        writer = self.dataset.writer
        if writer is not None:
            if exc_type is not None and writer._streaming_encoder is not None:
                writer.cancel_pending_videos()

            # finalize() handles flush_pending_videos + parquet + metadata
            self.dataset.finalize()

            # Clean up episode images if recording was interrupted (only for non-streaming mode)
            if exc_type is not None and writer._streaming_encoder is None:
                writer.cleanup_interrupted_episode(self.dataset.num_episodes)
        else:
            self.dataset.finalize()

        # Clean up any remaining images directory if it's empty
        img_dir = self.dataset.root / "images"
        if img_dir.exists():
            png_files = list(img_dir.rglob("*.png"))
            if len(png_files) == 0:
                shutil.rmtree(img_dir)
                logger.debug("Cleaned up empty images directory")
            else:
                logger.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        return False  # Don't suppress the original exception
