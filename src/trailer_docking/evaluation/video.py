"""Explicit MP4 writer for reproducible Gymnasium controller demonstrations."""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import numpy as np
from numpy.typing import NDArray


class MP4Recorder:
    """Stream RGB frames to H.264 without retaining a rollout in memory."""

    def __init__(self, path: str | Path, *, fps: float) -> None:
        resolved = Path(path).expanduser().resolve()
        if resolved.suffix.lower() != ".mp4":
            raise ValueError("video output path must end in .mp4")
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("video fps must be finite and positive")
        try:
            import imageio.v2 as imageio
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "MP4 recording requires: pip install -e '.[video]'"
            ) from error
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self.path = resolved
        self.fps = float(fps)
        self._writer = imageio.get_writer(
            str(resolved),
            fps=self.fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=2,
        )
        self._last_frame: NDArray[np.uint8] | None = None
        self._closed = False
        self.frame_count = 0

    def append(self, frame: NDArray[np.uint8]) -> None:
        """Append one HxWx3 uint8 RGB frame."""

        if self._closed:
            raise RuntimeError("cannot append to a closed video recorder")
        candidate = np.asarray(frame)
        if (
            candidate.dtype != np.uint8
            or candidate.ndim != 3
            or candidate.shape[2] != 3
        ):
            raise ValueError("video frame must be an HxWx3 uint8 RGB array")
        copied = candidate.copy()
        self._writer.append_data(copied)
        self._last_frame = copied
        self.frame_count += 1

    def hold_last(self, seconds: float = 1.0) -> None:
        """Repeat the terminal frame so the final status remains readable."""

        if not np.isfinite(seconds) or seconds < 0.0:
            raise ValueError("hold duration must be finite and non-negative")
        if self._last_frame is None:
            return
        for _ in range(round(seconds * self.fps)):
            self.append(self._last_frame)

    def close(self) -> None:
        if not self._closed:
            self._writer.close()
            self._closed = True

    def __enter__(self) -> MP4Recorder:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
