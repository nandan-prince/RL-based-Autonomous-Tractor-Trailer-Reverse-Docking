"""MP4 recorder validation and encoding smoke tests."""

import numpy as np
import pytest

from trailer_docking.evaluation.video import MP4Recorder


def test_mp4_recorder_writes_frames_and_terminal_hold(tmp_path) -> None:
    pytest.importorskip("imageio")
    output = tmp_path / "rollout.mp4"
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    with MP4Recorder(output, fps=5.0) as recorder:
        recorder.append(frame)
        recorder.hold_last(0.2)
        assert recorder.frame_count == 2
    assert output.exists()
    assert output.stat().st_size > 0
    with pytest.raises(RuntimeError, match="closed"):
        recorder.append(frame)


def test_mp4_recorder_rejects_invalid_inputs(tmp_path) -> None:
    with pytest.raises(ValueError, match=r"\.mp4"):
        MP4Recorder(tmp_path / "rollout.gif", fps=5.0)
    with pytest.raises(ValueError, match="fps"):
        MP4Recorder(tmp_path / "rollout.mp4", fps=0.0)

    recorder = MP4Recorder(tmp_path / "valid.mp4", fps=5.0)
    with pytest.raises(ValueError, match="HxWx3"):
        recorder.append(np.zeros((16, 16), dtype=np.uint8))
    with pytest.raises(ValueError, match="duration"):
        recorder.hold_last(-1.0)
    recorder.close()
