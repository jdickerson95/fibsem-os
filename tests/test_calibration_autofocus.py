"""Tests for metric-based `auto_focus_beam` and focus metrics in calibration.py."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fibsem import calibration
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    Point,
    MicroscopeState,
)


def _synthetic_fibsem_image(h: int = 32, w: int = 32) -> FibsemImage:
    rng = np.random.default_rng(42)
    data = (rng.random((h, w)) * 255).astype(np.uint8)
    iset = ImageSettings(
        resolution=(w, h),
        dwell_time=1e-6,
        hfw=100e-6,
        beam_type=BeamType.ION,
        autocontrast=False,
    )
    metadata = FibsemImageMetadata(
        image_settings=iset,
        pixel_size=Point(x=1e-9, y=1e-9),
        microscope_state=MicroscopeState(),
    )
    return FibsemImage(data=data, metadata=metadata)


def test_laplacian_variance_sharper_higher() -> None:
    """Blurred array should have lower Laplacian variance than a sharp one."""
    sharp = _synthetic_fibsem_image(64, 64)
    from scipy import ndimage

    blurred = FibsemImage(
        data=ndimage.gaussian_filter(sharp.data.astype(np.float32), sigma=3.0).astype(
            np.uint8
        ),
        metadata=sharp.metadata,
    )
    s_sharp = calibration._laplacian_variance(sharp, normalize="minmax")
    s_blur = calibration._laplacian_variance(blurred, normalize="minmax")
    assert s_sharp > s_blur


def test_auto_focus_beam_rejects_none_working_distance() -> None:
    class M:
        def get(self, key, beam_type=None):
            if key == "working_distance":
                return None
            raise KeyError

        def set(self, key, value, beam_type=None, **_kwargs):
            pass

    img = ImageSettings(
        resolution=(8, 8),
        beam_type=BeamType.ION,
        save=False,
        autocontrast=False,
    )
    st = SimpleNamespace(image=SimpleNamespace(path="."))

    with pytest.raises(RuntimeError, match="working distance is not available"):
        calibration.auto_focus_beam(
            M(),
            st,
            BeamType.ION,
            metric_fn=calibration._laplacian_variance,
            focus_image_settings=img,
            num_steps=3,
            step_size=1e-4,
        )


def test_auto_focus_beam_sweep_picks_best_wd() -> None:
    best_wd = 12.0e-3

    class M:
        def __init__(self) -> None:
            self._wd = 10.0e-3

        def get(self, key, beam_type=None):
            if key == "working_distance":
                return self._wd
            raise KeyError

        def set(self, *args, **kwargs):
            if "key" in kwargs:
                key, value, beam_type = (
                    kwargs["key"],
                    kwargs["value"],
                    kwargs.get("beam_type"),
                )
            else:
                key, value, beam_type = args[0], args[1], args[2]
            if key == "working_distance":
                self._wd = value

    scope = M()
    rng = np.random.default_rng(0)

    def fake_new_image(microscope, focus_image_settings):
        wd = microscope.get("working_distance", focus_image_settings.beam_type)
        scale = 1.0 / (1e-9 + abs(wd - best_wd))
        data = (rng.normal(size=(24, 24)) * scale + 128).astype(np.float32)
        data = np.clip(data, 0, 255).astype(np.uint8)
        iset = ImageSettings(
            resolution=data.shape[::-1],
            beam_type=focus_image_settings.beam_type,
            autocontrast=False,
        )
        metadata = FibsemImageMetadata(
            image_settings=iset,
            pixel_size=Point(x=1e-9, y=1e-9),
            microscope_state=MicroscopeState(),
        )
        return FibsemImage(data=data, metadata=metadata)

    img = ImageSettings(
        resolution=(24, 24),
        beam_type=BeamType.ION,
        save=False,
        autocontrast=False,
    )
    st = SimpleNamespace(image=SimpleNamespace(path="."))

    with patch("fibsem.calibration.acquire.new_image", side_effect=fake_new_image):
        calibration.auto_focus_beam(
            scope,
            st,
            BeamType.ION,
            metric_fn=calibration._laplacian_variance,
            focus_image_settings=img,
            num_steps=5,
            step_size=1.0e-3,
            kwargs={"normalize": "minmax"},
            verbose=False,
        )

    assert abs(scope._wd - best_wd) < 0.2e-3
