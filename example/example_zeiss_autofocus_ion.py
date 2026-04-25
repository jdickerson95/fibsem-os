"""
Connect to a ZEISS CrossBeam and run metric-based ion-beam autofocus via
``calibration.auto_focus_beam`` (working-distance sweep + sharpness metric).

This uses the Laplacian-variance metric and a reduced scan area. Tune
``step_size`` and ``num_steps`` for your column and sample (FIB charge buildup).

Requirements:
  - Same as ``example_zeiss.py`` (Windows, SmartSEM, COM)
  - SmartSEM must allow read/write of working distance (``AP_WD`` / ``AP_FIB_WD``);
    see ``fibsem.microscopes.zeiss_api.crossbeam_client`` helpers.

By default loads ``fibsem/config/zeiss-configuration.yaml`` (Zeiss-safe 1024×768
defaults). Override with ``setup_session(..., config_path=...)`` if needed.
"""

from __future__ import annotations

import argparse
import logging
import os

from fibsem import calibration, utils
from fibsem.config import CONFIG_PATH
from fibsem.structures import BeamType, FibsemRectangle, ImageSettings

DEFAULT_ZEISS_MICROSCOPE_CONFIG = os.path.join(CONFIG_PATH, "zeiss-configuration.yaml")

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zeiss ion-beam metric autofocus example.")
    parser.add_argument(
        "--config",
        default=DEFAULT_ZEISS_MICROSCOPE_CONFIG,
        help="Microscope YAML (default: fibsem/config/zeiss-configuration.yaml).",
    )
    args = parser.parse_args()

    microscope, settings = utils.setup_session(
        manufacturer="Zeiss",
        ip_address="",
        config_path=args.config,
    )

    wd_before = microscope.get("working_distance", BeamType.ION)
    logging.info("Ion WD before autofocus: %s m", wd_before)

    focus_settings = ImageSettings(
        resolution=tuple(settings.image.resolution),
        dwell_time=settings.image.dwell_time,
        hfw=settings.image.hfw,
        beam_type=BeamType.ION,
        save=False,
        autocontrast=False,
        autogamma=False,
        path=settings.image.path,
        filename="zeiss_ion_autofocus",
        reduced_area=FibsemRectangle(0.25, 0.25, 0.5, 0.5),
    )

    calibration.auto_focus_beam(
        microscope,
        settings,
        beam_type=BeamType.ION,
        metric_fn=calibration._laplacian_variance,
        focus_image_settings=focus_settings,
        step_size=0.05e-3,
        num_steps=5,
        kwargs={"normalize": "minmax", "gaussian_sigma": 0.0},
        verbose=True,
    )

    wd_after = microscope.get("working_distance", BeamType.ION)
    logging.info("Ion WD after autofocus: %s m", wd_after)


if __name__ == "__main__":
    main()

