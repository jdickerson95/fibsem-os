"""
Connect to a ZEISS CrossBeam (SmartSEM COM) and acquire SEM and/or FIB images.

This script does **not** move the stage or issue polepiece‑risk moves; it only
switches beam mode (SEM / FIB) and grabs frames via the Zeiss API.

================================================================================
RUNBOOK (read before first run on hardware)
================================================================================

1. **Where to run**
   Run on the **Windows PC that hosts SmartSEM** (local COM). The Zeiss
   adapter ignores ``ip_address`` for the actual connection.

2. **Environment**
   From the repo root, with your env activated::

       pip install -e ".[ui]"    # or minimal deps if you trim UI
       python example/example_zeiss.py [options]

3. **SmartSEM**
   - SmartSEM running, system idle, sample loaded and at a **safe** working
     distance (you set this in the UI before starting).
   - Confirm the grab path used by your SmartSEM / API layout matches what
     ``ZeissMicroscope`` expects (default ``C:/api/Grab.tif`` in code). If your
     site differs, adjust ``ZeissMicroscope._api_grab_path`` or SmartSEM output.

4. **Configuration YAML**
   ``setup_session`` loads the default microscope YAML (see ``fibsem.config``
   ``DEFAULT_CONFIGURATION_PATH``), then forces ``manufacturer="Zeiss"``.
   Imaging defaults come from the ``imaging:`` block unless you override
   ``--config``. This script defaults **autocontrast off** unless you pass
   ``--autocontrast`` (avoids extra auto‑brightness passes on first try).

5. **First run (recommended)**
   - Use **electron only** first (no FIB column switch)::

         python example/example_zeiss.py --electron-only

   - Then run full SEM+FIB; you will be prompted before **ion** acquisitions
     unless you pass ``--yes-i-know`` (still no stage motion).

6. **What the script changes on the instrument**
   - ``change_beam`` / freeze as implemented in ``zeiss_api`` when acquiring.
   - Scan **resolution** and **dwell** from your ``ImageSettings`` are passed
     into ``grab_frame``; Zeiss ``acquire_image`` does **not** push YAML ``hfw``
     into SmartSEM before grab in the current adapter (FOV stays as in UI).

================================================================================
"""

from __future__ import annotations

import argparse
import logging
import sys

import matplotlib

matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt

from fibsem import acquire, utils
from fibsem.structures import BeamType

logging.basicConfig(level=logging.INFO)


def _confirm_ion() -> None:
    try:
        input(
            "Ready to switch to the ION column and acquire (no stage moves). "
            "Press Enter to continue, or Ctrl+C to abort. "
        )
    except EOFError:
        print("Non-interactive terminal: set --yes-i-know to skip this prompt.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Zeiss CrossBeam imaging smoke test.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to a microscope YAML file (passed to setup_session as config_path).",
    )
    parser.add_argument(
        "--electron-only",
        action="store_true",
        help="Only acquire with the electron beam (no FIB / no ion prompt).",
    )
    parser.add_argument(
        "--autocontrast",
        action="store_true",
        help="Enable autocontrast before each grab (default: off for a gentler first run).",
    )
    parser.add_argument(
        "--yes-i-know",
        action="store_true",
        help="Skip the confirmation prompt before ion-beam acquisitions.",
    )
    args = parser.parse_args()

    microscope, settings = utils.setup_session(
        manufacturer="Zeiss",
        ip_address="",  # ignored by Zeiss COM; kept for API compatibility
        config_path=args.config,
    )

    settings.image.autocontrast = bool(args.autocontrast)

    stage = microscope.get_stage_position()
    logging.info("Current stage position (read-only): %s", stage.pretty)

    logging.info("Image settings in use:\n%s", settings.image)

    if args.electron_only:
        settings.image.beam_type = BeamType.ELECTRON
        eb_image = acquire.new_image(microscope, settings.image)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.imshow(eb_image.data, cmap="gray")
        ax.set_title("Electron only")
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        return

    settings.image.beam_type = BeamType.ELECTRON
    eb_image = acquire.new_image(microscope, settings.image)

    if not args.yes_i_know:
        _confirm_ion()

    settings.image.beam_type = BeamType.ION
    ib_image = acquire.new_image(microscope, settings.image)

    if not args.yes_i_know:
        _confirm_ion()

    ref_eb, ref_ib = acquire.take_reference_images(microscope, settings.image)

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    ax[0, 0].imshow(eb_image.data, cmap="gray")
    ax[0, 0].set_title("Electron (new_image)")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(ib_image.data, cmap="gray")
    ax[0, 1].set_title("Ion (new_image)")
    ax[0, 1].axis("off")
    ax[1, 0].imshow(ref_eb.data, cmap="gray")
    ax[1, 0].set_title("Electron (take_reference_images)")
    ax[1, 0].axis("off")
    ax[1, 1].imshow(ref_ib.data, cmap="gray")
    ax[1, 1].set_title("Ion (take_reference_images)")
    ax[1, 1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
