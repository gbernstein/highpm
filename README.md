# highpm
Search for high-proper-motion stars in DECam/DELVE data.
Work by Vernon Wetzell and Gary Bernstein.

Uses the `pixmappy` module, available on github.

## Getting Started

This package processes DECam/DELVE detections to find high proper motion candidates. The pipeline consists of several stages:

1. **Position Correction** (`positionCorrection.py`) - Applies astrometric corrections including turbulence effects
2. **Detection Packing** (`detectionPacking.py`) - Concatenates and repackages detections by HEALPix pixel
3. **Proper Motion Fitting** (`PM.py`) - Links detections into moving object candidates using modest-mover and fast-mover algorithms 
4. **Verification** (`fastChecks.py`) - Independent verification of candidates
5. **Fake Injection Testing** (`fake_pipeline.py`) - For completeness/purity testing

## Quick Start

To run the full pipeline:
1. Set up configuration in `config/config.yaml` 
2. Run `positionCorrection.py` on exposure skims
3. Run `detectionPacking.py` to repackage detections
4. Run `PM.py` to find candidates
5. Run `fastChecks.py` for verification

See [[Pipeline]] and [[Configuration]] in the wiki for detailed documentation.