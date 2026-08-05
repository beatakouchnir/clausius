"""Vendored code, copied rather than imported.

This repository is published; the earlier projects this code came from are not.
Copying the narrow slice actually called keeps the research package
self-contained, so a reader can clone this repo and have every import resolve.

    offload, offload_model, store_backend, moe   MoE offload runtime
    suite                                        task loaders, prompts, scorers
    calibration                                  ECE / AURC / recalibration

Not part of the installable package: pyproject builds only src/clausius, so
none of this reaches the wheel. It is provenance for readers, not payload.
"""
