# Inference Module

This package defines the interfaces and registries for the core Edge-side video processing pipeline, coordinating the detection-tracking-pose cascade.

## Contents
* `interfaces.py` — Declares the `BaseInferencePipeline` interface.
* `registry.py` — Manages model paths (`ModelRegistry`) and pipeline instantiations (`PipelineRegistry`).
* `exceptions.py` — Defines `InferencePipelineError`.
