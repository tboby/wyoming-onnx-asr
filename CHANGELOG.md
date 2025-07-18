# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2025-07-17

### Fixed
- Remove "en" from the multilingual model

## [0.3.2] - 2025-07-17

### Fixed
- Properly present all language codes for the multilingual model

## [0.3.1] - 2025-07-15

### Changed
- Update defaults to be easier for the addon

## [0.3.0] - 2025-07-13

### Added
- Simple quantization support (for models which support it in onnx-asr)
- Multilingual model setting
- Startup logging

### Changed
- Tests to test quantization

## [0.2.1] - 2025-07-07

### Added
- arm64 support for CPU (pipeline only change)

### Changed
- Re-organised dockerfiles to cause packages to actually be cached in a separate layer

## [0.2.0] - 2025-07-06

### Added
- TensorRT support for GPU optimization - it's not very good though
- Separate Docker image for TensorRT (gpu-tensorrt)
- GitHub workflow for building and publishing TensorRT Docker image

### Changed
- Restructured Docker setup with separate Dockerfiles for GPU and TensorRT
- Renamed Dockerfile.gpu to gpu.Dockerfile and created gpu-tensorrt.Dockerfile
- Split GPU dependencies in pyproject.toml to separate regular GPU and TensorRT options

## [0.1.0] - 2025-07-04

### Changed
- Fork from wyoming-faster-whisper to create wyoming-onnx-asr
- Switch from Faster Whisper to ONNX ASR backend
- Update Python requirement to >=3.10.0
- Replace faster-whisper dependency with onnx-asr[hub]>=0.6
- Changed project name and metadata in pyproject.toml
- Update repository URL to point to new GitHub location
- Update Docker configurations to use ONNX ASR
- Simplified formatting setup by:
  - Moving all formatting configurations to pyproject.toml
  - Adding format script command

### Added
- New benchmark tool (WyomingASRBenchmark.py) for ASR performance testing
- Simple ASR client tool (asr_client.py) for testing
- GPU support with ONNX runtime
- Optional dependencies for CPU and GPU configurations
- Docker Compose configurations for both CPU and GPU setups
- Sound file handling through soundfile>=0.12.1
- Mise tasks

### Removed
- Home Assistant Add-on integration
- Transformers support
- Whisper-specific configurations and dependencies
- Standalone formatting configuration file
- Script files -> use mise instead

### Fixed
- Updated test suite for ONNX ASR compatibility
