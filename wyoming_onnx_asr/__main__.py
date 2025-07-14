#!/usr/bin/env python3
import argparse
import asyncio
import logging
import sys
from functools import partial

import onnx_asr
import onnxruntime
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import NemoAsrEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-en", help="English model name", default="nemo-parakeet-tdt-0.6b-v2"
    )
    parser.add_argument(
        "--model-multilingual", help="Multilingual model name", default="whisper-base"
    )
    parser.add_argument(
        "-q", "--quantization", help="Model quantization ('int8' for example)"
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu", "gpu-trt"],
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument("--debug", action="store_true", help="Log DEBUG messages")
    parser.add_argument(
        "--log-format", default=logging.BASIC_FORMAT, help="Format for log messages"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Print version and exit",
    )
    args = parser.parse_args()

    # Validate that at least one model flag has a non-None value
    if not any([args.model_en is not None, args.model_multilingual is not None]):
        parser.error(
            "At least one of --model-en or --model-multilingual must be specified."
        )

    # Store resolved values in local variables
    eng_model_name = args.model_en
    multi_model_name = args.model_multilingual

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Create models list based on which models will be loaded
    asr_models = []

    # Add English model if specified
    if eng_model_name is not None:
        asr_models.append(
            AsrModel(
                name=eng_model_name,
                description=f"English model: {eng_model_name}",
                attribution=Attribution(),
                installed=True,
                languages=["en"],
                version="0.1",
            )
        )

    # Add multilingual model if specified
    if multi_model_name is not None:
        asr_models.append(
            AsrModel(
                name=multi_model_name,
                description=f"Multilingual model: {multi_model_name}",
                attribution=Attribution(),
                installed=True,
                languages=["*"],
                version="0.1",
            )
        )

    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="onnx-asr",
                description="Onnx ASR transcription",
                attribution=Attribution(
                    name="Thomas Boby",
                    url="https://github.com/tboby",
                ),
                installed=True,
                version=__version__,
                models=asr_models,
            )
        ],
    )

    # Build common ORT provider list + session options once (reuse existing logic)
    providers = ["CPUExecutionProvider"]
    session_options = onnxruntime.SessionOptions()

    if args.device == "gpu" or args.device == "gpu-trt":
        # Preload DLLs from NVIDIA site packages
        onnxruntime.preload_dlls(directory="")
        # Prepend CUDA
        providers = ["CUDAExecutionProvider"] + providers
    if args.device == "gpu-trt":
        providers = ["TensorrtExecutionProvider"] + providers
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

    # Load multiple models and build container
    models = {}

    # For each non-None model name: Call onnx_asr.load_model(...) exactly as before
    if eng_model_name is not None:
        _LOGGER.info(
            "Loading English model %s, %s ...", eng_model_name, args.quantization
        )
        try:
            eng_model = onnx_asr.load_model(
                model=eng_model_name,
                providers=providers,
                sess_options=session_options,
                quantization=args.quantization,
            )
            models["en"] = eng_model
        except Exception as e:
            _LOGGER.error(
                "Failed to load English model '%s': %s", eng_model_name, str(e)
            )
            _LOGGER.error(
                "Startup validation failed - unable to load required English model"
            )
            sys.exit(1)

    if multi_model_name is not None:
        _LOGGER.info(
            "Loading multilingual model %s, %s ...", multi_model_name, args.quantization
        )
        try:
            multi_model = onnx_asr.load_model(
                model=multi_model_name,
                providers=providers,
                sess_options=session_options,
                quantization=args.quantization,
            )
            models["multi"] = multi_model
        except Exception as e:
            _LOGGER.error(
                "Failed to load multilingual model '%s': %s", multi_model_name, str(e)
            )
            _LOGGER.error(
                "Startup validation failed - unable to load required multilingual model"
            )
            sys.exit(1)

    # Validate that at least one model was successfully loaded
    if not models:
        _LOGGER.error("Startup validation failed - no models were successfully loaded")
        _LOGGER.error(
            "Fatal configuration issue: server cannot start without at least one working model"
        )
        sys.exit(1)

    try:
        server = AsyncServer.from_uri(args.uri)
    except Exception as e:
        _LOGGER.error("Failed to create server from URI '%s': %s", args.uri, str(e))
        _LOGGER.error("Startup validation failed - invalid server URI configuration")
        sys.exit(1)

    _LOGGER.info("Ready")
    # Wrap a single shared asyncio.Lock() for all models (unchanged)
    model_lock = asyncio.Lock()

    await server.run(partial(NemoAsrEventHandler, wyoming_info, models, model_lock))


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        _LOGGER.error("Fatal error during server startup: %s", str(e))
        sys.exit(1)
