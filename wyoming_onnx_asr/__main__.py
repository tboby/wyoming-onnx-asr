#!/usr/bin/env python3
import argparse
import asyncio
import logging
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
        "--model",
        default="nemo-parakeet-tdt-0.6b-v2",
        help="Name of onnx-asr model to use",
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

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Resolve model name
    model_name = args.model

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
                models=[
                    AsrModel(
                        name=model_name,
                        description=model_name,
                        attribution=Attribution(
                            name="NVIDIA",
                            url="https://github.com/NVIDIA/NeMo",
                        ),
                        installed=True,
                        languages=["en"],
                        version="0.1",
                    )
                ],
            )
        ],
    )

    providers = ["CPUExecutionProvider"]
    session_options = onnxruntime.SessionOptions()
    # Load model
    _LOGGER.debug("Loading %s", args.model)
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

    whisper_model = onnx_asr.load_model(
        model=args.model,
        providers=providers,
        sess_options=session_options,
        quantization=args.quantization,
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()

    # assert isinstance(whisper_model, onnx_asr.ASRModel)
    await server.run(
        partial(NemoAsrEventHandler, wyoming_info, whisper_model, model_lock)
    )


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
