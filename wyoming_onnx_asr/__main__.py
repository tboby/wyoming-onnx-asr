#!/usr/bin/env python3
import argparse
import asyncio
import logging
from functools import partial

import onnx_asr
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
        help="Name of nemo-asr model to use (or auto)",
    )
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--device",
        default="cpu",
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
                name="nemo-asr",
                description="Nemo ASR transcription",
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

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load model
    _LOGGER.debug("Loading %s", args.model)
    if args.device == "gpu":
        import onnxruntime

        # Preload DLLs from NVIDIA site packages
        onnxruntime.preload_dlls(directory="")

    whisper_model = onnx_asr.load_model(model=args.model, providers=providers)

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
