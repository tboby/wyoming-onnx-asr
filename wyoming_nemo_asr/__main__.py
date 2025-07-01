#!/usr/bin/env python3
import argparse
import asyncio
import logging
import platform
import re
from functools import partial
print("before nemo_asr")
import nemo.collections.asr as nemo_asr
print("after nemo_asr")
from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from . import __version__
from .handler import NemoAsrEventHandler

_LOGGER = logging.getLogger(__name__)


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", required=True, help="unix:// or tcp://")
    parser.add_argument(
        "--data-dir",
        required=True,
        action="append",
        help="Data directory to check for downloaded models",
    )
    parser.add_argument(
        "--download-dir",
        help="Directory to download models into (default: first data dir)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use for inference (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        help="Compute type (float16, int8, etc.)",
    )
    parser.add_argument(
        "--initial-prompt",
        help="Optional text to provide as a prompt for the first window",
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

    if not args.download_dir:
        # Download to first data dir by default
        args.download_dir = args.data_dir[0]

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format=args.log_format
    )
    _LOGGER.debug(args)

    # Automatic configuration for ARM
    machine = platform.machine().lower()

    # Resolve model name
    model_name = args.model
    match = re.match(r"^(tiny|base|small|medium)[.-]int8$", args.model)
    if match:
        # Original models re-uploaded to huggingface
        model_size = match.group(1)
        model_name = f"{model_size}-int8"
        args.model = f"rhasspy/faster-whisper-{model_name}"

    if args.language == "auto":
        # Whisper does not understand "auto"
        args.language = None

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
                        languages=nemo_asr.asr.tokenizer._LANGUAGE_CODES,  # pylint: disable=protected-access
                        version=nemo_asr.__version__,
                    )
                ],
            )
        ],
    )

    # Load model
    _LOGGER.debug("Loading %s", args.model)

    whisper_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model)

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("Ready")
    model_lock = asyncio.Lock()


    assert isinstance(whisper_model, nemo_asr.models.ASRModel)
    await server.run(
        partial(
            NemoAsrEventHandler,
            wyoming_info,
            whisper_model,
            model_lock,
            initial_prompt=args.initial_prompt,
        )
    )


# -----------------------------------------------------------------------------


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
