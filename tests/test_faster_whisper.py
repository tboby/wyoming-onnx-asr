"""Tests for wyoming-faster-whisper"""

import asyncio
import os
import re
import sys
import wave
from asyncio.subprocess import PIPE
from pathlib import Path

import pytest
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info

_DIR = Path(__file__).parent
_PROGRAM_DIR = _DIR.parent
_LOCAL_DIR = _PROGRAM_DIR / "local"
_SAMPLES_PER_CHUNK = 1024

# Need to give time for the model to download
_START_TIMEOUT = 60
_TRANSCRIBE_TIMEOUT = 60


async def wait_for_server(uri: str, timeout: float = 10) -> None:
    """Wait for server to become available within timeout."""
    end_time = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            async with AsyncClient.from_uri(uri):
                return  # Successfully connected
        except ConnectionRefusedError:
            if asyncio.get_event_loop().time() >= end_time:
                raise  # Timeout exceeded
            await asyncio.sleep(1)

@pytest.mark.asyncio
async def test_nemo_asr() -> None:
    uri = "tcp://127.0.0.1:10300"

    # Set HF_HUB to local dir
    env = os.environ.copy()
    env["HF_HUB"] = str(_LOCAL_DIR)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "wyoming_onnx_asr",
        "--uri",
        uri,
        stdin=PIPE,
        stdout=PIPE,
        env=env
    )
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None

        await wait_for_server(uri)
        async with AsyncClient.from_uri(uri) as client:
            # Check info
            await client.write_event(Describe().event())
            while True:
                event = await asyncio.wait_for(
                    client.read_event(), timeout=_START_TIMEOUT
                )
                assert event is not None

                if not Info.is_type(event.type):
                    continue

                info = Info.from_event(event)
                assert len(info.asr) == 1, "Expected one asr service"
                asr = info.asr[0]
                assert len(asr.models) > 0, "Expected at least one model"
                assert any(
                    m.name == "nemo-parakeet-tdt-0.6b-v2" for m in asr.models
                ), "Expected nemo-parakeet-tdt-0.6b-v2 model"
                break

            # We want to use the whisper model
            await client.write_event(Transcribe(name="nemo-parakeet-tdt-0.6b-v2").event())

            # Test known WAV
            with wave.open(str(_DIR / "turn_on_the_living_room_lamp.wav"), "rb") as example_wav:
                await client.write_event(
                    AudioStart(
                        rate=example_wav.getframerate(),
                        width=example_wav.getsampwidth(),
                        channels=example_wav.getnchannels(),
                    ).event(),
                )
                for chunk in wav_to_chunks(example_wav, _SAMPLES_PER_CHUNK):
                    await client.write_event(chunk.event())

                await client.write_event(AudioStop().event())

            while True:
                event = await asyncio.wait_for(
                    client.read_event(), timeout=_TRANSCRIBE_TIMEOUT
                )
                assert event is not None

                if not Transcript.is_type(event.type):
                    continue

                transcript = Transcript.from_event(event)
                text = transcript.text.lower().strip()
                text = re.sub(r"[^a-z ]", "", text)
                assert text == "turn on the living room lamp"
                break

        proc.terminate()
        await proc.wait()
    finally:
        if proc.returncode is None:
            proc._transport.close()
