"""Tests for wyoming-onnx-asr"""

import asyncio
import os
import socket
import sys
import wave
from asyncio.subprocess import PIPE
from contextlib import closing
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


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


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


@pytest.fixture(params=[None, "int8"])
async def asr_server(request):
    """Fixture to start and stop the ASR server."""
    uri = f"tcp://127.0.0.1:{find_free_port()}"

    # Set HF_HUB to local dir
    env = os.environ.copy()
    env["HF_HUB"] = str(_LOCAL_DIR)

    command = [
        sys.executable,
        "-m",
        "wyoming_onnx_asr",
        "--uri",
        uri,
        "--model-en",
        "nemo-parakeet-tdt-0.6b-v2",  # Use new dual-model flag
    ]
    quantization = request.param
    if quantization:
        command.extend(["--quantization", quantization])

    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=PIPE,
        stdout=PIPE,
        env=env,
    )
    print(f"Started ASR server with command: {' '.join(command)}")

    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        await wait_for_server(uri, timeout=_START_TIMEOUT)
        assert proc.returncode is None
        yield uri
    finally:
        if proc.returncode is None:
            proc._transport.close()


@pytest.fixture
async def asr_client(asr_server):
    """Fixture to create and configure the ASR client."""
    async with AsyncClient.from_uri(await asr_server.__anext__()) as client:
        # Check info
        await client.write_event(Describe().event())
        while True:
            event = await asyncio.wait_for(client.read_event(), timeout=_START_TIMEOUT)
            assert event is not None

            if not Info.is_type(event.type):
                continue

            info = Info.from_event(event)
            assert len(info.asr) == 1, "Expected one asr service"
            asr = info.asr[0]
            assert len(asr.models) > 0, "Expected at least one model"
            print(asr.models)
            assert any(m.name == "nemo-parakeet-tdt-0.6b-v2" for m in asr.models)
            break

        # Configure the model
        await client.write_event(Transcribe(name="nemo-parakeet-tdt-0.6b-v2").event())
        yield client
        # Cleanup happens automatically when the async with block exits


async def transcribe_wav(client, wav_path):
    """Helper function to transcribe a WAV file and return the text."""
    with wave.open(str(wav_path), "rb") as wav_file:
        await client.write_event(
            AudioStart(
                rate=wav_file.getframerate(),
                width=wav_file.getsampwidth(),
                channels=wav_file.getnchannels(),
            ).event(),
        )
        for chunk in wav_to_chunks(wav_file, _SAMPLES_PER_CHUNK):
            await client.write_event(chunk.event())
        await client.write_event(AudioStop().event())

    while True:
        event = await asyncio.wait_for(client.read_event(), timeout=_TRANSCRIBE_TIMEOUT)
        assert event is not None

        if not Transcript.is_type(event.type):
            continue

        transcript = Transcript.from_event(event)
        text = transcript.text
        return text


@pytest.mark.asyncio
async def test_living_room_lamp(asr_client):
    """Test transcription of the living room lamp command."""
    async for client in asr_client:  # Use async for to get the client
        wav_path = _DIR / "turn_on_the_living_room_lamp.wav"
        text = await transcribe_wav(client, wav_path)
        assert text == "Turn on the living room lamp."
        break  # Only process the first (and only) yielded value


@pytest.mark.asyncio
async def test_kitchen_light(asr_client):
    """Test transcription of the harvard command."""
    async for client in asr_client:  # Use async for to get the client
        wav_path = _DIR / "harvard.wav"
        text = await transcribe_wav(client, wav_path)
        assert (
            text
            == "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pasteur are my favorite. A zestful food is the hot cross bun."
        )
        break  # Only process the first (and only) yielded value
