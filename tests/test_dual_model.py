"""Tests for dual-model scenarios in wyoming-onnx-asr"""

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
async def dual_model_server(request):
    """Fixture to start and stop the ASR server with both English and multilingual models."""
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
        "nemo-parakeet-tdt-0.6b-v2",  # English model
        "--model-multilingual",
        "onnx-community/whisper-large-v3-turbo",  # Use same model for testing (in real scenario would be different)
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
    print(f"Started dual-model ASR server with command: {' '.join(command)}")

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
async def dual_model_client(dual_model_server):
    """Fixture to create and configure the dual-model ASR client."""
    async with AsyncClient.from_uri(await dual_model_server.__anext__()) as client:
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
            assert len(asr.models) == 2, (
                "Expected two models (English and multilingual)"
            )
            print(f"Available models: {asr.models}")

            # Verify both models are available
            model_names = [m.name for m in asr.models]
            assert "nemo-parakeet-tdt-0.6b-v2" in model_names
            break

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
async def test_dual_model_english_default(dual_model_client):
    """Test that English model is used by default."""
    async for client in dual_model_client:
        # Don't specify a model - should default to English
        wav_path = _DIR / "turn_on_the_living_room_lamp.wav"
        text = await transcribe_wav(client, wav_path)
        assert text == "Turn on the living room lamp."
        break


@pytest.mark.asyncio
async def test_dual_model_explicit_english(dual_model_client):
    """Test explicitly selecting English model."""
    async for client in dual_model_client:
        # Explicitly request English model
        await client.write_event(Transcribe(language="en").event())

        wav_path = _DIR / "harvard.wav"
        text = await transcribe_wav(client, wav_path)
        assert (
            text
            == "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pasteur are my favorite. A zestful food is the hot cross bun."
        )
        break


@pytest.mark.asyncio
async def test_dual_model_server_info(dual_model_client):
    """Test that server info correctly reports both models."""
    async for client in dual_model_client:
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
            assert len(asr.models) == 2, "Expected two models"

            # Verify model details
            model_names = [m.name for m in asr.models]
            assert "nemo-parakeet-tdt-0.6b-v2" in model_names

            # Check that we have both English and multilingual models
            descriptions = [m.description for m in asr.models]
            assert any("English model" in desc for desc in descriptions)
            assert any("Multilingual model" in desc for desc in descriptions)
            break
        break


@pytest.mark.asyncio
async def test_dual_model_switching(dual_model_client):
    """Test switching between models in a single client session."""
    async for client in dual_model_client:
        # First use the English model
        await client.write_event(Transcribe(language="en").event())
        wav_path = _DIR / "turn_on_the_living_room_lamp.wav"
        text_en = await transcribe_wav(client, wav_path)
        assert text_en == "Turn on the living room lamp."

        # Then switch to the multilingual model
        await client.write_event(Transcribe(language="nl").event())
        wav_path = _DIR / "harvard.wav"
        text_multi = await transcribe_wav(client, wav_path)
        assert (
            text_multi
            == "The stale smell of old beer lingers. It takes heat to bring out the odor. A cold dip restores health and zest. A salt pickle tastes fine with ham. Tacos al pasteur are my favorite. A zestful food is the hot cross bun."
        )

        # Switch back to English
        await client.write_event(Transcribe(language="en").event())
        wav_path = _DIR / "turn_on_the_living_room_lamp.wav"
        text_en_again = await transcribe_wav(client, wav_path)
        assert text_en_again == "Turn on the living room lamp."
        break
