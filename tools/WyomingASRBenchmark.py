import argparse
import asyncio
import json
import time
import wave
from pathlib import Path
from typing import Dict, List, Tuple
from statistics import mean, stdev
from wyoming.audio import AudioStart, AudioStop, wav_to_chunks
from wyoming.asr import Transcribe, Transcript
from wyoming.client import AsyncClient
from wyoming.info import Describe, Info

SAMPLES_PER_CHUNK = 1024


async def get_available_models(uri: str) -> List[str]:
    async with AsyncClient.from_uri(uri) as client:
        await client.write_event(Describe().event())
        event = await client.read_event()
        if event and Info.is_type(event.type):
            info = Info.from_event(event)
            if info.asr and info.asr[0].models:
                return [model.name for model in info.asr[0].models]
    return []


async def transcribe_wav(
    uri: str, wav_path: Path, model_name: str
) -> Tuple[str, float]:
    start_time = time.time()

    async with AsyncClient.from_uri(uri) as client:
        await client.write_event(Transcribe(name=model_name).event())

        with wave.open(str(wav_path), "rb") as wav_file:
            await client.write_event(
                AudioStart(
                    rate=wav_file.getframerate(),
                    width=wav_file.getsampwidth(),
                    channels=wav_file.getnchannels(),
                ).event()
            )

            for chunk in wav_to_chunks(wav_file, SAMPLES_PER_CHUNK):
                await client.write_event(chunk.event())

            await client.write_event(AudioStop().event())

        while True:
            event = await client.read_event()
            if event is None:
                break

            if Transcript.is_type(event.type):
                transcript = Transcript.from_event(event)
                elapsed_time = time.time() - start_time
                return transcript.text, elapsed_time

    raise RuntimeError("No transcription received")


async def run_benchmark(
    uri: str, wav_path: Path, models: List[str], iterations: int
) -> Dict:
    results = {}

    for model in models:
        print(f"\nBenchmarking model: {model}")
        times = []
        transcripts = []

        for i in range(iterations):
            try:
                print(f"  Iteration {i + 1}/{iterations}...", end="", flush=True)
                text, elapsed = await transcribe_wav(uri, wav_path, model)
                times.append(elapsed)
                transcripts.append(text)
                print(f" {elapsed:.2f}s")
            except Exception as e:
                print(f"\nError with model {model}: {str(e)}")
                break

        if times:
            results[model] = {
                "mean_time": mean(times),
                "std_dev": stdev(times) if len(times) > 1 else 0,
                "min_time": min(times),
                "max_time": max(times),
                "successful_iterations": len(times),
                "transcripts": transcripts,
            }

    return results


async def main():
    parser = argparse.ArgumentParser(description="Benchmark Wyoming ASR models")
    parser.add_argument("wav_file", type=Path, help="Path to WAV file to transcribe")
    parser.add_argument(
        "--uri", default="tcp://localhost:10300", help="Wyoming ASR service URI"
    )
    parser.add_argument(
        "--iterations", type=int, default=3, help="Number of iterations per model"
    )
    parser.add_argument(
        "--models", nargs="+", help="Specific models to test (default: all available)"
    )
    parser.add_argument("--output", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    if not args.wav_file.exists():
        print(f"Error: WAV file {args.wav_file} does not exist")
        return 1

    try:
        # Get available models if not specified
        if not args.models:
            print("Discovering available models...")
            args.models = await get_available_models(args.uri)
            if not args.models:
                print("Error: No models found")
                return 1
            print(f"Found models: {', '.join(args.models)}")

        # Run benchmark
        results = await run_benchmark(
            args.uri, args.wav_file, args.models, args.iterations
        )

        # Print results
        print("\nBenchmark Results:")
        print("-" * 80)
        for model, data in results.items():
            print(f"\nModel: {model}")
            print(f"  Average Time: {data['mean_time']:.2f}s ± {data['std_dev']:.2f}s")
            print(f"  Range: {data['min_time']:.2f}s - {data['max_time']:.2f}s")
            print(
                f"  Successful Iterations: {data['successful_iterations']}/{args.iterations}"
            )
            print(f"  Sample Transcript: {data['transcripts'][0]}")

        # Save results if requested
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    asyncio.run(main())
