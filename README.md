# Wyoming Onnx ASR

[Wyoming protocol](https://github.com/rhasspy/wyoming) server for the [onnx-asr](https://github.com/istupakov/onnx-asr/) speech to text system.

## Docker Image

```shell
docker run -it -p 10300:10300 -v /path/to/local/data:/data ghcr.io/tboby/wyoming-onnx-asr
```

or for gpu

```shell
docker run -it -p 10300:10300 --gpus all -v /path/to/local/data:/data ghcr.io/tboby/wyoming-onnx-asr-gpu
```

there is also gpu TensorRT support, but it's a huge image and doesn't seem to make much performance difference

## Local Install

Install [uv](https://docs.astral.sh/uv/)

Clone the repository and use `uv`:

``` sh
git clone https://github.com/tboby/wyoming-onnx-asr.git
cd wyoming-onnx-asr
uv sync
```

Run a server anyone can connect to:

```sh
uv run --uri 'tcp://0.0.0.0:10300'
```

The `--model` can also be a HuggingFace model like `Systran/faster-distil-whisper-small.en`

**NOTE**: Models are downloaded temporarily to the `HF_HUB_CACHE` directory, which defaults to `~/.cache/huggingface/hub`.
You may need to adjust this environment variable when using a read-only root filesystem (e.g., `HF_HUB_CACHE=/tmp`).

## Running tooling
Format: `uv run -s script/format`