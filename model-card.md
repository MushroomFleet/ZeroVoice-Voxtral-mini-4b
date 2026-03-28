---
library_name: vllm
language:
- en
- fr
- es
- pt
- it
- nl
- de
- ar
- hi
license: cc-by-nc-4.0
inference: false
base_model:
- mistralai/Ministral-3-3B-Base-2512
extra_gated_description: >-
  If you want to learn more about how we process your personal data, please read
  our <a href="https://mistral.ai/terms/">Privacy Policy</a>.
tags:
- mistral-common
pipeline_tag: text-to-speech
---

# Voxtral 4B TTS 2603 

Voxtral TTS is a frontier, open-weights text-to-speech model that’s fast, instantly adaptable, and produces lifelike speech for voice agents. The model is released with BF16 weights and a set of reference voices. These voices are licensed under CC BY-NC 4, which is the license that the model inherits.

For more details, see our:
- [🔊 Demo](https://console.mistral.ai/build/audio/text-to-speech)
- [✍️ Blog post](https://mistral.ai/news/voxtral-tts)
- [🔬 Research Paper](https://mistral.ai/static/research/voxtral-tts.pdf)


## Key Features

Voxtral TTS delivers enterprise-grade text-to-speech for production voice agents, with the following capabilities:

- **Realistic, expressive speech** with natural prosody and emotional range across 9 major languages, with support for diverse dialects  
- **Text-to-Speech generation** with 20 preset voices and easy adaptation to new voices  
- **Multilingual support**: English, French, Spanish, German, Italian, Portuguese, Dutch, Arabic, and Hindi  
- **Very low latency** with fast time-to-first-audio, plus streaming and batch inference support  
- **24 kHz audio output** in WAV, PCM, FLAC, MP3, AAC, and Opus formats  
- **Production-ready performance** for high-throughput, real-time voice agent workflows

> [!Tip]
> For voice customization, visit our [AI Studio](https://console.mistral.ai/build/audio/text-to-speech).

### Use Cases

- Customer support and call center infrastructure.
- Financial services. _-- with video demo on banking KYC voice agents._
- Manufacturing and industrial operations.
- Public services and government.
- Compliance and risk.
- Supply chain and logistics.
- Automotive and in-vehicle systems.
- Sales and marketing.
- Real-time translation.

> [!Warning]
> Responsible Use - 
> You are responsible for complying with applicable laws and avoiding misuse.

## Benchmark Results

  - Measured using [vllm_omni/examples/offline_inference/voxtral_tts/end2end.py](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/voxtral_tts).
  - Input: 500-character text with a 10-second audio reference.
  - Hardware: single NVIDIA H200.
  - vllm version: v0.18.0.

*Note*: The RTF in `end2end.py` uses an inverted formula (higher = better). The table below converts it back to the standard RTF convention (lower = better)

  | Concurrency | Latency | RTF   | Throughput (char/s/GPU) |
  |:-----------:|:-------:|:-----:|:-----------------------:|
  | 1           | 70 ms   | 0.103 | 119.14                  |
  | 16          | 331 ms  | 0.237 | 879.11                  |
  | 32          | 552 ms  | 0.302 | 1430.78                 |


## Usage

The model can also be deployed with the following libraries:
- [`vllm-omni (recommended)`](https://github.com/vllm-project/vllm-omni): See [here](#vllm-omni-recommended)

### vLLM Omni (recommended)

> [!Tip]
> We've worked hand-in-hand with the vLLM-Omni team to have production-grade support for Voxtral 4B TTS 2603 with vLLM-Omni.
> Special thanks goes out to Han Gao, Hongsheng Liu, Roger Wang, and Yueqian Lin from the vLLM-Omni team.


**Installation**

Make sure to install [vllm](https://github.com/vllm-project/vllm) from the latest (>= 0.18.0) pypi package. 
See [here](https://docs.vllm.ai/en/latest/getting_started/installation/) for a full installation guide.

```
uv pip install -U vllm
```

You can also make use of a ready-to-go [docker image](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile) or on the [docker hub](https://hub.docker.com/layers/vllm/vllm-openai/v0.18.0/images/sha256-96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05).


Next, you should install [`vllm-omni`](https://github.com/vllm-project/vllm-omni) from "main".

```
uv pip install git+https://github.com/vllm-project/vllm-omni.git --upgrade
```

> [!Warning]
> If you do are seeing an error due to `git` not being installed, make sure to
> run `apt update;apt install -y git` and try again.

Installing `vllm >= 0.18.0` should automatically install `mistral_common >= 1.10.0` which you can verify by running:

```sh
python3 -c "import mistral_common; print(mistral_common.__version__)" # should print >= 1.10.0
```

#### Serve

Due to size and the BF16 format of the weights - `Voxtral-4B-TTS-2603` can run on a single GPU with >= 16GB memory.

```bash
vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
```

#### Client

```py
import io
import httpx
import soundfile as sf
 
BASE_URL = "http://<your-server-url>:8000/v1"
 
payload = {
    "input": "Paris is a beautiful city!",
    "model": "mistralai/Voxtral-4B-TTS-2603",
    "response_format": "wav",
    "voice": "casual_male",
}
 
response = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=120.0)
response.raise_for_status()
 
audio_array, sr = sf.read(io.BytesIO(response.content), dtype="float32")
print(f"Got audio: {len(audio_array)} samples at {sr} Hz")

# you can play the audio with a library like `sounddevice.play` for example
```

#### Demo

To run it:

```sh
git clone https://github.com/vllm-project/vllm-omni.git && \
cd vllm-omni && \
uv pip install gradio==5.50 && \
python examples/online_serving/voxtral_tts/gradio_demo.py \
  --host <your-server-url> \
  --port 8000
```

Alternatively you can also try it out live here ➡️ [**HF Space**](https://huggingface.co/spaces/mistralai/voxtral-tts-demo).

## License

The provided voice-references compatible with this model are licensed under [CC BY-NC 4](https://creativecommons.org/licenses/by-nc/4.0/), e.g. from EARS, CML-TTS, IndicVoices-R and Arabic Natural Audio datasets. Thus, this model inherits the same license.

*You must not use this model in a manner that infringes, misappropriates, or otherwise violates any third party’s rights, including intellectual property rights.*