<div align="center">
  <img src="./voice-assistant-frontend/.github/assets/template-graphic.svg" alt="App Icon" width="120" />
  <h1>🧠 Local Voice Agent</h1>
  <p>A full-stack, Dockerized AI voice assistant with speech, text, and voice synthesis powered by <a href="https://livekit.io?utm_source=demo">LiveKit</a>.</p>
</div>

[Demo Video](https://github.com/user-attachments/assets/67a76e94-aacb-4087-b09c-d4e46d8e695e)

## 🧩 Overview

This repo contains everything needed to run a real-time AI voice assistant locally using:

- 🎙️ **LiveKit Agents** for STT ↔ LLM ↔ TTS
- 🧠 **Ollama** for running local LLMs (e.g., `gemma3:4b`)
- 🗣️ **Chatterbox** for TTS voice synthesis (Multilingual - 23 languages including Hindi)
- 👂 **Whisper (via VoxBox)** for speech-to-text with auto language detection
- 🌐 **Bilingual Support** - Automatically detects and responds in Hindi/English/Hinglish
- 🔍 **RAG** powered by Sentence Transformers and FAISS
- 💬 **Agent Starter React** UI for LiveKit voice sessions
- 🐳 Fully containerized via Docker Compose

## ✨ Features

- **Multilingual Support**: Speaks both Hindi 🇮🇳 and English 🇬🇧 naturally
- **Auto Language Detection**: STT automatically detects spoken language
- **Hinglish Mode**: Seamlessly mixes Hindi and English based on user preference
- **Optimized for Low VRAM**: Uses `gemma3:4b` for efficient GPU memory usage
- **High-Quality TTS**: Chatterbox multilingual TTS with 23 language support

## 🏁 Quick Start

```bash
./test.sh
```

This script:
- Cleans up existing containers
- Builds all services
- Launches the full stack (custom agent, React frontend, LLM, STT, TTS, and signaling server)

Once it finishes, open [http://localhost:3000](http://localhost:3000) to access the LiveKit React UI, or connect using any other LiveKit client.

## 📦 Architecture

Each service is containerized and communicates over a shared Docker network:
- `livekit`: WebRTC signaling server
- `agent`: Custom Python agent with LiveKit SDK and local models
- `frontend`: Next.js (agent-starter-react) UI that joins LiveKit rooms and requests tokens
- `whisper`: Speech-to-text using `vox-box` and Whisper large-v3 (auto language detection)
- `ollama`: Local LLM provider (`gemma3:4b`)
- `chatterbox`: Multilingual TTS engine (23 languages including Hindi)

## 🧠 Agent Instructions

Your agent lives in [`agent/myagent.py`](./agent/myagent.py). It uses:
- `openai.STT` → routes to Whisper with auto language detection
- `openai.LLM` → routes to Ollama (`gemma3:4b`)
- `openai.TTS` → routes to Chatterbox (multilingual)
- `silero.VAD` → for voice activity detection
- `SentenceTransformer` → embeds documents and queries for RAG
- `FAISS` → performs similarity search for knowledge retrieval

The agent supports Retrieval-Augmented Generation (RAG) by loading documents from the `agent/docs` directory. These documents are embedded using the all-MiniLM-L6-v2 model and indexed using FAISS for fast similarity search. During conversations, relevant document snippets are automatically retrieved to enhance the agent's responses.

The agent persona "Gyanika" is a bilingual learning assistant that naturally switches between Hindi and English based on student communication style.

All metrics from each component are logged for debugging.

## 🔐 Environment Variables

You can find environment examples in:
- [`/.env`](./.env)
- [`/agent/.env`](./agent/.env)
- [`/voice-assistant-frontend/.env.example`](./voice-assistant-frontend/.env.example)

These provide keys and internal URLs for each service. Most keys are placeholders for local dev use.

## 🧪 Testing & Dev

To test or redeploy:

```bash
docker compose down -v --remove-orphans
docker compose up --build
```

The services will restart and build fresh containers.

## 🧰 Project Structure

```
.
├── agent/                     # Python voice agent (bilingual)
├── ollama/                    # LLM serving (gemma3:4b)
├── whisper/                   # Whisper via vox-box (auto-detect language)
├── chatterbox/                # Multilingual TTS (23 languages)
├── livekit/                   # Signaling server
├── voice-assistant-frontend/  # LiveKit agent-starter-react (Next.js UI)
└── docker-compose.yml         # Brings it all together
```

## 🛠️ Requirements

- Docker + Docker Compose v2.24+
- NVIDIA driver + NVIDIA Container Toolkit (GPU scheduling for Whisper, Chatterbox, Ollama)
- At least one CUDA-capable GPU with 8GB+ VRAM (optimized for lower memory usage)
- Recommended system RAM: 16GB+

## ⚙️ Environment Configuration

Populate the provided samples before starting the stack:

- [`agent/.env`](./agent/.env) – LiveKit URL/key/secret plus API placeholders
- [`voice-assistant-frontend/.env.example`](./voice-assistant-frontend/.env.example) – copy to `.env.local` if you run the React frontend outside Docker
- Root `.env` (optional) – shared defaults for Compose overrides

⚠️ **Important:** the agent worker and the frontend must agree on the same agent name. Set `LIVEKIT_AGENT_NAME` for the Python service and `NEXT_PUBLIC_AGENT_NAME` (or `AGENT_NAME`) for the frontend to the same string (the Compose file defaults to `local-agent`). Without this match, LiveKit will never instruct the worker to join the room.

If you deploy the frontend somewhere other than `localhost:3000`, set `NEXT_PUBLIC_SITE_URL` (or `SITE_URL`) so Next.js can generate correct Open Graph/Twitter previews without falling back to `http://localhost:3000`.

When running through Docker Compose, the `environment` blocks already inject the same values. If you run the frontend outside Docker, ensure `.env.local` (or your shell env) contains matching keys (see sample files) so LiveKit connections succeed.

If you prefer to route speech-to-text requests through a different hostname or port (for example, when calling the Whisper service through a host-mapped port instead of the Docker network), set `WHISPER_STT_URL` before launching Compose. The agent defaults to `http://whisper:80/v1` when the variable is omitted.

## 🆘 Troubleshooting

- **Frontend can't reach LiveKit**: make sure `docker compose up -d livekit agent frontend whisper ollama chatterbox` is running, then tail `docker compose logs -f frontend livekit` to confirm the Next.js API route can mint tokens.
- **Agent never connects to the call**: verify `LIVEKIT_AGENT_NAME` (agent worker) equals `NEXT_PUBLIC_AGENT_NAME` / `AGENT_NAME` (frontend). If they differ or are blank, LiveKit will start the room without scheduling a worker.
- **Agent keeps leaving rooms**: inspect `docker compose logs agent` for stack traces (model downloads, missing GPUs, missing cloud credentials). The job will end if a worker exits, and LiveKit will log `JS_FAILED`.
- **Whisper retries downloading weights**: the large-v3 model is ~3.3 GB; keep the `whisper-data` volume mounted so it only downloads once, and ensure the container has GPU access (check with `docker compose exec whisper nvidia-smi`).
- **Hindi TTS not working**: ensure `chatterbox/config.yaml` has `repo_id: chatterbox` (not `chatterbox-turbo`) for multilingual support.

## 🙌 Credits

- Built with ❤️ by [PRO X PC](https://github.com/ranjanjyoti152)
- Uses [LiveKit Agents](https://docs.livekit.io/agents/)
- Local LLMs via [Ollama](https://ollama.com/)
- TTS via [Chatterbox](https://github.com/resemble-ai/chatterbox)
