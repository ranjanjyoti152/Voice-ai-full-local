import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import (
    APIConnectionError,
    APITimeoutError,
    JobContext,
    WorkerOptions,
    cli,
)
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, groq
from livekit.agents import ChatContext, ChatMessage
from sentence_transformers import SentenceTransformer
import faiss
import asyncio
from functools import partial
from prompts import AGENT_INSTRUCTION

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", os.getenv("LIVEKIT_HOST", "ws://localhost:7880"))
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_AGENT_NAME = os.getenv("LIVEKIT_AGENT_NAME", "local-agent")
WHISPER_STT_URL = os.getenv("WHISPER_STT_URL", "http://whisper:80/v1")
TTS_URL = os.getenv("TTS_URL", "http://chatterbox:8004/v1")

logger = logging.getLogger("local-agent")
logger.setLevel(logging.INFO)

# load a small embedding model once
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# load all your docs
docs = []
docs_dir = os.path.join(os.path.dirname(__file__), "docs")
if os.path.exists(docs_dir):
    for fn in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, fn), encoding="utf-8") as f:
            docs.append(f.read())

# embed and build FAISS index
if docs:
    embs = embed_model.encode(docs, show_progress_bar=False)
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
else:
    dim = embed_model.get_sentence_embedding_dimension()
    index = faiss.IndexFlatL2(dim)
    logger.warning("No documents found in docs directory. RAG will return empty context.")

async def rag_lookup(query: str) -> str:
    """Perform RAG lookup for a given query"""
    loop = asyncio.get_running_loop()

    q_emb = await loop.run_in_executor(None, lambda: embed_model.encode([query]))
    D, I = await loop.run_in_executor(None, lambda: index.search(q_emb, min(3, index.ntotal)))

    ctx = ""
    if index.ntotal > 0:
        ctx = "\n\n---\n\n".join(docs[i] for i in I[0])

    print(f"RAG content: {ctx}")

    return ctx

class ResilientOpenAISTT(openai.STT):
    def __init__(self, *args, max_retries: int = 5, retry_delay: float = 2.0, **kwargs):
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        super().__init__(*args, **kwargs)

    async def recognize(self, *args, **kwargs):  # type: ignore[override]
        for attempt in range(1, self._max_retries + 1):
            try:
                return await super().recognize(*args, **kwargs)
            except (APIConnectionError, APITimeoutError) as exc:
                if attempt == self._max_retries:
                    raise
                delay = self._retry_delay * attempt
                logger.warning(
                    "STT connection failed (attempt %s/%s): %s. Retrying in %.1fs.",
                    attempt,
                    self._max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)


class LocalAgent(Agent):
    def __init__(self) -> None:
        stt = ResilientOpenAISTT(
            base_url=WHISPER_STT_URL,
            model="Systran/faster-whisper-large-v3",
            language="auto",  # Auto-detect language (Hindi/English/etc.)
            max_retries=6,
            retry_delay=2.0,
        )
        llm = openai.LLM(base_url="http://ollama:11434/v1", model="gemma3:4b", timeout=120)
        tts = openai.TTS(base_url=TTS_URL, voice="Emily.wav")
        vad_inst = silero.VAD.load()
        super().__init__(
            instructions=AGENT_INSTRUCTION,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad_inst
        )

        def llm_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_llm_metrics_collected(metrics))
        llm.on("metrics_collected", llm_metrics_wrapper)

        def stt_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_stt_metrics_collected(metrics))
        stt.on("metrics_collected", stt_metrics_wrapper)

        def eou_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_eou_metrics_collected(metrics))
        stt.on("eou_metrics_collected", eou_metrics_wrapper)

        def tts_metrics_wrapper(metrics):
            import asyncio
            asyncio.create_task(self.on_tts_metrics_collected(metrics))
        tts.on("metrics_collected", tts_metrics_wrapper)

        def vad_metrics_wrapper(event):
            import asyncio
            asyncio.create_task(self.on_vad_event(event))
        vad_inst.on("metrics_collected", vad_metrics_wrapper)

    async def on_llm_metrics_collected(self, metrics):
        logger.info(f"LLM Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'request_id': {metrics.request_id}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'duration': {metrics.duration}, " +
            f"'ttft': {getattr(metrics, 'ttft', None)}, " +
            f"'cancelled': {getattr(metrics, 'cancelled', None)}, " +
            f"'completion_tokens': {getattr(metrics, 'completion_tokens', None)}, " +
            f"'prompt_tokens': {getattr(metrics, 'prompt_tokens', None)}, " +
            f"'total_tokens': {getattr(metrics, 'total_tokens', None)}, " +
            f"'tokens_per_second': {getattr(metrics, 'tokens_per_second', None)}" +
            "}")

    async def on_stt_metrics_collected(self, metrics):
        logger.info(f"STT Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'request_id': {metrics.request_id}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'duration': {metrics.duration}, " +
            f"'speech_id': {getattr(metrics, 'speech_id', None)}, " +
            f"'error': {str(getattr(metrics, 'error', None)) if getattr(metrics, 'error', None) else None}, " +
            f"'streamed': {getattr(metrics, 'streamed', None)}, " +
            f"'audio_duration': {getattr(metrics, 'audio_duration', None)}" +
            "}")

    async def on_eou_metrics_collected(self, metrics):
        logger.info(f"EOU Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'end_of_utterance_delay': {getattr(metrics, 'end_of_utterance_delay', None)}, " +
            f"'transcription_delay': {getattr(metrics, 'transcription_delay', None)}, " +
            f"'speech_id': {getattr(metrics, 'speech_id', None)}, " +
            f"'error': {str(getattr(metrics, 'error', None)) if getattr(metrics, 'error', None) else None}" +
            "}")

    async def on_tts_metrics_collected(self, metrics):
        logger.info(f"TTS Metrics: {{" +
            f"'type': {metrics.type}, " +
            f"'label': {metrics.label}, " +
            f"'request_id': {metrics.request_id}, " +
            f"'timestamp': {metrics.timestamp if isinstance(metrics.timestamp, (int, float)) else metrics.timestamp.isoformat() if metrics.timestamp else None}, " +
            f"'ttfb': {getattr(metrics, 'ttfb', None)}, " +
            f"'duration': {metrics.duration}, " +
            f"'audio_duration': {getattr(metrics, 'audio_duration', None)}, " +
            f"'cancelled': {getattr(metrics, 'cancelled', None)}, " +
            f"'characters_count': {getattr(metrics, 'characters_count', None)}, " +
            f"'streamed': {getattr(metrics, 'streamed', None)}, " +
            f"'speech_id': {getattr(metrics, 'speech_id', None)}, " +
            f"'error': {str(getattr(metrics, 'error', None)) if getattr(metrics, 'error', None) else None}" +
            "}")

    async def on_vad_event(self, event):
        None

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage,
    ) -> None:
        rag_content = await rag_lookup(new_message.text_content)
        if rag_content:
            turn_ctx.add_message(
                role="user",
                content=f"Additional information relevant to the user's next message: {rag_content}"
            )
            logger.info(f"Added RAG content to chat context: {rag_content}")

async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession()

    await session.start(
        agent=LocalAgent(),
        room=ctx.room
    )

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            agent_name=LIVEKIT_AGENT_NAME,
            job_memory_warn_mb=1500
        )
    )