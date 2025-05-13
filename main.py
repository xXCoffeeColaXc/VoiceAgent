from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import AsyncIterable
from livekit.agents.voice.agent import ModelSettings
import re
from tools.arxiv_tools import search_arxiv, download_arxiv_pdf
from tools.rag_tools import ingest_document, query_document, list_documents, describe_document
from tools.sanitizing_tts import SanitizingTTS


class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, concise AI research assistant. "
                "Speak naturally, as if you're chatting with a colleague.\n\n"
                "➤ When the user says **find papers** or similar:\n"
                "    • Call the arXiv search tool (`search_arxiv`).\n"
                "    • Return only the numbered paper titles - no authors, links, or Markdown.\n\n"
                "➤ If the user asks for **more details** about one of those papers:\n"
                "    • Call `search_arxiv` again with `titles_only=false` and read back a 1-sentence summary.\n"
                "    • Do not read the PDF URL unless the user explicitly asks for it.\n\n"
                "➤ If they ask to **download** or **ingest** a paper:\n"
                "    • Call `download_arxiv_pdf` (just fetch), or\n"
                "      `ingest_document` (fetch + vector-store) as appropriate.\n\n"
                "➤ The vector store is your long-term memory:\n"
                "    • `list_documents`  → enumerate ingested doc_ids + titles.\n"
                "    • `describe_document(doc_id)` → read that paper's metadata.\n"
                "    • `query_document(question, doc_id?)` → answer deep questions, referencing the doc_id if given.\n\n"
                "Formatting rules:\n"
                "    • Speak in plain sentences - no code blocks, no back-ticks.\n"
                "    • Never say raw URLs, file paths, or tool names unless the user asks.\n"
                "    • No asterisks, underscores, or other Markdown symbols in speech.\n\n"
                "Keep replies short, warm, and conversational."),
            tools=[
                search_arxiv, download_arxiv_pdf, ingest_document, query_document, list_documents, describe_document
            ],
        )


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        # stt=openai.STT(
        #     model="whisper-1",  # switch to Whisper
        #     language="en",  # ISO-639-1 code
        #     prompt="",  # whisper-1 supports an optional prompt
        #     use_realtime=False,  # use the HTTP REST endpoint
        #     detect_language=False,  # if True, auto-detects language
        # ),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.2, parallel_tool_calls=True),
        tts=openai.TTS(model="gpt-4o-mini-tts"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await session.generate_reply(instructions="Hello! I can help you find and read AI papers.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
