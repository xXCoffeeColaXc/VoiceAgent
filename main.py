from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import openai, cartesia, deepgram, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from typing import AsyncIterable
from livekit.agents.voice.agent import ModelSettings
import re
from tools.arxiv_tools import search_arxiv, download_arxiv_pdf
from tools.rag_tools import ingest_document, query_document
from tools.sanitizing_tts import SanitizingTTS


class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=("You are a concise research assistant. "
                          "When the user asks to find papers, call `search_arxiv(query)` "
                          "— it will list only titles by default. "
                          "If they request more details, call `search_arxiv(query, titles_only=false)`."
                          "`download_arxiv_pdf` to fetch PDFs, "
                          "`ingest_document` to index them, "
                          "and `query_document` to answer questions."),
            tools=[search_arxiv, download_arxiv_pdf, ingest_document, query_document],
        )

    # def tts_node(self, text: AsyncIterable[str],
    #              model_settings: ModelSettings) -> AsyncIterable["rtc.AudioFrame"] | AsyncIterable[str]:
    #     """
    #     Intercept the stream of text segments from the LLM,
    #     sanitize each one, and then hand off to the default TTS node.
    #     """

    #     async def sanitize_stream():
    #         async for segment in text:
    #             # strip markdown asterisks and URLs
    #             clean = re.sub(r"\*+", "", segment)
    #             clean = re.sub(r"http[s]?://\S+", "", clean)
    #             yield clean

    #     # pass the cleaned text into the normal TTS node pipeline
    #     return Agent.default.tts_node(self, sanitize_stream(), model_settings)


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
