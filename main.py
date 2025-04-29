from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from tools.arxiv_tools import search_arxiv, download_arxiv_pdf


class Assistant(Agent):

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a concise voice assistant. "
                "When the user asks for research papers, respond with exactly the paper titles, one per line."),
            tools=[search_arxiv, download_arxiv_pdf],
        )  # ← register the tool


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.2,
            parallel_tool_calls=True,
        ),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC(),),
    )

    # First turn
    await session.generate_reply(instructions="Greet the user and offer help with AI research.")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
