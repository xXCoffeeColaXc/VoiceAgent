import re
from livekit.plugins.cartesia import TTS as BaseTTS  # Cartesia TTS plugin :contentReference[oaicite:0]{index=0}


def sanitize_reply(text: str) -> str:
    # remove Markdown bold/italic markers
    text = re.sub(r"\*+", "", text)
    # remove any leftover URLs
    text = re.sub(r"http[s]?://\S+", "", text)
    return text.strip()


class SanitizingTTS(BaseTTS):

    async def synthesize(self, transcript: str, **kwargs):
        """
        Override the core synthesize method:
        1. Sanitize the LLM’s transcript
        2. Call the parent to get audio
        """
        clean = sanitize_reply(transcript)
        return await super().synthesize(clean, **kwargs)

    async def stream(self, transcript: str, **kwargs):
        """
        If your version of the plugin uses a streaming API,
        override that too so partial segments are sanitized.
        """
        clean = sanitize_reply(transcript)
        async for chunk in super().stream(clean, **kwargs):
            yield chunk
