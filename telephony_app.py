import logging
from typing import Optional
from fastapi import FastAPI
from dotenv import load_dotenv
from vocode import getenv
from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.agent.factory import AgentFactory
from vocode.streaming.models.synthesizer import GoogleSynthesizerConfig, RimeSynthesizerConfig
from vocode.streaming.models.transcriber import AssemblyAITranscriberConfig
from vocode.streaming.transcriber.factory import TranscriberFactory
import os

load_dotenv()

from vocode.streaming.agent.claude_agent import ClaudeAgent
from vocode.streaming.models.agent import (
    AgentConfig,
    ChatGPTAgentConfig,
    EchoAgentConfig,
)
from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.telephony import TwilioConfig
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.telephony.conversation.outbound_call import OutboundCall

from vocode.streaming.telephony.server.base import InboundCallConfig, TelephonyServer

app = FastAPI(docs_url=None)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

config_manager = RedisConfigManager()



BASE_URL = os.environ['NGROK_URL']


class SpellerAgentConfig(AgentConfig, type="agent_speller"):
    pass




class ClaudeAgentFactory(AgentFactory):
    def create_agent(self, agent_config: AgentConfig) -> BaseAgent:
        return ClaudeAgent(agent_config=agent_config)


telephony_server = TelephonyServer(
    base_url=BASE_URL,
    config_manager=config_manager,
    inbound_call_configs=[
        InboundCallConfig(
            url="/inbound_call",
            agent_config=ChatGPTAgentConfig(
                initial_message=BaseMessage(text="Hello! Claude speaking."),
                prompt_preamble="You are speaking on the phone. Keep your response short, unless the user explicitly requests a long answer. Use conversational language, no formatting or bullet point lists because the user can't see them.",
                generate_responses=True,
                allowed_idle_time=120,
            ),
            synthesizer_config=GoogleSynthesizerConfig(
                sampling_rate=8000,
                audio_encoding="mulaw",
                language_code="en-GB",
                voice_name="en-GB-Neural2-B",
                speaking_rate=1.0,
            )
            # synthesizer_config=RimeSynthesizerConfig(
            #     speaker="young_male_unmarked-14",
            #     audio_encoding="mulaw",
            #     sampling_rate=8000,
            # ),
        )
    ],
    agent_factory=ClaudeAgentFactory(), # TODO fix this?
    logger=logger,
)

app.include_router(telephony_server.get_router())
