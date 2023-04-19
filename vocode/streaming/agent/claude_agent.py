import random
import time
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAIChat
from langchain.memory import ConversationBufferMemory
from langchain.schema import ChatMessage, AIMessage
import openai
import anthropic
import os
import re
import json
from typing import Generator, Optional

from typing import Generator
import logging
from vocode import getenv

from vocode.streaming.agent.base_agent import BaseAgent
from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.utils.sse_client import SSEClient
from vocode.streaming.agent.utils import stream_llm_response


class ClaudeAgent(BaseAgent):
    def __init__(
        self,
        agent_config: ChatGPTAgentConfig,
        logger: logging.Logger = None,
        openai_api_key: Optional[str] = None,
    ):
        super().__init__(agent_config)
        openai.api_key = openai_api_key or getenv("OPENAI_API_KEY")
        #logger.debug("Initializing ChatGPTAgent")
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed in")
        self.agent_config = agent_config
        self.logger = logger or logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(agent_config.prompt_preamble),
                MessagesPlaceholder(variable_name="history"),
                HumanMessagePromptTemplate.from_template("{input}"),
            ]
        )
        self.memory = ConversationBufferMemory(return_messages=True)
        if agent_config.initial_message:
            if (
                agent_config.generate_responses
            ):  # we use ChatMessages for memory when we generate responses
                self.memory.chat_memory.messages.append(
                    ChatMessage(
                        content=agent_config.initial_message.text, role="assistant"
                    )
                )
            else:
                self.memory.chat_memory.add_ai_message(
                    agent_config.initial_message.text
                )
        self.llm = ChatOpenAI(
            model_name=self.agent_config.model_name,
            temperature=self.agent_config.temperature,
            max_tokens=self.agent_config.max_tokens,
            openai_api_key=openai.api_key,
        )
        self.conversation = ConversationChain(
            memory=self.memory, prompt=self.prompt, llm=self.llm
        )
        self.first_response = (
            self.create_first_response(agent_config.expected_first_prompt)
            if agent_config.expected_first_prompt
            else None
        )
        self.is_first_response = True

    def create_first_response(self, first_prompt):
        return self.conversation.predict(input=first_prompt)

    def respond(
        self,
        human_input,
        is_interrupt: bool = False,
        conversation_id: Optional[str] = None,
    ) -> tuple[str, bool]:
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            self.memory.chat_memory.add_user_message(human_input)
            self.memory.chat_memory.add_ai_message(cut_off_response)
            return cut_off_response, False
        self.logger.debug("LLM responding to human input")
        if self.is_first_response and self.first_response:
            self.logger.debug("First response is cached")
            self.is_first_response = False
            text = self.first_response
        else:
            text = self.conversation.predict(input=human_input)
        self.logger.debug(f"LLM response: {text}")
        return text, False

    def generate_response(
        self,
        human_input,
        is_interrupt: bool = False,
        conversation_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        # Handle reset.
        if human_input.strip().lower() in ["reset", "reset.", "reset!"]:
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[:1]
            yield "The conversation history was reset."
            return
        self.memory.chat_memory.messages.append(
            ChatMessage(role="user", content=human_input)
        )
        if is_interrupt and self.agent_config.cut_off_response:
            cut_off_response = self.get_cut_off_response()
            self.memory.chat_memory.messages.append(
                ChatMessage(role="assistant", content=cut_off_response)
            )
            yield cut_off_response
            return
        prompt_messages = [
            ChatMessage(role="system", content=self.agent_config.prompt_preamble)
        ] + self.memory.chat_memory.messages
        # Construct conversation so far.
        prompt = f"\n\nHuman: {self.agent_config.prompt_preamble}\n\nAssistant: Understood."
        for msg in self.memory.chat_memory.messages[1:]:
            name = "Human" if msg.role == "user" else "Assistant"
            prompt += f"\n\n{name}: {msg.content}"
        prompt += "\n\nAssistant:"
        print("PROMPT" + prompt)
        c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])

        response = c.completion_stream(
            prompt=prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=512,
            model="claude-v1",
            stream=True,
        )
        bot_memory_message = ChatMessage(role="assistant", content="")
        self.memory.chat_memory.messages.append(bot_memory_message)
        delta = ""

        sentence_pattern = r'(?<=[.!?])\s+'
        buffer = ""
        completion = ""
        for data in response:
            completion = data['completion']
            delta = completion[len(bot_memory_message.content):]
            bot_memory_message.content = completion
            buffer += delta

            # Check if there are any complete sentences in the current completion
            sentences = re.split(sentence_pattern, buffer)
            
            # Yield all complete sentences except for the last (possibly incomplete) one
            for sentence in sentences[:-1]:
                yield sentence.strip()

            # Keep the last (possibly incomplete) sentence for the next iteration
            buffer = sentences[-1]

        # If there is a trailing sentence without punctuation, yield it
        if buffer.strip():
            bot_memory_message.content = completion
            yield buffer.strip()
        


    def update_last_bot_message_on_cut_off(self, message: str):
        for memory_message in self.memory.chat_memory.messages[::-1]:
            if (
                isinstance(memory_message, ChatMessage)
                and memory_message.role == "assistant"
            ) or isinstance(memory_message, AIMessage):
                memory_message.content = message
                return


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    agent = ChatGPTAgent(
        ChatGPTAgentConfig(
            prompt_preamble="The assistant is having a pleasant conversation about life. If the user hasn't completed their thought, the assistant responds with 'PASS'",
        )
    )
    while True:
        response = agent.respond(input("Human: "))[0]
        print(f"AI: {response}")
        # for response in agent.generate_response(input("Human: ")):
        #     print(f"AI: {response}")
