import asyncio
import os
from datetime import datetime, timezone
from uuid import uuid4

from dotenv import load_dotenv
from uagents import Context, Model, Protocol, Agent
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    TextContent,
    chat_protocol_spec,
)

from pipeline import HELP, run_review

### XFIX: pre-merge verifier agent

## Send this agent a GitHub pull request link (or a pasted diff) from ASI:One chat.
## It reviews the change with Meta Muse, cross-checks the findings with ASI:One,
## gets a merge verdict from SubQ Snap, and remembers every run in MongoDB.
## The pipeline lives in pipeline.py; this file is the Fetch.ai chat wrapper.
##
## Based on https://uagents.fetch.ai/docs/examples/asi-1
## Secrets are read from .env (git-ignored) instead of being hardcoded.

load_dotenv()

agent = Agent(
    name=os.environ["AGENT_NAME"],
    seed=os.environ["AGENT_SEED"],
    port=8001,
    mailbox=True,
    publish_agent_details=True,
)

class Health(Model):
    status: str


@agent.on_rest_get("/health", Health)
async def health(ctx: Context):
    # A local endpoint keeps the agent's web server (and its inspector link)
    # alive in mailbox mode; the SDK skips the server otherwise.
    return Health(status="ok")


# We create a new protocol which is compatible with the chat protocol spec. This ensures
# compatibility between agents
protocol = Protocol(spec=chat_protocol_spec)


def text_message(text, end_session=False):
    content = [TextContent(type="text", text=text)]
    if end_session:
        # signal that the session is over; the agent keeps no chat history between messages
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(timestamp=datetime.now(timezone.utc), msg_id=uuid4(), content=content)


# We define the handler for the chat messages that are sent to your agent
@protocol.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    # send the acknowledgement for receiving the message
    await ctx.send(
        sender,
        ChatAcknowledgement(timestamp=datetime.now(timezone.utc), acknowledged_msg_id=msg.msg_id),
    )

    # collect up all the text chunks
    text = ''
    for item in msg.content:
        if isinstance(item, TextContent):
            text += item.text
    ctx.logger.info(f"Review request from {sender}: {text[:200]}")

    # the pipeline runs in a worker thread; this posts each stage's update back to the chat
    loop = asyncio.get_running_loop()

    def progress(update):
        try:
            asyncio.run_coroutine_threadsafe(ctx.send(sender, text_message(update)), loop).result(timeout=30)
        except Exception:
            ctx.logger.exception('Could not send progress update')

    # the pipeline makes blocking network calls, so run it off the event loop
    try:
        response = await asyncio.to_thread(run_review, text, progress)
    except Exception as e:
        ctx.logger.exception('Review failed')
        response = f"XFIX could not finish this review: {e}\n\n{HELP}"

    # send the response back to the user
    await ctx.send(sender, text_message(response, end_session=True))


@protocol.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    # we are not interested in the acknowledgements for this example, but they can be useful to
    # implement read receipts, for example.
    pass

# attach the protocol to the agent
agent.include(protocol, publish_manifest=True)

if __name__ == "__main__":
    agent.run()
