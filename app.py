import json
from operator import itemgetter
from pathlib import Path

import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")


def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    setup_runnable()

    # Load preset questions
    filename_answers = Path("preset_answers.json")
    if filename_answers.exists():
        with open(filename_answers, "r") as f:
            preset_qa = json.load(f)["questions"]
    cl.user_session.set("preset_qa", preset_qa)

    # Store preset question embeddings
    questions = [qa["question"] for qa in preset_qa]
    question_embeddings = embedder.encode(questions, convert_to_tensor=True)
    cl.user_session.set("preset_embeddings", question_embeddings)


def get_preset_answer(question: str, threshold: float) -> str | None:
    # Get similarity of question to preset questions
    preset_qa = cl.user_session.get("preset_qa")  # type: list[dict[str, str]]
    preset_embeddings = cl.user_session.get("preset_embeddings")  # type: Tensor
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, preset_embeddings)

    # Find the best match
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[0][best_match_idx].item()

    if best_match_score > threshold:
        return preset_qa[best_match_idx]["answer"]

    return None


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    threshold = 0.8
    preset_answer = get_preset_answer(message.content, threshold=threshold)

    if preset_answer:
        res.content = preset_answer
        await res.send()

    else:
        async for chunk in runnable.astream(
            {"question": message.content},
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await res.stream_token(chunk)

        await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)
