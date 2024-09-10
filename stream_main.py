import re
import os
import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.memory.buffer import ConversationBufferMemory


st.set_page_config(
    page_title="SiteGPT",
    page_icon="😈",
)

st.title("😈SiteGPT")
st.markdown(
    """
            #### 웹페이지 읽어 드립니다
            *진행을 위해 아래 순서를 따라주세요*
            1. 왼쪽 설정 창에 OpenAPI API키를 입력해주세요
            2. 읽어들일 웹페이지의 사이트맵(xml)을 입력해주세요
            """
)
# 세션 상태 초기화
session_defaults = {
    "messages": [],
    "api_key": None,
    "api_key_check": False,
    "url": None,
    "url_check": False,
    "url_name": None,
}

for key, default in session_defaults.items():
    st.session_state.setdefault(key, default)


API_KEY_PATTERN = r"sk-.*"
API_KEY_ERROR = "API_KEY를 입력하세요"
ENTER_URL = "URL을 입력해주세요"


# 콜백 핸들러 클래스 정의
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# 메시지 저장 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# 메시지 전송 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# 채팅 기록 표시 함수
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# API 키 저장 함수
def save_api_key():
    if re.match(API_KEY_PATTERN, st.session_state["api_key"]):
        st.session_state["api_key_check"] = True


# url 저장 함수
def save_url():
    st.session_state["url_check"] = bool(st.session_state["url"])
    st.session_state["url_name"] = (
        st.session_state["url"].split("://")[1].replace("/", "_")
        if st.session_state["url_check"]
        else None
    )


# 웹사이트 로딩 및 벡터 저장소 생성 함수
@st.cache_resource(show_spinner="웹사이트 로딩중")
def load_website(url):
    os.makedirs("./.cache/sitemap", exist_ok=True)
    cache_dir = LocalFileStore(
        f"./.cache/sitemap/embeddings/{st.session_state['url_name']}"
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"https:\/\/developers.cloudflare.com/ai-gateway.*",
            r"https:\/\/developers.cloudflare.com/vectorize.*",
            r"https:\/\/developers.cloudflare.com/workers-ai.*",
        ],
    )
    loader.requests_per_second = 50
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.session_state["api_key"],
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    history = inputs["history"]
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "history": history,
    }


# 최적의 답변 선택 함수
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    history = inputs["history"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "history": history,
        }
    )


# HTML 페이지 파싱 함수
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


# 메모리에서 대화 기록 로드 함수
def load_memory(_):
    return memory.load_memory_variables({})["history"]


# 답변 생성을 위한 프롬프트 템플릿
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question.\
    If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}

"""
)


# 최종 답변 선택을 위한 프롬프트 템플릿
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.\
            After selecting the best answers, translate the final answer into Korean.

            Cite the sources of the answers as they are in the original language, do not translate or change them.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# 답변 생성을 위한 LLM 모델 설정
llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
    openai_api_key=st.session_state["api_key"],
)

# 대화 기록을 저장하기 위한 메모리 설정
memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
    memory_key="history",
)


with st.sidebar:
    st.text_input(
        "API_KEY 입력",
        placeholder="OpenAPI API_KEY",
        on_change=save_api_key,
        key="api_key",
        type="password",
    )

    if st.session_state["api_key_check"]:
        st.success("API_KEY가 저장되었습니다.")
    else:
        st.warning(API_KEY_ERROR)

    st.divider()
    st.text_input(
        ENTER_URL,
        placeholder="https://example.com/sitemap.xml",
        key="url",
        on_change=save_url,
    )

    if st.session_state["url_check"]:
        st.success("URL이 저장되었습니다.")
    else:
        st.warning(ENTER_URL)


if not st.session_state["api_key_check"]:
    st.warning(API_KEY_ERROR)
if not st.session_state["url_check"]:
    st.warning(ENTER_URL)
else:
    if st.session_state["url_check"]:
        if ".xml" not in st.session_state["url"]:
            with st.sidebar:
                st.error(".xml 형식의 사이트맵 URL을 넣어주세요")
        else:
            retriever = load_website(st.session_state["url"])
            send_message("준비되었습니다. 질문해주세요", "ai", save=False)
            paint_history()
            message = st.chat_input("Ask a question to the website.")
            if message:
                if re.match(API_KEY_PATTERN, st.session_state["api_key"]):
                    send_message(message, "human")
                    try:
                        chain = (
                            {
                                "docs": retriever,
                                "question": RunnablePassthrough(),
                                "history": RunnableLambda(load_memory),
                            }
                            | RunnableLambda(get_answers)
                            | RunnableLambda(choose_answer)
                        )

                        def invoke_chain(question):
                            result = chain.invoke(question)
                            if hasattr(result, "content"):
                                result = result.content
                            else:
                                st.error("AI 응답에 'content' 속성이 없습니다.")
                            memory.save_context(
                                {"input": question},
                                {"output": result},
                            )
                            return result

                        with st.chat_message("ai"):
                            invoke_chain(message)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

                else:
                    message = "잘못된 API_KEY 형식입니다"
                    send_message(message, "ai")
    else:
        st.session_state["messages"] = []
