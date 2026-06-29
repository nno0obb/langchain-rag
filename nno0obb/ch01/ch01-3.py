import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
API_KEY = os.getenv("API_KEY")
CHATTING_BASE_URL = os.getenv("CHATTING_BASE_URL")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
MODEL = os.getenv("MODEL")


def main():
    prompt = ChatPromptTemplate.from_template(
        "초등학생이 이해할 수 있는 방식으로 답을 설명하세요.: <질문>: {query}"
    )

    llm = ChatOpenAI(
        model=MODEL,
        base_url=CHATTING_BASE_URL,
        api_key=API_KEY,
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    print(chain.invoke({"query": "피타고라스 정리의 공식은 무엇인가요?"}))


if __name__ == "__main__":
    main()
