import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
API_KEY = os.getenv("API_KEY")
CHATTING_BASE_URL = os.getenv("CHATTING_BASE_URL")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
MODEL = os.getenv("MODEL")


def main():
    llm = ChatOpenAI(
        model=MODEL,
        base_url=CHATTING_BASE_URL,
        api_key=API_KEY,
    )
    print(llm.invoke("피타고라스 정리의 공식은 무엇인가요?"))


if __name__ == "__main__":
    main()
