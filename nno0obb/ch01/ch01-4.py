import os
from operator import itemgetter

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()
API_KEY = os.getenv("API_KEY")
CHATTING_BASE_URL = os.getenv("CHATTING_BASE_URL")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
MODEL = os.getenv("MODEL")


def get_chat_openai():
    return ChatOpenAI(
        model=MODEL,
        base_url=CHATTING_BASE_URL,
        api_key=API_KEY,
    )


def main():
    basictopic = (
        ChatPromptTemplate.from_template("{topic}에 대한 논쟁을 한국어로 생성합니다.")
        | get_chat_openai()
        | StrOutputParser()
        | {"base": RunnablePassthrough()}
    )

    positive = (
        ChatPromptTemplate.from_template(
            "{base}의 장점 또는 긍정적인 측면을 나열하세요."
        )
        | get_chat_openai()
        | StrOutputParser()
    )

    negative = (
        ChatPromptTemplate.from_template(
            "{base}의 단점 또는 부정적인 측면을 나열하세요."
        )
        | get_chat_openai()
        | StrOutputParser()
    )

    final = (
        ChatPromptTemplate.from_messages(
            [
                ("ai", "{original_response}"),
                ("human", "긍정:\n{results_1}\n\n부정:\n{results_2}"),
                ("system", "비평에 대한 최종 답변 생성"),
            ]
        )
        | get_chat_openai()
        | StrOutputParser()
    )

    chain = (
        basictopic
        | RunnableParallel(
            results_1=positive,
            results_2=negative,
            original_response=itemgetter("base"),
        )
        | RunnableParallel(
            results_1=itemgetter("results_1"),
            results_2=itemgetter("results_2"),
            original_response=itemgetter("original_response"),
            final_answer=final,
        )
        | RunnableMap(
            {
                "positive_result": itemgetter("results_1"),
                "negative_result": itemgetter("results_2"),
                "original_response": itemgetter("original_response"),
                "final_answer": itemgetter("final_answer"),
            }
        )
    )

    result = chain.invoke({"topic": "social media"})

    positive_result = result["positive_result"]
    negative_result = result["negative_result"]
    original_answer = result["original_response"]
    final_answer = result["final_answer"]

    print("소셜미디어에 대한 긍정 의견:\n", positive_result)
    print("\n소셜미디어에 대한 부정 의견:\n", negative_result)
    print("\n원본 의견", original_answer)
    print("\n최종 의견", final_answer)


if __name__ == "__main__":
    main()
