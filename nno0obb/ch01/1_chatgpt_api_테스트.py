import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("API_KEY")
CHATTING_BASE_URL = os.getenv("CHATTING_BASE_URL")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
MODEL = os.getenv("MODEL")


def main():
    client = OpenAI(
        base_url=os.getenv("CHATTING_BASE_URL"),
        api_key=os.getenv("API_KEY"),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "너는 상담원이야"},
            {"role": "user", "content": "서울 명소는?"},
        ],
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
