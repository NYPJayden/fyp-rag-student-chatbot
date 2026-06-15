import os
import re
import requests
from dotenv import load_dotenv

load_dotenv()


class RAGFlowClient:
    def __init__(self):
        self.base_url = os.getenv("RAGFLOW_BASE_URL", "http://localhost").rstrip("/")
        self.chat_id = os.getenv("RAGFLOW_CHAT_ID")
        self.api_key = os.getenv("RAGFLOW_API_KEY")
        self.model = os.getenv("RAGFLOW_MODEL", "model")

        if not self.chat_id:
            raise ValueError("RAGFLOW_CHAT_ID is missing from .env")
        if not self.api_key:
            raise ValueError("RAGFLOW_API_KEY is missing from .env")

    def ask(self, question: str) -> str:
        url = f"{self.base_url}/api/v1/chats_openai/{self.chat_id}/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "stream": False,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            answer = data["choices"][0]["message"]["content"]
            return self._clean_answer(answer)

        except requests.exceptions.RequestException as error:
            return f"Sorry, I could not connect to the RAGFlow server. Error: {error}"

        except (KeyError, IndexError, TypeError) as error:
            return f"Sorry, I received an unexpected response from RAGFlow. Error: {error}"

    def _clean_answer(self, text: str) -> str:
        """
        Basic cleanup because the current model sometimes exposes reasoning text.
        This keeps the response more readable for Telegram users.
        """
        if not text:
            return "The answer you are looking for is not found in the dataset!"

        # Remove common internal reasoning phrases if they appear.
        patterns = [
            r"We need to.*?(?=\n\n|$)",
            r"Let's .*?(?=\n\n|$)",
            r"Thus final answer\.?",
            r"Thus output.*?(?=\n\n|$)",
        ]

        cleaned = text.strip()

        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()

        # Remove excessive blank lines.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned if cleaned else "The answer you are looking for is not found in the dataset!"