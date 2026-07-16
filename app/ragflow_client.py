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
            return (
                "Sorry, I could not connect to the diploma knowledge base right now.\n"
                f"Error: {error}"
            )

        except (KeyError, IndexError, TypeError) as error:
            return (
                "Sorry, I received an unexpected response from RAGFlow.\n"
                f"Error: {error}"
            )

    def _clean_answer(self, text: str) -> str:
        """
        Cleans RAGFlow/LLM output so it looks better in Telegram.
        The goal is to remove confusing Markdown, HTML tags, tables,
        and internal reasoning-style text.
        """
        if not text:
            return self._not_found_message()

        cleaned = text.strip()

        if "The answer you are looking for is not found in the dataset" in cleaned:
            return self._not_found_message()

        cleaned = self._remove_reasoning_text(cleaned)
        cleaned = self._remove_html_tags(cleaned)
        cleaned = self._convert_markdown_table(cleaned)
        cleaned = self._remove_markdown_symbols(cleaned)
        cleaned = self._clean_spacing(cleaned)

        if not cleaned:
            return self._not_found_message()

        return cleaned

    def _not_found_message(self) -> str:
        return (
            "I could not find this information in the current diploma knowledge base.\n\n"
            "Please check the official NYP website, admissions page, or course handbook "
            "for the most updated information."
        )

    def _remove_reasoning_text(self, text: str) -> str:
        patterns = [
            r"(?is)<think>.*?</think>",
            r"(?is)we need to.*?(?=\n\n|$)",
            r"(?is)let's .*?(?=\n\n|$)",
            r"(?is)thus final answer\.?",
            r"(?is)thus output.*?(?=\n\n|$)",
            r"(?is)reasoning:.*?(?=\n\n|$)",
        ]

        cleaned = text

        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned).strip()

        return cleaned

    def _remove_html_tags(self, text: str) -> str:
        cleaned = text

        cleaned = cleaned.replace("<br>", "\n")
        cleaned = cleaned.replace("<br/>", "\n")
        cleaned = cleaned.replace("<br />", "\n")

        # Remove any remaining HTML tags.
        cleaned = re.sub(r"<[^>]+>", "", cleaned)

        return cleaned

    def _convert_markdown_table(self, text: str) -> str:
        """
        Converts simple Markdown table rows into readable plain text.
        Example:
        | JAE ELR2B2-C | 6 to 12 points |
        becomes:
        JAE ELR2B2-C: 6 to 12 points
        """
        lines = text.splitlines()
        converted_lines = []

        for line in lines:
            stripped = line.strip()

            # Skip Markdown table separator lines such as |---|---|
            if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", stripped):
                continue

            # Convert Markdown table row
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]

                if len(cells) == 2:
                    left = self._remove_markdown_symbols(cells[0])
                    right = self._remove_markdown_symbols(cells[1])

                    if left.lower() not in ["category", "aspect", "points"]:
                        converted_lines.append(f"{left}: {right}")
                    continue

                if len(cells) > 2:
                    readable = " - ".join(
                        self._remove_markdown_symbols(cell) for cell in cells if cell.strip()
                    )
                    if readable:
                        converted_lines.append(readable)
                    continue

            converted_lines.append(line)

        return "\n".join(converted_lines)

    def _remove_markdown_symbols(self, text: str) -> str:
        cleaned = text

        # Remove bold/italic markdown symbols
        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        cleaned = cleaned.replace("*", "")

        # Convert headings into plain lines
        cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

        # Remove leftover table pipes
        cleaned = cleaned.replace("|", "")

        # Remove markdown blockquotes
        cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)

        return cleaned

    def _clean_spacing(self, text: str) -> str:
        cleaned = text

        # Remove extra spaces
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # Remove repeated blank lines
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        # Clean each line
        lines = [line.strip() for line in cleaned.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        return cleaned.strip()