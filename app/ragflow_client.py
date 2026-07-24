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

        self.ollama_refine_enabled = os.getenv("OLLAMA_REFINE_ENABLED", "false").lower() == "true"
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.ollama_refine_model = os.getenv("OLLAMA_REFINE_MODEL", "gpt-oss:20b")

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
            cleaned_answer = self._clean_answer(answer, question)

            if self.ollama_refine_enabled:
                refined_answer = self._refine_with_ollama(question, cleaned_answer)
                return self._clean_answer(refined_answer, question)

            return cleaned_answer

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

    def _refine_with_ollama(self, question: str, answer: str) -> str:
        """
        Uses Ollama/gpt-oss to rewrite the answer into a cleaner Telegram-friendly format.
        If Ollama fails, the original cleaned answer is returned.
        """
        if not answer or "could not find this information" in answer.lower():
            return answer

        url = f"{self.ollama_base_url}/api/chat"

        system_prompt = (
            "You are refining answers for a Telegram chatbot about NYP Engineering diplomas. "
            "Rewrite the given answer so it is short, clear, and easy for students to read. "
            "Do not add new facts and do not change any numbers. "
            "Do not remove important warnings. "
            "Do not include internal reasoning or Markdown tables. "
            "Never include source IDs, chunk IDs, document IDs, filenames, Markdown filenames, "
            "underscored file names, or internal references such as ID 0 or C87_Robotics_and_Mechatronics. "
            "Use only the proper public diploma name and course code, such as "
            "Robotics & Mechatronics (C87). "
            "Use simple plain text with short bullet points if useful."
        )

        user_prompt = (
            f"User question:\n{question}\n\n"
            f"Original answer:\n{answer}\n\n"
            "Rewrite the answer for Telegram. Keep it accurate and easy to read."
        )

        payload = {
            "model": self.ollama_refine_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 350,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()

            refined = data["message"]["content"].strip()

            if not refined:
                return answer

            return refined

        except requests.exceptions.RequestException:
            return answer

        except (KeyError, TypeError):
            return answer

    def _clean_answer(
        self,
        text: str,
        question: str | None = None,
    ) -> str:
        """
        Cleans RAGFlow or LLM output so it looks better in Telegram.
        """
        if not text:
            return self._not_found_message(question)

        cleaned = text.strip()

        if "The answer you are looking for is not found in the dataset" in cleaned:
            return self._not_found_message(question)

        cleaned = self._remove_reasoning_text(cleaned)
        cleaned = self._remove_internal_source_text(cleaned)
        cleaned = self._remove_html_tags(cleaned)
        cleaned = self._convert_markdown_table(cleaned)
        cleaned = self._remove_markdown_symbols(cleaned)
        cleaned = self._clean_spacing(cleaned)

        if not cleaned:
            return self._not_found_message(question)

        return cleaned

    def _not_found_message(self, question: str | None = None) -> str:
        """
        Returns a more useful fallback based on the user's question.
        """
        normalised_question = (question or "").lower()

        fee_terms = [
            "course fee",
            "course fees",
            "tuition fee",
            "tuition fees",
            "school fee",
            "fees payable",
        ]

        graduation_terms = [
            "credit",
            "credits",
            "graduate",
            "graduation requirement",
            "graduation requirements",
        ]

        if any(term in normalised_question for term in fee_terms):
            return (
                "I could not find the exact course fee information in the current "
                "diploma knowledge base.\n\n"
                "Please check the official NYP fees or admissions page for the "
                "latest payable amount."
            )

        if any(term in normalised_question for term in graduation_terms):
            return (
                "I could not find the exact graduation credit requirements in the "
                "current diploma knowledge base.\n\n"
                "Please check the official NYP course handbook or academic "
                "regulations for the confirmed requirements."
            )

        return (
            "I could not find this information in the current diploma knowledge base.\n\n"
            "Please check the official NYP website, admissions page, or course "
            "handbook for the most updated information."
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
    
    def _remove_internal_source_text(self, text: str) -> str:
        """
        Removes internal RAGFlow references, document IDs,
        chunk IDs and Markdown filenames from the answer.
        """
        cleaned = text

        patterns = [
            # Example: ID 0 - "C87_Robotics_and_Mechatronics"
            r'(?i)\(?\s*ID\s*\d+\s*[-–—:]\s*["“”\']?'
            r'[A-Z]\d{2}_[A-Za-z0-9_]+(?:\.md)?["“”\']?\s*\)?',

            # Example: C87_Robotics_and_Mechatronics_17_Recommended_Interests
            r'(?i)\b[A-Z]\d{2}_[A-Za-z0-9_]+(?:\.md)?\b',

            # Example: source ID 3, chunk ID 5, document 2
            r'(?i)\b(?:source|document|file|chunk)\s*'
            r'(?:id)?\s*[:#-]?\s*\d+\b',

            # Remove remaining standalone ID numbers
            r'(?i)\bID\s*\d+\b',
        ]

        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned)

        # Clean brackets or punctuation left after removal.
        cleaned = re.sub(r"\(\s*\)", "", cleaned)
        cleaned = re.sub(r"\[\s*\]", "", cleaned)
        cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)

        return cleaned


    def _remove_html_tags(self, text: str) -> str:
        cleaned = text

        cleaned = cleaned.replace("<br>", "\n")
        cleaned = cleaned.replace("<br/>", "\n")
        cleaned = cleaned.replace("<br />", "\n")

        cleaned = re.sub(r"<[^>]+>", "", cleaned)

        return cleaned

    def _convert_markdown_table(self, text: str) -> str:
        lines = text.splitlines()
        converted_lines = []

        for line in lines:
            stripped = line.strip()

            if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?", stripped):
                continue

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

        cleaned = cleaned.replace("**", "")
        cleaned = cleaned.replace("__", "")
        cleaned = cleaned.replace("*", "")

        cleaned = re.sub(r"^\s*#{1,6}\s*", "", cleaned, flags=re.MULTILINE)

        cleaned = cleaned.replace("|", "")

        cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)

        return cleaned

    def _clean_spacing(self, text: str) -> str:
        cleaned = text

        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        lines = [line.strip() for line in cleaned.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        return cleaned.strip()