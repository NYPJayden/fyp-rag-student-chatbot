import os
import re
import asyncio

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.ragflow_client import RAGFlowClient

load_dotenv()

ragflow = RAGFlowClient()


DIPLOMA_ALIASES = {
    "Aerospace Engineering (C26)": [
        "aerospace engineering",
        "c26",
    ],
    "AI & Data Engineering (C31)": [
        "ai and data engineering",
        "ai data engineering",
        "artificial intelligence and data engineering",
        "c31",
    ],
    "Sustainability in Engineering with Business (C41)": [
        "sustainability in engineering with business",
        "sustainability engineering with business",
        "c41",
    ],
    "Common Engineering Programme (C42)": [
        "common engineering programme",
        "common engineering program",
        "c42",
    ],
    "Advanced & Digital Manufacturing (C62)": [
        "advanced and digital manufacturing",
        "advanced digital manufacturing",
        "c62",
    ],
    "Biomedical Engineering (C71)": [
        "biomedical engineering",
        "c71",
    ],
    "Cloud Engineering (C75)": [
        "cloud engineering",
        "c75",
    ],
    "Robotics & Mechatronics (C87)": [
        "robotics and mechatronics",
        "robotics mechatronics",
        "c87",
    ],
    "Electronic & Computer Engineering (C89)": [
        "electronic and computer engineering",
        "electronics and computer engineering",
        "c89",
    ],
}


def normalise_text(text: str) -> str:
    """Converts text into a simpler form for matching."""
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_diploma(text: str) -> str | None:
    """Returns the official diploma name if one is found."""
    normalised = normalise_text(text)

    for diploma, aliases in DIPLOMA_ALIASES.items():
        for alias in aliases:
            if normalise_text(alias) in normalised:
                return diploma

    return None


def needs_diploma_clarification(question: str) -> bool:
    """
    Checks whether the user is asking for diploma-specific information
    without providing a diploma name.
    """
    if find_diploma(question):
        return False

    normalised = normalise_text(question)

    diploma_specific_topics = [
        "cut off",
        "cutoff",
        "jae",
        "pfp",
        "admission point",
        "entry point",
        "course fee",
        "tuition fee",
        "graduation",
        "credit",
        "career",
        "job",
        "student say",
        "students say",
        "testimonial",
    ]

    return any(topic in normalised for topic in diploma_specific_topics)


def build_clarified_question(original_question: str, diploma: str) -> str:
    """
    Converts the original incomplete question and follow-up diploma name
    into one complete question for RAGFlow.
    """
    normalised = normalise_text(original_question)

    if "pfp" in normalised:
        return f"What is the PFP range for {diploma}?"

    if any(
        topic in normalised
        for topic in ["cut off", "cutoff", "jae", "admission point", "entry point"]
    ):
        return f"What are the cut-off points for {diploma}?"

    if any(topic in normalised for topic in ["course fee", "tuition fee"]):
        return f"What are the course fees payable for {diploma}?"

    if any(topic in normalised for topic in ["graduation", "credit"]):
        return f"What are the graduation requirements and required credits for {diploma}?"

    if any(
        topic in normalised
        for topic in ["student say", "students say", "testimonial"]
    ):
        return f"What do students say about {diploma}?"

    if any(topic in normalised for topic in ["career", "job"]):
        return f"What career opportunities are available after completing {diploma}?"

    return f"{original_question}\nThe specific diploma is {diploma}."


async def send_answer(update: Update, answer: str) -> None:
    """Sends long answers in smaller Telegram messages."""
    if not update.message:
        return

    max_length = 3500

    if len(answer) <= max_length:
        await update.message.reply_text(answer)
        return

    for index in range(0, len(answer), max_length):
        await update.message.reply_text(answer[index:index + max_length])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()

    message = (
        "Hello! I am the NYP Engineering Diploma Assistant.\n\n"
        "You can ask me questions such as:\n"
        "- What is AI & Data Engineering?\n"
        "- Compare AI & Data Engineering and Cloud Engineering.\n"
        "- I enjoy robotics and automation. Which diploma should I choose?"
    )

    if update.message:
        await update.message.reply_text(message)


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = (
        "Ask me about NYP Engineering diploma information.\n\n"
        "Examples:\n"
        "1. What is Cloud Engineering?\n"
        "2. What are the cut-off points for AI & Data Engineering?\n"
        "3. What careers can I pursue after Biomedical Engineering?\n\n"
        "Use /cancel to cancel a clarification request."
    )

    if update.message:
        await update.message.reply_text(message)


async def cancel(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    context.user_data.pop("pending_question", None)

    if update.message:
        await update.message.reply_text(
            "The previous clarification request has been cancelled."
        )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    if not update.message or not update.message.text:
        return

    user_question = update.message.text.strip()

    if not user_question:
        return

    pending_question = context.user_data.get("pending_question")

    # The bot previously asked the user to specify a diploma.
    if pending_question:
        diploma = find_diploma(user_question)

        if not diploma:
            await update.message.reply_text(
                "I still need a valid NYP Engineering diploma name.\n\n"
                "For example: AI & Data Engineering or Cloud Engineering.\n"
                "Use /cancel to cancel this request."
            )
            return

        context.user_data.pop("pending_question", None)

        combined_question = build_clarified_question(
            pending_question,
            diploma,
        )

        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action=ChatAction.TYPING,
        )

        answer = await asyncio.to_thread(
            ragflow.ask,
            combined_question,
        )

        await send_answer(update, answer)
        return

    # The question needs a diploma name before it can be answered.
    if needs_diploma_clarification(user_question):
        context.user_data["pending_question"] = user_question

        await update.message.reply_text(
            "Which NYP Engineering diploma are you asking about?\n\n"
            "Please reply with the diploma name, for example: "
            "AI & Data Engineering."
        )
        return

    # Normal one-message question.
    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING,
    )

    answer = await asyncio.to_thread(
        ragflow.ask,
        user_question,
    )

    await send_answer(update, answer)


def run_bot() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing from .env")

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("cancel", cancel))
    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    print("Telegram bot is running...", flush=True)
    application.run_polling()


if __name__ == "__main__":
    run_bot()