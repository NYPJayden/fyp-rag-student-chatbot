import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

from app.ragflow_client import RAGFlowClient

load_dotenv()

ragflow = RAGFlowClient()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "Hello! I am the NYP Engineering Diploma Assistant.\n\n"
        "You can ask me questions such as:\n"
        "- What is AI & Data Engineering?\n"
        "- Compare AI & Data Engineering and Cloud Engineering.\n"
        "- I enjoy robotics and automation. Which diploma should I choose?"
    )
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = (
        "Ask me about NYP Engineering diploma information.\n\n"
        "Example questions:\n"
        "1. What is Cloud Engineering?\n"
        "2. What careers can I pursue after AI & Data Engineering?\n"
        "3. I like sustainability and business. Which diploma should I choose?\n\n"
        "If the answer is not in the knowledge base, I may say that it is not found in the dataset."
    )
    await update.message.reply_text(message)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text

    await update.message.reply_text("Searching the diploma knowledge base...")

    answer = ragflow.ask(user_question)

    # Telegram messages have a length limit, so split long answers.
    max_length = 3500
    if len(answer) <= max_length:
        await update.message.reply_text(answer)
    else:
        for i in range(0, len(answer), max_length):
            await update.message.reply_text(answer[i:i + max_length])


def run_bot():
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN is missing from .env")

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Telegram bot is running...")
    app.run_polling()


if __name__ == "__main__":
    run_bot()