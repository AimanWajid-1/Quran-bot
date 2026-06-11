import os
from dotenv import load_dotenv
from groq import Groq
from quran_rag import search_verse

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are a warm, friendly Quranic companion bot on WhatsApp. You have two modes:

MODE 1 — GENERAL CONVERSATION:
If the user says something casual like hi, hello, how are you, thank you, etc:
- Respond warmly and naturally like a friendly person would
- Keep it short and conversational
- Mention you are here to share Quranic guidance whenever they need it
- Do NOT force a verse into every casual message

MODE 2 — QURANIC GUIDANCE:
If the user shares a feeling, problem, or asks about Islam:
- Acknowledge their feeling or question in 1 short sentence
- Pick the MOST relevant verse from the ones provided to you
- Share the verse reference (Surah name and ayah number)
- Give the English meaning in simple plain words
- Explain in 2-3 sentences WHY this verse applies to their situation,
  connect it directly like a wise friend would, not a lecture
- End with one short encouraging sentence

MODE 3 — OFF TOPIC:
If no Quran verses were found because the question is unrelated to Islam or spirituality:
- Politely explain you are a Quranic companion and can only help with
  spiritual, emotional, or Islamic topics
- Be warm, not dismissive
- Invite them to ask something faith-related

General rules:
- Never make up Quranic verses — only use verses provided to you
- You are given up to 3 verses — pick the best one, do not list all of them
- Keep responses under 150 words
- Respond in the same language the user writes in (Urdu or English)
- Never be preachy or robotic
"""

CASUAL_TRIGGERS = [
    "hi", "hello", "hey", "salam", "assalam", "how are you",
    "good morning", "good night", "good evening", "thanks",
    "thank you", "shukran", "ok", "okay", "sure", "bye",
    "goodbye", "khuda hafiz", "allah hafiz", "jazakallah"
]

def is_casual_message(message: str) -> bool:
    message_lower = message.lower().strip()
    # Split into words to avoid partial matches like "sure" inside "muddaththir"
    words = message_lower.split()
    return any(trigger in words for trigger in CASUAL_TRIGGERS)

def ask_ai(user_message: str, verse_context: str | None) -> str:
    if is_casual_message(user_message):
        prompt = f"""
{SYSTEM_PROMPT}

The user sent a casual message: "{user_message}"

This is MODE 1 — respond warmly and naturally. No verse needed.
"""
    elif verse_context is None:
        prompt = f"""
{SYSTEM_PROMPT}

The user asked: "{user_message}"

No relevant Quran verses were found for this message — it appears to be off-topic.
This is MODE 3 — politely redirect them.
"""
    else:
        prompt = f"""
{SYSTEM_PROMPT}

--- Top Most Relevant Verses (pick the best one) ---
{verse_context}
---

The user said: "{user_message}"

This is MODE 2 — pick the most relevant verse and respond warmly.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


print("🌙 Quran Bot (RAG) — type anything, Ctrl+C to quit\n")

while True:
    user_input = input("You: ")
    if not user_input.strip():
        continue

    if is_casual_message(user_input):
        print("💬 [casual — no verse fetch]\n")
        verse = None
    else:
        verse = search_verse(user_input, top_k=3)
        if verse:
            print(f"\n📖 Verses found:\n{verse}\n")
        else:
            print("\n⚠️  No relevant verse found — off topic or low similarity\n")

    reply = ask_ai(user_input, verse)
    print(f"🤖 Bot: {reply}\n")
    print("-" * 50 + "\n")