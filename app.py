import os
import requests
import schedule
import time
import threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq
from quran_rag import search_verse, get_random_verse

load_dotenv()

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "myquranbot123")
WHATSAPP_TOKEN    = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID   = os.getenv("PHONE_NUMBER_ID")
YOUR_NUMBER       = os.getenv("YOUR_WHATSAPP_NUMBER")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── System prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a warm, friendly Quranic companion bot on WhatsApp. You have three modes:

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
- IMPORTANT: If none of the provided verses directly relate to the question,
  say so honestly — do not force a connection that is not there

MODE 3 — OFF TOPIC:
If no Quran verses were found because the question is unrelated to Islam or spirituality:
- Politely explain you are a Quranic companion and can only help with
  spiritual, emotional, or Islamic topics
- Be warm, not dismissive
- Invite them to ask something faith-related

MODE 4 — EXACT VERSE LOOKUP:
If the user asked for a specific surah or ayah and the full text is provided:
- Present the Arabic and English translation cleanly and beautifully
- Add a very brief 1-2 sentence reflection on the verse
- Do not add unrelated content

General rules:
- Never make up Quranic verses — only use verses provided to you
- You are given up to 3 verses — pick the best one, do not list all of them
- Keep responses under 150 words
- Respond in the same language the user writes in (Urdu or English)
- Never be preachy or robotic
"""

# ── Casual message detection (whole word match) ──────────────
CASUAL_TRIGGERS = [
    "hi", "hello", "hey", "salam", "assalam", "how are you",
    "good morning", "good night", "good evening", "thanks",
    "thank you", "shukran", "ok", "okay", "sure", "bye",
    "goodbye", "khuda hafiz", "allah hafiz", "jazakallah"
]

def is_casual_message(message: str) -> bool:
    message_lower = message.lower().strip()
    words = message_lower.split()
    return any(trigger in words for trigger in CASUAL_TRIGGERS)


# ── AI response ──────────────────────────────────────────────
def ask_ai(user_message: str, verse_context) -> str:
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

No relevant Quran verses were found — the message appears off-topic.
This is MODE 3 — politely redirect them.
"""
    else:
        prompt = f"""
{SYSTEM_PROMPT}

--- Verified Quranic Verse(s) ---
{verse_context}
---

The user said: "{user_message}"

If this is an exact verse or surah lookup (MODE 4), present it cleanly and beautifully.
If this is a topic or feeling (MODE 2), pick the most relevant verse and explain it warmly.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI error: {e}")
        return "I'm having a little trouble right now. Please try again in a moment. 🌙"


# ── Send WhatsApp message ────────────────────────────────────
def send_whatsapp_message(to: str, message: str):
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"WhatsApp send error: {response.text}")


# ── Daily prayer reminders ───────────────────────────────────
PRAYER_TIMES = {
    "Fajr":   "05:00",
    "Dhuhr":  "13:00",
    "Asr":    "16:30",
    "Maghrib": "18:45",
    "Isha":   "21:00",
}

PRAYER_MESSAGES = {
    "Fajr":    "🌅 *Fajr Mubarak!* A new day begins with Allah's blessing.",
    "Dhuhr":   "🌤️ *Dhuhr time.* Take a moment to pause and remember Allah.",
    "Asr":     "🌇 *Asr reminder.* The afternoon prayer — don't let the day pass by.",
    "Maghrib": "🌆 *Maghrib time.* The sun sets — a moment of gratitude.",
    "Isha":    "🌙 *Isha reminder.* End your day in peace with Allah's words.",
}

def send_prayer_reminder(prayer_name: str):
    """Send a prayer time reminder with a random verse."""
    if not YOUR_NUMBER:
        print("YOUR_WHATSAPP_NUMBER not set in .env — skipping reminder")
        return

    verse   = get_random_verse()
    intro   = PRAYER_MESSAGES.get(prayer_name, "🕌 Prayer time reminder.")
    message = f"{intro}\n\n📖 *Verse of the moment:*\n{verse}"

    send_whatsapp_message(YOUR_NUMBER, message)
    print(f"✅ {prayer_name} reminder sent to {YOUR_NUMBER}")


def setup_scheduler():
    """Schedule all 5 daily prayer reminders."""
    for prayer, time_str in PRAYER_TIMES.items():
        schedule.every().day.at(time_str).do(send_prayer_reminder, prayer_name=prayer)
        print(f"⏰ {prayer} reminder scheduled at {time_str}")


def run_scheduler():
    """Run scheduler in background thread."""
    setup_scheduler()
    while True:
        schedule.run_pending()
        time.sleep(30)  # Check every 30 seconds


# ── Webhook verification ─────────────────────────────────────
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode      = request.args.get("hub.mode")
    token     = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Webhook verified!")
        return challenge, 200
    return "Forbidden", 403


# ── Incoming messages ────────────────────────────────────────
@app.route("/webhook", methods=["POST"])
def receive_message():
    data = request.get_json()

    try:
        entry   = data["entry"][0]
        changes = entry["changes"][0]["value"]

        # Ignore delivery receipts and read status updates
        if "messages" not in changes:
            return jsonify({"status": "ok"}), 200

        message    = changes["messages"][0]
        user_phone = message["from"]

        # Handle non-text messages
        if message.get("type") != "text":
            send_whatsapp_message(
                user_phone,
                "I can only read text messages for now. Please type your question. 🌙"
            )
            return jsonify({"status": "ok"}), 200

        user_text = message["text"]["body"]
        print(f"📩 Message from {user_phone}: {user_text}")

        # Get verse or None
        if is_casual_message(user_text):
            verse = None
        else:
            verse = search_verse(user_text, top_k=3)

        # Get AI reply and send
        reply = ask_ai(user_text, verse)
        send_whatsapp_message(user_phone, reply)
        print(f"✅ Reply sent to {user_phone}")

    except (KeyError, IndexError) as e:
        print(f"Message parsing error: {e}")

    return jsonify({"status": "ok"}), 200


# ── Health check ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health_check():
    return "🌙 Quran Bot is running!", 200


# ── Start ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    print("⏰ Scheduler started in background")

    # Start Flask
    app.run(host="0.0.0.0", port=7860, debug=False)
