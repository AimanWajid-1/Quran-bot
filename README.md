# 🌙 Quran Companion Bot

A WhatsApp-integrated AI bot that serves as your personal Quranic companion — finding relevant verses by meaning, answering Islamic questions, and sending daily prayer reminders.

---

## What It Does

- **Semantic verse search** — Tell it how you feel and it finds the most relevant verse from all 6236 Quran ayahs
- **Exact lookup** — Ask for any surah or ayah by name or number (e.g. "give me surah yaseen" or "2:255")
- **General conversation** — Greets you warmly and keeps conversation natural
- **Off-topic filtering** — Politely redirects non-Islamic questions
- **Daily reminders** — Sends a verse at all 5 prayer times automatically
- **Bilingual** — Responds in Urdu or English based on how you write

---

## Architecture

```
WhatsApp (your phone)
        ↓
Meta WhatsApp Business API
        ↓
Flask Webhook (Railway)
        ↓
    ┌───────────────────────────────┐
    │  RAG System (ChromaDB)        │
    │  6236 verses → vector search  │
    │  Finds most relevant ayah     │
    └───────────────────────────────┘
        ↓
Groq API (Llama 3.3 70B)
        ↓
WhatsApp reply sent back to you
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Webhook server | Python + Flask |
| AI model | Groq (Llama 3.3 70B) — free |
| Vector database | ChromaDB |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Quran dataset | quran-json (Sahih International translation) |
| WhatsApp integration | Meta WhatsApp Business API |
| Hosting | Railway |
| Version control | GitHub |

---

## Features In Detail

**RAG (Retrieval Augmented Generation)**
All 6236 Quran verses are embedded into a vector database. When you send a message, it is converted to a vector and the 3 most semantically similar verses are retrieved. The AI then picks the best one and explains it in context.

**Query Expansion**
Arabic/Islamic terms like "zina", "tawbah", "sabr" are automatically expanded to their English meaning before searching, so the vector search finds the right verses even for Arabic queries.

**Exact Lookup**
Detects patterns like 6:59, surah fatiha, or surah 36 and fetches the exact text directly from the JSON dataset — no AI guessing involved.

**Scheduled Reminders**
A background thread runs alongside the webhook server sending verses at Fajr, Dhuhr, Asr, Maghrib and Isha times daily.

---

## Challenges We Faced

Building this was a 3-day journey with several real challenges:

- Gemini API regional restrictions — Pakistan free tier had zero quota, forced switch to Groq
- Keyword-based search was too limited — Replaced with full RAG system for semantic understanding
- Meta webhook configuration — New Meta UI made webhook setup non-obvious
- Railway environment variables — Docker containers required explicit deploy after adding variables
- WhatsApp token expiry — Temporary tokens expire every 24 hours; solved with permanent system user token
- Hugging Face SSL blocking — Free tier blocked outbound HTTPS to Meta; migrated to Railway

---

## Setup Guide

For complete step-by-step deployment instructions including:
- Setting up Meta WhatsApp Business API
- Building the ChromaDB vector index
- Deploying to Railway
- Configuring webhooks
- Getting permanent API tokens

**See the attached PDF: Quran_Bot_Deployment_Guide.pdf**

---

## Project Structure

```
quran-bot/
├── app.py              ← Main Flask webhook + scheduler
├── quran_rag.py        ← RAG search system
├── build_index.py      ← One-time script to embed all verses
├── quran_en.json       ← Full Quran dataset (Sahih International)
├── quran_index/        ← ChromaDB vector store
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Environment Variables

```
WHATSAPP_TOKEN          ← From Meta system user (permanent)
PHONE_NUMBER_ID         ← From Meta dashboard
WHATSAPP_VERIFY_TOKEN   ← Any string you choose
GROQ_API_KEY            ← From console.groq.com
YOUR_WHATSAPP_NUMBER    ← Your number for daily reminders
```

---

*Built with the intention of making the Quran more accessible in daily life. May it be of benefit.* 🌙
