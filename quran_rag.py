"""
quran_rag.py — RAG-based Quran search with:
- Exact lookup by surah number:ayah (e.g. 6:59)
- Exact lookup by surah name (e.g. surah fatiha)
- Semantic RAG search for feelings/topics
- Query expansion for Islamic/Arabic terms
- Relevance filtering to avoid off-topic verses
"""

import json
import re
import chromadb
from sentence_transformers import SentenceTransformer

# ── Load dataset and models once ────────────────────────────
print("🔄 Loading Quran dataset...")
with open("quran_en.json", encoding="utf-8") as f:
    _QURAN = json.load(f)

print("🔄 Loading RAG system...")
_model      = SentenceTransformer("all-MiniLM-L6-v2")
_client     = chromadb.PersistentClient(path="./quran_index")
_collection = _client.get_collection("quran")
print(f"✅ RAG ready — {_collection.count()} verses indexed")


# ── Surah name → number map ──────────────────────────────────
SURAH_MAP = {
    "fatiha": 1, "fatihah": 1, "fateha": 1, "fatehah": 1,
    "baqara": 2, "baqarah": 2, "bakara": 2,
    "imran": 3, "ali imran": 3,
    "nisa": 4, "nisaa": 4, "nissa": 4,
    "maida": 5, "maidah": 5,
    "anam": 6, "inam": 6, "an'am": 6, "anam": 6,
    "araf": 7, "a'raf": 7,
    "anfal": 8,
    "tawba": 9, "taubah": 9, "tauba": 9,
    "yunus": 10, "younus": 10,
    "hud": 11,
    "yusuf": 12, "yousuf": 12,
    "rad": 13, "raad": 13,
    "ibrahim": 14,
    "hijr": 15,
    "nahl": 16,
    "isra": 17, "israa": 17, "bani israel": 17,
    "kahf": 18,
    "maryam": 19,
    "taha": 20,
    "anbiya": 21,
    "hajj": 22,
    "muminun": 23, "mominoon": 23,
    "nur": 24, "noor": 24,
    "furqan": 25,
    "shuara": 26,
    "naml": 27,
    "qasas": 28,
    "ankabut": 29,
    "rum": 30,
    "luqman": 31,
    "sajda": 32,
    "ahzab": 33,
    "saba": 34,
    "fatir": 35,
    "yasin": 36, "yaseen": 36,
    "saffat": 37,
    "sad": 38,
    "zumar": 39,
    "ghafir": 40,
    "fussilat": 41,
    "shura": 42,
    "zukhruf": 43,
    "dukhan": 44,
    "jathiya": 45,
    "ahqaf": 46,
    "muhammad": 47,
    "fath": 48,
    "hujurat": 49,
    "qaf": 50,
    "dhariyat": 51,
    "tur": 52,
    "najm": 53,
    "qamar": 54,
    "rahman": 55,
    "waqia": 56, "waqiah": 56,
    "hadid": 57,
    "mujadila": 58,
    "hashr": 59,
    "mumtahina": 60,
    "saff": 61,
    "jumuah": 62, "juma": 62,
    "munafiqun": 63,
    "taghabun": 64,
    "talaq": 65,
    "tahrim": 66,
    "mulk": 67,
    "qalam": 68,
    "haqqa": 69,
    "maarij": 70,
    "nuh": 71,
    "jinn": 72,
    "muzzammil": 73,
    "muddathir": 74,
    "qiyama": 75,
    "insan": 76, "dahr": 76,
    "mursalat": 77,
    "naba": 78,
    "naziat": 79,
    "abasa": 80,
    "takwir": 81,
    "infitar": 82,
    "mutaffifin": 83,
    "inshiqaq": 84,
    "buruj": 85,
    "tariq": 86,
    "ala": 87,
    "ghashiya": 88,
    "fajr": 89,
    "balad": 90,
    "shams": 91,
    "layl": 92,
    "duha": 93, "duha": 93,
    "sharh": 94, "inshirah": 94,
    "tin": 95,
    "alaq": 96,
    "qadr": 97,
    "bayyina": 98,
    "zalzala": 99,
    "adiyat": 100,
    "qaria": 101,
    "takathur": 102,
    "asr": 103,
    "humaza": 104,
    "fil": 105,
    "quraysh": 106,
    "maun": 107,
    "kawthar": 108,
    "kafirun": 109,
    "nasr": 110,
    "masad": 111, "lahab": 111,
    "ikhlas": 112,
    "falaq": 113,
    "nas": 114,
}

# ── Islamic term expander ────────────────────────────────────
TERM_EXPANSION = {
    "zina":      "fornication adultery unlawful sexual intercourse prohibition",
    "riba":      "interest usury prohibition loans money",
    "nikah":     "marriage wedding spouse husband wife",
    "tawbah":    "repentance forgiveness returning to Allah",
    "tauba":     "repentance forgiveness returning to Allah",
    "hijab":     "modesty covering women dress code",
    "jihad":     "struggle striving in the way of Allah",
    "jannah":    "paradise heaven reward believers",
    "jahannam":  "hellfire punishment disbelievers",
    "salah":     "prayer worship five times daily",
    "namaz":     "prayer worship five times daily",
    "sawm":      "fasting Ramadan abstaining",
    "roza":      "fasting Ramadan abstaining",
    "zakah":     "charity obligatory giving poor",
    "hajj":      "pilgrimage Mecca obligation",
    "shirk":     "associating partners with Allah polytheism",
    "kufr":      "disbelief rejection of faith",
    "gunnah":    "sin wrongdoing repentance forgiveness",
    "gunah":     "sin wrongdoing repentance forgiveness",
    "dua":       "supplication calling upon Allah prayer",
    "rizq":      "provision sustenance Allah provides",
    "sabr":      "patience perseverance hardship",
    "shukr":     "gratitude thankfulness blessings Allah",
    "tawakkul":  "reliance trust in Allah",
    "nifaq":     "hypocrisy two-faced believers",
    "hasad":     "envy jealousy heart disease",
    "kibr":      "arrogance pride ego",
    "taqwa":     "God consciousness fear of Allah piety",
    "ikhlas":    "sincerity devotion intention Allah",
    "akhirah":   "afterlife hereafter death resurrection",
    "duniya":    "worldly life materialism attachment",
}

NON_ISLAMIC_KEYWORDS = [
    "2+2", "calculate", "equation", "algebra", "calculus",
    "multiply", "divide", "subtract", "integral", "derivative",
    "python", "javascript", "code", "program", "function",
    "bug", "debug", "html", "css", "database", "sql",
    "weather", "temperature", "recipe", "football", "cricket",
    "movie", "song", "music", "game", "stock", "crypto", "bitcoin",
]

ISLAMIC_KEYWORDS = [
    "allah", "islam", "quran", "muslim", "prayer", "faith", "dua",
    "feel", "feeling", "sad", "happy", "worried", "anxious", "lost",
    "stressed", "grateful", "hopeless", "angry", "scared", "lonely",
    "guidance", "help", "sin", "forgive", "forgiveness", "death",
    "life", "purpose", "patience", "trust", "hope", "peace", "heart",
    "soul", "repent", "mercy", "love", "family", "marriage", "halal",
    "haram", "heaven", "hell", "prophet", "sunnah", "ramadan",
    "surah", "ayah", "verse", "translation", "meaning",
] + list(TERM_EXPANSION.keys())


# ════════════════════════════════════════════════════════════
# EXACT LOOKUP FUNCTIONS
# ════════════════════════════════════════════════════════════

def detect_exact_request(message: str):
    """
    Detects if user is asking for a specific surah/ayah by reference.
    Returns ("reference", surah_no, ayah_no) or
            ("surah", surah_no, None) or
            None if not a direct request
    """
    message_lower = message.lower()

    # Pattern 1 — numeric reference like 6:59 or 2:255
    ref_match = re.search(r'\b(\d{1,3})\s*:\s*(\d{1,3})\b', message)
    if ref_match:
        s, a = int(ref_match.group(1)), int(ref_match.group(2))
        if 1 <= s <= 114:
            return ("reference", s, a)

    # Pattern 2 — surah name mentioned
    for name, number in SURAH_MAP.items():
        if name in message_lower:
            ayah_match = re.search(
                r'(?:ayah|ayat|verse|aya|aayat)\s*(\d+)', message_lower
            )
            ayah = int(ayah_match.group(1)) if ayah_match else None
            return ("surah", number, ayah)

    # Pattern 3 — "surah 36" or "chapter 2"
    surah_num_match = re.search(
        r'(?:surah|sura|chapter)\s+(\d+)', message_lower
    )
    if surah_num_match:
        s = int(surah_num_match.group(1))
        if 1 <= s <= 114:
            ayah_match = re.search(
                r'(?:ayah|ayat|verse|aya)\s*(\d+)', message_lower
            )
            ayah = int(ayah_match.group(1)) if ayah_match else None
            return ("surah", s, ayah)

    return None


def lookup_exact(surah_number: int, ayah_number: int = None) -> str:
    """
    Fetch a specific ayah or full surah directly from the JSON dataset.
    """
    surah      = _QURAN[surah_number - 1]
    surah_name = surah["transliteration"]
    surah_trans = surah["translation"]
    verses     = surah["verses"]
    total      = len(verses)

    if ayah_number:
        if ayah_number > total:
            return f"Surah {surah_name} only has {total} verses."
        v = verses[ayah_number - 1]
        return (
            f"📖 Surah {surah_name} — {surah_trans} ({surah_number}:{ayah_number})\n\n"
            f"Arabic: {v.get('text', '')}\n\n"
            f"English: {v.get('translation', '')}"
        )
    else:
        # Full surah — show all, paginate if long
        limit  = min(total, 20)
        output = [f"📖 Surah {surah_name} — {surah_trans} ({total} verses)\n"]
        for i, v in enumerate(verses[:limit], 1):
            output.append(f"{surah_number}:{i}  {v.get('translation', '')}")
        if total > 20:
            output.append(
                f"\n...{total - 20} more verses. "
                f"Ask for a specific ayah e.g. '{surah_number}:21'"
            )
        return "\n".join(output)


# ════════════════════════════════════════════════════════════
# SEMANTIC RAG SEARCH
# ════════════════════════════════════════════════════════════

def expand_query(message: str) -> str:
    message_lower = message.lower()
    expansions = [
        exp for term, exp in TERM_EXPANSION.items()
        if term in message_lower
    ]
    return message + " " + " ".join(expansions) if expansions else message


def is_islamic_query(message: str) -> bool:
    message_lower = message.lower()
    if any(kw in message_lower for kw in NON_ISLAMIC_KEYWORDS):
        return False
    return any(kw in message_lower for kw in ISLAMIC_KEYWORDS)


# ════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════

def search_verse(user_message: str, top_k: int = 3) -> str | None:
    """
    Main function called by the bot.
    1. Check for exact surah/ayah request → return exact text
    2. Check for Islamic topic → semantic RAG search
    3. Off-topic → return None
    """

    # ── Step 1: Exact lookup ─────────────────────────────────
    exact = detect_exact_request(user_message)
    if exact:
        _, surah_no, ayah_no = exact
        return lookup_exact(surah_no, ayah_no)

    # ── Step 2: Reject non-Islamic queries ───────────────────
    if not is_islamic_query(user_message):
        return None

    # ── Step 3: Semantic RAG search ──────────────────────────
    expanded        = expand_query(user_message)
    query_embedding = _model.encode(expanded).tolist()

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    SIMILARITY_THRESHOLD = 0.65
    output = []
    for i in range(len(results["ids"][0])):
        if results["distances"][0][i] < SIMILARITY_THRESHOLD:
            meta    = results["metadatas"][0][i]
            english = results["documents"][0][i]
            output.append(
                f"Verse {i+1}:\n"
                f"Arabic: {meta['arabic']}\n"
                f"English: {english}\n"
                f"Reference: {meta['reference']}"
            )

    return "\n\n".join(output) if output else None


# ════════════════════════════════════════════════════════════
# DAILY VERSE
# ════════════════════════════════════════════════════════════

def get_random_verse() -> str:
    import random
    total  = _collection.count()
    offset = random.randint(0, total - 1)
    results = _collection.get(
        limit=1, offset=offset,
        include=["documents", "metadatas"]
    )
    meta    = results["metadatas"][0]
    english = results["documents"][0]
    return (
        f"Arabic: {meta['arabic']}\n"
        f"English: {english}\n"
        f"Reference: {meta['reference']}"
    )