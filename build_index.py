"""
Run this ONCE to build the Quran vector index.
It will take 5-15 minutes depending on your machine.
After it finishes you never need to run it again.
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

# ── Load Quran dataset ───────────────────────────────────────
print("📖 Loading Quran dataset...")
with open("quran_en.json", encoding="utf-8") as f:
    quran = json.load(f)

# ── Flatten all verses into a list ──────────────────────────
verses = []
for surah in quran:
    surah_number   = surah["id"]
    surah_name_en  = surah["translation"]      # e.g. "The Opening"
    surah_name_ar  = surah["transliteration"]  # e.g. "Al-Fatihah"

    for verse in surah["verses"]:
        ayah_number = verse["id"]
        arabic_text = verse.get("text", "")
        english_text = verse.get("translation", "")

        verses.append({
            "id":          f"{surah_number}:{ayah_number}",
            "surah_no":    surah_number,
            "ayah_no":     ayah_number,
            "surah_en":    surah_name_en,
            "surah_ar":    surah_name_ar,
            "arabic":      arabic_text,
            "english":     english_text,
            "reference":   f"Surah {surah_name_ar} ({surah_number}:{ayah_number})"
        })

print(f"✅ Loaded {len(verses)} verses from 114 surahs")

# ── Load embedding model (downloads once, ~90MB) ─────────────
print("\n🔄 Loading embedding model (may download on first run)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded")

# ── Generate embeddings for all English translations ─────────
print("\n🔄 Embedding all verses — this takes a few minutes...")
texts = [v["english"] for v in verses]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
print("✅ Embeddings done")

# ── Store in ChromaDB ────────────────────────────────────────
print("\n🔄 Saving to ChromaDB...")
chroma_client = chromadb.PersistentClient(path="./quran_index")

# Delete existing collection if rebuilding
try:
    chroma_client.delete_collection("quran")
except:
    pass

collection = chroma_client.create_collection(
    name="quran",
    metadata={"hnsw:space": "cosine"}
)

# Insert in batches of 500 to avoid memory issues
BATCH_SIZE = 500
for i in range(0, len(verses), BATCH_SIZE):
    batch_verses     = verses[i : i + BATCH_SIZE]
    batch_embeddings = embeddings[i : i + BATCH_SIZE].tolist()

    collection.add(
        ids        = [v["id"] for v in batch_verses],
        embeddings = batch_embeddings,
        documents  = [v["english"] for v in batch_verses],
        metadatas  = [{
            "surah_no":  v["surah_no"],
            "ayah_no":   v["ayah_no"],
            "surah_en":  v["surah_en"],
            "surah_ar":  v["surah_ar"],
            "arabic":    v["arabic"],
            "reference": v["reference"]
        } for v in batch_verses]
    )
    print(f"  Saved verses {i+1} to {min(i+BATCH_SIZE, len(verses))}")

print(f"\n✅ All {len(verses)} verses saved to ChromaDB")
print("🎉 Index built! You can now run test_bot.py")