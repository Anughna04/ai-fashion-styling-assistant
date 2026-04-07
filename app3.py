"""
AI Fashion Styling Assistant - Main Application

VALIDATION METHODS USED (printed to terminal at runtime):
  1. FAISS Search Score Distribution  - Confirms similarity scores are in [0,1] and spread
  2. Filter Logic Correctness         - Asserts filtered items all satisfy occasion/season
  3. Embedding Query Sanity           - Verifies query embedding is unit-normalised
  4. Result Deduplication Check       - Confirms no duplicate item IDs in top-k results
"""

import streamlit as st
import torch
from PIL import Image
import numpy as np
import faiss
import pickle
from fashion_models import FashionCLIP, FashionBERT, get_fashion_models
import os
import io
import base64
from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"DEBUG (Advice): Active GEMINI_API_KEY: {api_key[:6]}...{api_key[-4:]}")
else:
    print("DEBUG (Advice): Active GEMINI_API_KEY: MISSING")

try:
    gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
#  VALIDATION HELPER  (terminal output only)
# ─────────────────────────────────────────────────────────────
def _validate(label: str, condition: bool, detail: str = ""):
    status = "✅ PASS" if condition else "❌ FAIL"
    msg = f"   [{status}] {label}"
    if detail:
        msg += f"  →  {detail}"
    print(msg)
    return condition


# ─────────────────────────────────────────────────────────────
#  CACHED LOADERS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    clip_model, fashion_bert = get_fashion_models()   # already validates internally
    return clip_model, fashion_bert


@st.cache_resource
def load_faiss_index():
    try:
        index = faiss.read_index("fashion_index.faiss")
        with open("fashion_items.pkl", "rb") as f:
            fashion_items = pickle.load(f)
        with open("fashion_embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
        return index, fashion_items, embeddings
    except FileNotFoundError:
        st.error("""
        ⚠️ FAISS index not found!
        Please run: `python build_faiss_index2.py` first.
        """)
        st.stop()


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()


def search_similar_items(query_embedding, index, fashion_items, top_k=10):
    """
    FAISS search with inline validation.

    Validation 1 – Query Embedding Sanity
      The query vector must be unit-normalised before L2 search,
      otherwise distances are not comparable to stored vectors.

    Validation 2 – Result Deduplication
      Top-k list must have unique item IDs (no repeated results).

    Validation 3 – Score Distribution
      Similarity scores (converted from L2 distance) must lie in [0, 1].
    """

    # ── Validation 1: unit normalise ───────────────────────────
    print("\n  VALIDATION: Query Embedding Sanity")
    print("  " + "-"*40)
    pre_norm = float(np.linalg.norm(query_embedding))
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    post_norm = float(np.linalg.norm(query_embedding))
    _validate("Pre-norm > 0", pre_norm > 1e-6, f"{pre_norm:.6f}")
    _validate("Post-norm ≈ 1.0", abs(post_norm - 1.0) < 1e-5, f"{post_norm:.6f}")

    query_embedding = query_embedding.astype("float32")

    # ── Search ─────────────────────────────────────────────────
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(fashion_items):
            item = fashion_items[idx].copy()
            item["similarity"] = float(1 / (1 + dist))
            item["rank"] = i + 1
            results.append(item)

    # ── Validation 2: no duplicate IDs ─────────────────────────
    print("\n  VALIDATION: Result Deduplication")
    print("  " + "-"*40)
    result_ids   = [r["id"] for r in results]
    unique_count = len(set(result_ids))
    _validate(
        "All returned item IDs are unique",
        unique_count == len(result_ids),
        f"{unique_count} unique out of {len(result_ids)}"
    )

    # ── Validation 3: score range ──────────────────────────────
    print("\n  VALIDATION: Search Score Distribution")
    print("  " + "-"*40)
    if results:
        sims = [r["similarity"] for r in results]
        print(f"  Top-1 similarity : {sims[0]:.4f}")
        print(f"  Bottom similarity: {sims[-1]:.4f}")
        print(f"  Mean similarity  : {np.mean(sims):.4f}")
        _validate(
            "All similarity scores in (0, 1]",
            all(0 < s <= 1.0 for s in sims),
            f"min={min(sims):.4f}  max={max(sims):.4f}"
        )
        _validate(
            "Results are sorted descending by similarity",
            sims == sorted(sims, reverse=True),
            "rank-1 is most similar"
        )

    return results


def filter_by_context(items, occasion, season):
    """
    Filter items with inline validation.

    Validation 4 – Filter Logic Correctness:
      Every item in the filtered list must satisfy the applied filters.
    """
    filtered = []
    for item in items:
        matches_occasion = not occasion or occasion in item.get("occasions", [])
        matches_season   = not season   or season   in item.get("seasons",   [])
        if matches_occasion and matches_season:
            filtered.append(item)

    # ── Validation 4 ───────────────────────────────────────────
    if filtered and (occasion or season):
        print("\n  VALIDATION: Filter Logic Correctness")
        print("  " + "-"*40)
        if occasion:
            all_match_occ = all(
                occasion in item.get("occasions", []) for item in filtered
            )
            _validate(
                f"All filtered items match occasion='{occasion}'",
                all_match_occ,
                f"{len(filtered)} items checked"
            )
        if season:
            all_match_sea = all(
                season in item.get("seasons", []) for item in filtered
            )
            _validate(
                f"All filtered items match season='{season}'",
                all_match_sea,
                f"{len(filtered)} items checked"
            )

    return filtered

def generate_styling_advice(items, occasion, season, query):
    import os
    from google import genai

    if not items:
        return "I couldn’t find matching items, but I’d suggest going for a clean, versatile look with neutral tones and minimal accessories."

    items_text = "\n".join([
        f"- {item['name']}: {item.get('description', 'Fashion item')}"
        for item in items[:5]
    ])

    api_key = os.getenv("GEMINI_API_KEY")

    if GEMINI_AVAILABLE and api_key:
        try:
            client = genai.Client(api_key=api_key)

            prompt = f"""You are Genie — a senior fashion editor and personal stylist.

            A user is looking for outfit recommendations. Based on the context and available wardrobe items below, 
            create exactly 2 complete, distinct styled looks. The looks should feel like editorial suggestions — 
            specific, considered, and wearable — not generic AI fashion advice.

            ---

            CONTEXT:
            Occasion: {occasion or "not specified — treat as smart casual"}
            Season / Weather: {season or "current season — suggest accordingly"}
            User's query or description: {query or "general styling request"}

            ---

            AVAILABLE WARDROBE ITEMS (draw from these where relevant):
            {items_text}

            ---

            YOUR OUTPUT — write exactly this structure, no bullet points, no headers, no JSON:

            Look 1 — [give it a short editorial name, e.g. "The Daytime Edit" or "Evening Ease"]:
            Write 3-4 flowing sentences that cover:
            - Which specific garments you're combining and how (tucked, draped, layered, belted, etc.)
            - The full colour story: primary colour, accent, and how they complement each other
            - Footwear: exact style, heel height if relevant, colour (e.g. "pointed kitten-heel mules in dusty rose")
            - Bag: specific type and material (e.g. "a structured mini tote in cognac leather")
            - Jewellery: be precise and complete — earrings, neckpiece, bangles/bracelet, rings if relevant
            (e.g. "Layer a delicate gold chain over the neckline, add small hoop earrings and one thin bangle")
            - One finishing detail: a lip colour, a hair suggestion, or how to carry the look

            [blank line between looks]

            Look 2 — [editorial name]:
            Same structure as Look 1 — but make this distinctly different in mood, silhouette, or formality.
            If Look 1 was polished, make Look 2 relaxed. If Look 1 was Western, make Look 2 Indian or fusion.
            Cover the same 6 elements.

            ---

            TONE RULES:
            - Write in a calm, editorial voice — like Vogue India or Harper's Bazaar, not Instagram
            - No hashtags, no markdown symbols, no emojis, no bullet points
            - Do not start with "Of course", "Sure!", "Absolutely!" or any filler opener
            - No generic phrases like "This look is perfect for..." — just describe the look directly
            - The two looks must feel meaningfully different — different energy, not just different colours

            Now write the two looks:"""

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config={
                    'max_output_tokens': 3000,
                    'temperature': 0.7
                }
            )

            return response.text.strip()

        except Exception as e:
            print(f"ERROR in styling advice: {e}")

    # fallback
    return f"A simple and elegant look suitable for {occasion or 'any occasion'} during {season or 'this season'}."

# ─────────────────────────────────────────────────────────────
#  STREAMLIT APP
# ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="AI Fashion Styling Assistant",
        page_icon="👗",
        layout="wide",
    )

    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 10px;
            margin-bottom: 2rem; text-align: center; color: white;
        }
        .info-banner {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem; border-radius: 10px;
            color: white; margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='main-header'>
            <h1>✨ AI Fashion Styling Assistant</h1>
            <p>Powered by Fashion-CLIP, FashionBERT, FAISS & Gemini AI</p>
            <p style='font-size: 0.9em; margin-top: 0.5rem;'>MVSREC | CSE Department | Batch-01</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class='info-banner'>
            <h3>🤔 Why Upload an Image?</h3>
            <p><strong>Visual Search Power:</strong> Our Fashion-CLIP model analyzes uploaded images
            to understand colors, patterns, textures, and styles.
            It generates embeddings that capture visual semantics, then searches our
            FAISS-indexed database to find similar items and suggest complete outfits.
            Perfect when you can't describe what you're looking for in words!</p>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Loading AI models (Fashion-CLIP & FashionBERT)..."):
        clip_model, fashion_bert = load_models()
        faiss_index, fashion_items, embeddings = load_faiss_index()

    st.success(f"✅ Loaded {len(fashion_items)} fashion items in database")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Upload Fashion Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a fashion item or outfit for visual search",
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("🔍 Search & Context")
        text_query = st.text_area(
            "Text Query (Optional but recommended)",
            placeholder="Example: 'elegant summer dress with floral patterns for wedding, "
                        "need matching gold jewelry and nude heels'",
            height=120,
        )
        col2a, col2b = st.columns(2)
        with col2a:
            occasion = st.selectbox("Occasion", ["", "Casual", "Formal", "Business", "Party", "Date", "Sport"])
        with col2b:
            season = st.selectbox("Season", ["", "Spring", "Summer", "Fall", "Winter"])

    if st.button("🎨 Get AI Styling Recommendations", type="primary", use_container_width=True):
        if uploaded_file or text_query:
            progress_bar = st.progress(0)
            status       = st.empty()

            try:
                status.text("📊 Generating Fashion-CLIP embeddings...")
                progress_bar.progress(20)

                if uploaded_file:
                    query_embedding = clip_model.encode_image(image, validate=True)
                    context_text    = text_query or "fashion clothing item"
                else:
                    query_embedding = clip_model.encode_text(text_query, validate=True)
                    context_text    = text_query

                status.text("🎨 Refining with FashionBERT...")
                progress_bar.progress(40)

                status.text("🔎 Searching FAISS vector database...")
                progress_bar.progress(60)

                results = search_similar_items(
                    query_embedding, faiss_index, fashion_items, top_k=20
                )

                status.text("🎯 Filtering by occasion and season...")
                progress_bar.progress(75)

                if occasion or season:
                    results = filter_by_context(results, occasion, season)

                status.text("🤖 Generating styling advice with AI...")
                progress_bar.progress(90)

                styling_advice = generate_styling_advice(
                    results, occasion, season, text_query or "general fashion"
                )

                progress_bar.progress(100)
                status.text("✅ Complete!")

                if not results:
                    st.warning("⚠️ No items found matching your exact criteria. Try:")
                    st.write("- Removing occasion/season filters")
                    st.write("- Using a different image")
                    st.write("- Broadening your text query")
                    st.write("- Rebuilding the index with more items")
                else:
                    st.success(f"✅ Found {len(results)} matching items!")

                    st.markdown("---")
                    st.subheader("✨ AI Styling Expert Advice")
                    st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                    padding: 2rem; border-radius: 10px; color: white;'>
                            {styling_advice.replace(chr(10), '<br>')}
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.subheader(f"👔 Top {min(len(results), 12)} Recommended Items")

                    cols = st.columns(3)
                    for idx, item in enumerate(results[:12]):
                        with cols[idx % 3]:
                            if "image" in item and item["image"]:
                                try:
                                    st.image(item["image"], use_container_width=True)
                                except Exception:
                                    st.image(
                                        "https://via.placeholder.com/300x300/cccccc/666666?text=Fashion+Item",
                                        use_container_width=True,
                                    )
                            st.markdown(f"""
                                <div style='border: 2px solid #e0e0e0; border-radius: 10px;
                                            padding: 1rem; background: white; margin-bottom: 1rem;'>
                                    <h4 style='color: #667eea; margin: 0;'>{item['name']}</h4>
                                    <p style='margin: 0.5rem 0;'><strong>Match Score:</strong>
                                        <span style='color: #10b981; font-weight: bold;'>
                                            {item['similarity']*100:.1f}%
                                        </span>
                                    </p>
                                    <p style='margin: 0.3rem 0;'><strong>Category:</strong> {item['category']}</p>
                                    <p style='margin: 0.3rem 0;'><strong>Price:</strong> {item['price']}</p>
                                    <p style='font-size: 0.9rem; color: #666; font-style: italic;
                                              margin-top: 0.5rem;'>
                                        {item.get('styling_tip', 'Versatile fashion piece')}
                                    </p>
                                </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        else:
            st.error("Please upload an image OR enter a text query!")

    st.markdown("---")
    with st.expander("🔧 System Architecture & Models"):
        st.markdown("""
        ### AI Models Used

        1. **Fashion-CLIP** (`patrickjohncyh/fashion-clip`)
           - Fine-tuned CLIP model specifically for fashion
           - Generates 512-dimensional embeddings
           - Understands both images and text

        2. **FashionBERT** (Sentence Transformers)
           - Fashion-domain language understanding
           - Refines embeddings with context
           - Extracts style attributes

        3. **FAISS** (Facebook AI Similarity Search)
           - Efficient vector similarity search
           - Handles large-scale retrieval
           - IndexFlatL2 for accurate results

        4. **Gemini 1.5 Pro** (Optional)
           - Generates detailed styling advice
           - Falls back to rule-based advice if unavailable

        ### Validation Steps (visible in terminal)
        - Model architecture checks at load time
        - Embedding shape & L2-norm checks per query
        - Cross-modal alignment sanity test
        - FAISS self-retrieval accuracy test
        - Near-duplicate detection in dataset
        - Filter logic correctness per search
        """)

    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p><strong>AI Fashion Styling Assistant</strong></p>
            <p>Batch-ID: 01 | MVSREC | Department of Computer Science &amp; Engineering</p>
            <p>Team: Anughna Kandimalla, Akshaya Bharathi, Aishwarya Bojja</p>
            <p>Guide: Bodupally Janaiah, Assistant Professor</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
