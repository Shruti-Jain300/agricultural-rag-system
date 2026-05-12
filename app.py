import os
import logging

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Suppress Streamlit's noisy "No secrets file found" warnings
logging.getLogger("streamlit.runtime.secrets").setLevel(logging.ERROR)

from pathlib import Path
from typing import Dict, List, Any, Optional
import json

import numpy as np
import pandas as pd
import streamlit as st
import chromadb
import tensorflow as tf
from PIL import Image
from sentence_transformers import SentenceTransformer

from llm_client import generate_grounded_answer, DEFAULT_MODEL


# ============================================================
# Streamlit Page Config
# ============================================================
st.set_page_config(
    page_title="Agricultural Knowledge Retrieval System with RAG",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# Safe Settings Helper
# ============================================================
def get_setting(name: str, default=None):
    """Read from st.secrets first, then environment variables, then default."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


# ============================================================
# App Constants
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# General RAG
GENERAL_RUN_ID = get_setting("GENERAL_RUN_ID", "20260511_201239")
COLLECTION_NAME = get_setting("COLLECTION_NAME", f"agrigenius_{GENERAL_RUN_ID}")
CHUNKS_PATH = BASE_DIR / "data" / "chunks.parquet"
CHROMA_PATH = str(BASE_DIR / "chroma_db")

# Cotton RAG
COTTON_RUN_ID = get_setting("COTTON_RUN_ID", "20260510_162258")
COTTON_COLLECTION_NAME = get_setting("COTTON_COLLECTION_NAME", f"cotton_guide_{COTTON_RUN_ID}")
COTTON_CHUNKS_PATH = BASE_DIR / "data" / "cotton_chunks.parquet"
COTTON_CHROMA_PATH = str(BASE_DIR / "cotton_chroma_db")

# Embeddings
EMBED_MODEL_NAME = get_setting("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

# Models
MODELS_DIR = BASE_DIR / "models"
POTATO_MODEL_PATH = MODELS_DIR / "potato_classification_model.h5"
TOMATO_MODEL_PATH = MODELS_DIR / "tomato_classification_model.h5"
COTTON_MODEL_PATH = MODELS_DIR / "cotton_plant_disease_classifier.h5"

# Advisory JSON
ADVISORY_DIR = BASE_DIR / "Advisory"
POTATO_TOMATO_REMEDIES_PATH = ADVISORY_DIR / "potato_tomato_remedies.json"
COTTON_REMEDIES_PATH = ADVISORY_DIR / "cotton_remedies.json"

# Classes
POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

TOMATO_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

COTTON_CLASSES = [
    "Aphids",
    "Army worm",
    "Bacterial Blight",
    "Healthy leaf",
    "Powdery Mildew",
    "Target spot",
]


# ============================================================
# App Header
# ============================================================
st.title("🌾 Agricultural Knowledge Retrieval System with RAG")
st.subheader("AgriAssist Project")

st.write(
    "Ask agriculture-related questions in the General RAG tab. "
    "Use the Leaf Advisory tab for potato and tomato leaf disease prediction. "
    "Use the Cotton Disease tab for cotton leaf disease prediction. "
    "Use the Cotton Farming Assistant tab for cotton-specific advisory and irrigation guidance from the cotton production guide."
)


# ============================================================
# Cached Resources
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


def _build_chroma_collection_in_memory(chunks_df: pd.DataFrame) -> chromadb.Collection:
    """
    Build a ChromaDB collection using EphemeralClient (in-memory).
    Used as a fallback when the persistent DB folder is locked or corrupt.
    """
    client = chromadb.EphemeralClient()
    collection = client.create_collection(name=COLLECTION_NAME)
    _populate_collection(collection, chunks_df)
    return collection


def _populate_collection(collection: chromadb.Collection, chunks_df: pd.DataFrame):
    """Encode and insert all chunks into the given collection."""
    model = load_embedding_model()
    docs = chunks_df["chunk_text"].astype(str).tolist()
    ids = chunks_df["chunk_id"].astype(str).tolist()

    metadatas = []
    for _, row in chunks_df.iterrows():
        metadatas.append({
            "source_type": str(row.get("source_type", "")),
            "source_name": str(row.get("source_name", "")),
            "file_name": str(row.get("file_name", "")),
            "chunk_index_in_file": int(row.get("chunk_index_in_file", 0)),
            "chunk_words": int(row.get("chunk_words", 0)),
            "chunk_chars": int(row.get("chunk_chars", 0)),
        })

    batch_size = 64
    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))
        batch_docs = docs[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]
        batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embeddings,
        )


@st.cache_resource
def build_or_load_vectordb():
    """
    Load the general vector DB if healthy.
    On Windows, shutil.rmtree can fail with PermissionError (WinError 32) when
    ChromaDB's SQLite file is still open from a previous session/hot-reload.
    Strategy:
      1. Try to open the persistent collection and health-check it.
      2. If that fails, try to delete and rebuild the persistent DB.
      3. If deletion is blocked (file lock), fall back to an in-memory EphemeralClient.
    """
    import shutil

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"General chunks.parquet not found at: {CHUNKS_PATH}")

    chunks_df = pd.read_parquet(CHUNKS_PATH)
    chunks_df["chunk_text"] = chunks_df["chunk_text"].fillna("").astype(str)
    chunks_df = chunks_df[chunks_df["chunk_text"].str.strip() != ""].copy()

    # --- Step 1: Try to use the existing persistent collection ---
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=COLLECTION_NAME)

        # Health-check: run a tiny query
        test_embedding = load_embedding_model().encode("test query").tolist()
        collection.query(
            query_embeddings=[test_embedding],
            n_results=1,
            include=["documents"],
        )
        return collection, len(chunks_df)

    except Exception:
        pass  # Fall through to rebuild

    # --- Step 2: Try to delete and rebuild the persistent DB ---
    try:
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.create_collection(name=COLLECTION_NAME)
        _populate_collection(collection, chunks_df)
        return collection, len(chunks_df)

    except PermissionError:
        # --- Step 3: Folder is locked (Windows WinError 32) — use in-memory DB ---
        st.warning(
            "⚠️ The persistent vector database is locked by another process. "
            "Using an in-memory database for this session. "
            "Close any other Streamlit instances and restart to fix this."
        )
        collection = _build_chroma_collection_in_memory(chunks_df)
        return collection, len(chunks_df)


@st.cache_resource
def load_cotton_vectordb():
    """
    Loads the already-built cotton-specific Chroma collection.
    """
    if not COTTON_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"cotton_chunks.parquet not found at: {COTTON_CHUNKS_PATH}")

    if not os.path.exists(COTTON_CHROMA_PATH):
        raise FileNotFoundError(f"cotton_chroma_db not found at: {COTTON_CHROMA_PATH}")

    chunks_df = pd.read_parquet(COTTON_CHUNKS_PATH)
    chunks_df["chunk_text"] = chunks_df["chunk_text"].fillna("").astype(str)
    chunks_df = chunks_df[chunks_df["chunk_text"].str.strip() != ""].copy()

    client = chromadb.PersistentClient(path=COTTON_CHROMA_PATH)

    collections = client.list_collections()
    existing_names = [c.name if hasattr(c, "name") else str(c) for c in collections]

    if COTTON_COLLECTION_NAME not in existing_names:
        raise ValueError(
            f"Cotton collection '{COTTON_COLLECTION_NAME}' not found inside: {COTTON_CHROMA_PATH}"
        )

    collection = client.get_collection(name=COTTON_COLLECTION_NAME)
    return collection, len(chunks_df)


@st.cache_resource
def load_leaf_models():
    """
    Loads legacy .h5 leaf disease models with compatibility patches
    for dtype policy and preprocessing / augmentation layer deserialization.
    """
    if not POTATO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Potato model not found at: {POTATO_MODEL_PATH}")

    if not TOMATO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Tomato model not found at: {TOMATO_MODEL_PATH}")

    if not COTTON_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cotton model not found at: {COTTON_MODEL_PATH}")

    def patch_dtype(kwargs):
        dtype_cfg = kwargs.get("dtype")
        if isinstance(dtype_cfg, dict):
            kwargs["dtype"] = dtype_cfg.get("config", {}).get("name", "float32")
        return kwargs

    def patch_common_kwargs(kwargs):
        kwargs.pop("data_format", None)
        kwargs.pop("pad_to_aspect_ratio", None)
        kwargs.pop("fill_mode", None)
        kwargs.pop("fill_value", None)
        kwargs.pop("antialias", None)
        kwargs = patch_dtype(kwargs)
        return kwargs

    class PatchedInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedResizing(tf.keras.layers.Resizing):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRescaling(tf.keras.layers.Rescaling):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomFlip(tf.keras.layers.RandomFlip):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomRotation(tf.keras.layers.RandomRotation):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomZoom(tf.keras.layers.RandomZoom):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomContrast(tf.keras.layers.RandomContrast):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomTranslation(tf.keras.layers.RandomTranslation):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    policy_cls = tf.keras.mixed_precision.Policy

    custom_objects = {
        "InputLayer": PatchedInputLayer,
        "Resizing": PatchedResizing,
        "Rescaling": PatchedRescaling,
        "RandomFlip": PatchedRandomFlip,
        "RandomRotation": PatchedRandomRotation,
        "RandomZoom": PatchedRandomZoom,
        "RandomContrast": PatchedRandomContrast,
        "RandomTranslation": PatchedRandomTranslation,
        "DTypePolicy": policy_cls,
        "Policy": policy_cls,
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        potato_model = tf.keras.models.load_model(
            POTATO_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )
        tomato_model = tf.keras.models.load_model(
            TOMATO_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )
        cotton_model = tf.keras.models.load_model(
            COTTON_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )

    return {
        "potato": potato_model,
        "tomato": tomato_model,
        "cotton": cotton_model,
    }


@st.cache_resource
def load_json_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_potato_tomato_remedies():
    return load_json_file(POTATO_TOMATO_REMEDIES_PATH)


@st.cache_resource
def load_cotton_remedies():
    return load_json_file(COTTON_REMEDIES_PATH)


# ============================================================
# RAG Utility Functions
# ============================================================
def retrieve_documents(
    query: str,
    top_k: int,
    model,
    collection
) -> Dict[str, List[Any]]:
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "docs": results.get("documents", [[]])[0],
        "metas": results.get("metadatas", [[]])[0],
        "distances": results.get("distances", [[]])[0],
        "ids": results.get("ids", [[]])[0],
    }


def run_rag_pipeline(
    query: str,
    top_k: int,
    llm_context_k: int,
    model,
    collection
) -> Dict[str, Any]:
    retrieved = retrieve_documents(
        query=query,
        top_k=top_k,
        model=model,
        collection=collection
    )

    docs = retrieved["docs"]
    metas = retrieved["metas"]
    distances = retrieved["distances"]
    ids = retrieved["ids"]

    answer: Optional[str] = None
    error_message: Optional[str] = None

    if docs:
        try:
            answer = generate_grounded_answer(
                query=query,
                docs=docs,
                metas=metas,
                max_chunks=llm_context_k,
            )
        except Exception as e:
            error_message = f"LLM generation failed: {e}"
            answer = "The system retrieved relevant evidence, but grounded answer generation failed."
    else:
        answer = "No relevant documents were retrieved for this query."

    return {
        "query": query,
        "answer": answer,
        "error_message": error_message,
        "docs": docs,
        "metas": metas,
        "distances": distances,
        "ids": ids,
    }


def render_generated_answer(answer: Optional[str], error_message: Optional[str]):
    st.subheader("Generated Answer")

    if answer:
        st.success(answer)

    if error_message:
        st.error(error_message)
        st.info("Retrieved evidence is still shown below.")


def render_retrieved_evidence(
    docs: List[str],
    metas: List[Dict[str, Any]],
    distances: List[Any],
    ids: List[str]
):
    st.subheader("Retrieved Evidence")

    if not docs:
        st.warning("No evidence found for this query.")
        return

    for i in range(len(docs)):
        distance_value = distances[i] if i < len(distances) else None
        similarity = None

        try:
            if distance_value is not None:
                similarity = 1 - float(distance_value)
        except Exception:
            similarity = None

        title = f"Result {i + 1}"
        if distance_value is not None:
            try:
                title += f" | Distance: {float(distance_value):.4f}"
            except Exception:
                pass
        if similarity is not None:
            title += f" | Similarity: {similarity:.4f}"

        with st.expander(title, expanded=(i == 0)):
            meta = metas[i] if i < len(metas) and metas[i] else {}

            st.write(f"**Chunk ID:** {ids[i] if i < len(ids) else 'N/A'}")
            st.write(f"**Source Type:** {meta.get('source_type', '')}")
            st.write(f"**Source Name:** {meta.get('source_name', '')}")
            st.write(f"**File Name:** {meta.get('file_name', '')}")
            st.write(f"**Chunk Index:** {meta.get('chunk_index_in_file', '')}")
            st.write("**Retrieved Text:**")
            st.write(docs[i])


# ============================================================
# Advisory Utility Functions
# ============================================================
def render_remedy_details(remedy_data: Dict[str, Any]):
    st.subheader("Disease Advisory")

    st.write(f"**Disease:** {remedy_data.get('disease_name', 'N/A')}")
    st.write(f"**Crop:** {remedy_data.get('crop', 'N/A')}")
    st.write(f"**Cause:** {remedy_data.get('cause', 'N/A')}")

    symptoms = remedy_data.get("symptoms", [])
    remedy = remedy_data.get("remedy", [])
    prevention = remedy_data.get("prevention", [])
    severity_note = remedy_data.get("severity_note", "")

    if symptoms:
        st.write("**Symptoms**")
        for item in symptoms:
            st.write(f"- {item}")

    if remedy:
        st.write("**Recommended Remedy**")
        for item in remedy:
            st.write(f"- {item}")

    if prevention:
        st.write("**Prevention**")
        for item in prevention:
            st.write(f"- {item}")

    if severity_note:
        st.info(severity_note)


# ============================================================
# Leaf Advisory Utility Functions
# ============================================================
def preprocess_leaf_image(uploaded_file, target_size=(256, 256)):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_with_single_model(model, img_array, class_names, crop_name):
    prediction = model.predict(img_array, verbose=0)
    prediction = np.array(prediction)

    if prediction.ndim > 1:
        prediction = prediction[0]

    prediction = np.array(prediction, dtype=np.float32)

    if (
        np.max(prediction) > 1.0
        or np.min(prediction) < 0.0
        or abs(np.sum(prediction) - 1.0) > 0.05
    ):
        exp_scores = np.exp(prediction - np.max(prediction))
        prediction = exp_scores / np.sum(exp_scores)

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100.0
    predicted_class = class_names[predicted_index]

    return {
        "crop": crop_name,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2)
    }


def predict_leaf_disease(uploaded_file, models_dict):
    display_image, img_array = preprocess_leaf_image(uploaded_file)

    potato_result = predict_with_single_model(
        model=models_dict["potato"],
        img_array=img_array,
        class_names=POTATO_CLASSES,
        crop_name="Potato"
    )

    tomato_result = predict_with_single_model(
        model=models_dict["tomato"],
        img_array=img_array,
        class_names=TOMATO_CLASSES,
        crop_name="Tomato"
    )

    all_results = [potato_result, tomato_result]
    best_result = max(all_results, key=lambda x: x["confidence"])

    return display_image, best_result, all_results


def predict_cotton_disease(uploaded_file, models_dict):
    display_image, img_array = preprocess_leaf_image(
        uploaded_file,
        target_size=(180, 180)
    )

    cotton_result = predict_with_single_model(
        model=models_dict["cotton"],
        img_array=img_array,
        class_names=COTTON_CLASSES,
        crop_name="Cotton"
    )

    return display_image, cotton_result


def format_prediction_label(predicted_class: str) -> str:
    return predicted_class.replace("___", " - ").replace("_", " ")


# ============================================================
# Load Shared Resources
# ============================================================
embedding_model = load_embedding_model()

collection = None
chunk_count = 0
vectordb_error = None

try:
    collection, chunk_count = build_or_load_vectordb()
except Exception as e:
    vectordb_error = str(e)

cotton_collection_error = None
cotton_collection = None
cotton_chunk_count = 0

try:
    cotton_collection, cotton_chunk_count = load_cotton_vectordb()
except Exception as e:
    cotton_collection_error = str(e)

potato_tomato_remedies_error = None
potato_tomato_remedies = {}
try:
    potato_tomato_remedies = load_potato_tomato_remedies()
except Exception as e:
    potato_tomato_remedies_error = str(e)

cotton_remedies_error = None
cotton_remedies = {}
try:
    cotton_remedies = load_cotton_remedies()
except Exception as e:
    cotton_remedies_error = str(e)


# ============================================================
# Sidebar
# ============================================================
if collection is not None:
    st.sidebar.success(f"General collection ready: {chunk_count} chunks")
else:
    st.sidebar.error("General collection unavailable")
    if vectordb_error:
        st.sidebar.caption(vectordb_error)

st.sidebar.write(f"Embedding model: {EMBED_MODEL_NAME}")
st.sidebar.write(f"General collection: {COLLECTION_NAME}")
st.sidebar.write(f"LLM: {DEFAULT_MODEL}")

if cotton_collection is not None:
    st.sidebar.success(f"Cotton collection ready: {cotton_chunk_count} chunks")
    st.sidebar.write(f"Cotton collection: {COTTON_COLLECTION_NAME}")
else:
    st.sidebar.warning("Cotton collection not loaded")
    if cotton_collection_error:
        st.sidebar.caption(cotton_collection_error)

top_k = st.sidebar.slider("Top-K retrieval results", min_value=1, max_value=10, value=5)
llm_context_k = st.sidebar.slider("Chunks sent to LLM", min_value=1, max_value=5, value=3)


# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 General RAG Search",
    "🌿 Leaf Advisory System",
    "🧵 Cotton Disease Prediction",
    "💧 Cotton Farming Assistant & Irrigation Advisor"
])


# ============================================================
# Tab 1 - General RAG Search
# ============================================================
with tab1:
    if collection is None:
        st.error("❌ General vector database could not be loaded.")
        if vectordb_error:
            st.exception(Exception(vectordb_error))
        st.info(
            "**Fix (local):** Close all other Streamlit instances, then restart.\n\n"
            "**Fix (cloud):** Ensure `chroma_db/` folder is included in your repo or built at startup."
        )
    else:
        query = st.text_input("Ask an agriculture question:", key="general_query")

        examples = [
            "What irrigation schemes support farmers?",
            "What is the purpose of PMKSY irrigation scheme?",
            "What government support is available for farmers?",
            "What agricultural statistics are available from government sources?",
        ]

        st.caption("Example queries:")
        st.code("\n".join(examples), language="text")

        if st.button("Search", key="search_button"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Retrieving evidence and generating grounded answer..."):
                    result = run_rag_pipeline(
                        query=query.strip(),
                        top_k=top_k,
                        llm_context_k=llm_context_k,
                        model=embedding_model,
                        collection=collection
                    )

                render_generated_answer(
                    answer=result["answer"],
                    error_message=result["error_message"]
                )
                render_retrieved_evidence(
                    docs=result["docs"],
                    metas=result["metas"],
                    distances=result["distances"],
                    ids=result["ids"]
                )


# ============================================================
# Tab 2 - Leaf Advisory (Potato + Tomato)
# ============================================================
with tab2:
    st.write(
        "Upload a leaf image to predict disease and view crop advisory details for potato and tomato. "
        "This tab uses trained classification models and a fixed advisory knowledge base."
    )

    uploaded_image = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_image_upload"
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Leaf Image", width=350)

        if st.button("Predict Disease", key="predict_leaf_button"):
            with st.spinner("Loading leaf models and running prediction..."):
                try:
                    leaf_models = load_leaf_models()

                    _, best_result, all_results = predict_leaf_disease(
                        uploaded_file=uploaded_image,
                        models_dict=leaf_models
                    )

                    st.session_state["leaf_best_result"] = best_result
                    st.session_state["leaf_all_results"] = all_results

                except Exception as e:
                    st.error(f"Leaf models could not be loaded or prediction failed: {e}")

    if "leaf_best_result" in st.session_state:
        best_result = st.session_state["leaf_best_result"]
        all_results = st.session_state.get("leaf_all_results", [])

        st.subheader("Prediction Result")
        st.success(
            f"Predicted Class: {format_prediction_label(best_result['predicted_class'])}"
        )
        st.info(f"Predicted Crop: {best_result['crop']}")
        st.info(f"Confidence: {best_result['confidence']}%")

        st.subheader("Model-wise Confidence Comparison")
        comparison_rows = []
        for item in all_results:
            comparison_rows.append({
                "Model": item["crop"],
                "Predicted Class": format_prediction_label(item["predicted_class"]),
                "Confidence (%)": item["confidence"]
            })

        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

        if potato_tomato_remedies_error:
            st.warning(f"Advisory file could not be loaded: {potato_tomato_remedies_error}")
        else:
            if st.button("Show Detailed Remedy", key="leaf_remedy_button"):
                remedy_data = potato_tomato_remedies.get(best_result["predicted_class"])
                if remedy_data:
                    render_remedy_details(remedy_data)
                else:
                    st.warning("No advisory details found for this disease.")
    else:
        st.info("Upload a leaf image and click Predict Disease to continue.")


# ============================================================
# Tab 3 - Cotton Disease Prediction
# ============================================================
with tab3:
    st.write(
        "Upload a cotton leaf image to predict disease and view detailed remedy guidance. "
        "This tab uses the trained cotton classification model and a fixed cotton advisory knowledge base."
    )

    uploaded_cotton_image = st.file_uploader(
        "Upload cotton leaf image",
        type=["jpg", "jpeg", "png"],
        key="cotton_image_upload"
    )

    if uploaded_cotton_image is not None:
        st.image(uploaded_cotton_image, caption="Uploaded Cotton Leaf Image", width=350)

        if st.button("Predict Cotton Disease", key="predict_cotton_button"):
            with st.spinner("Loading cotton model and running prediction..."):
                try:
                    leaf_models = load_leaf_models()

                    _, cotton_result = predict_cotton_disease(
                        uploaded_file=uploaded_cotton_image,
                        models_dict=leaf_models
                    )

                    st.session_state["cotton_result"] = cotton_result

                except Exception as e:
                    st.error(f"Cotton model could not be loaded or prediction failed: {e}")

    if "cotton_result" in st.session_state:
        cotton_result = st.session_state["cotton_result"]

        st.subheader("Cotton Prediction Result")
        st.success(
            f"Predicted Class: {format_prediction_label(cotton_result['predicted_class'])}"
        )
        st.info(f"Predicted Crop: {cotton_result['crop']}")
        st.info(f"Confidence: {cotton_result['confidence']}%")

        comparison_rows = [{
            "Model": cotton_result["crop"],
            "Predicted Class": format_prediction_label(cotton_result["predicted_class"]),
            "Confidence (%)": cotton_result["confidence"]
        }]

        st.subheader("Prediction Summary")
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

        if cotton_remedies_error:
            st.warning(f"Cotton advisory file could not be loaded: {cotton_remedies_error}")
        else:
            if st.button("Show Detailed Remedy", key="cotton_remedy_button"):
                remedy_data = cotton_remedies.get(cotton_result["predicted_class"])
                if remedy_data:
                    render_remedy_details(remedy_data)
                else:
                    st.warning("No advisory details found for this disease.")
    else:
        st.info("Upload a cotton leaf image and click Predict Cotton Disease to continue.")


# ============================================================
# Tab 4 - Cotton Farming Assistant & Irrigation Advisor
# ============================================================
with tab4:
    st.write(
        "Ask cotton-specific questions using the cotton production guide knowledge base. "
        "This tab uses a dedicated cotton LLM-RAG collection built from the cotton PDF."
    )

    cotton_examples = [
        "What irrigation recommendations are given for cotton in the guide?",
        "How should nitrogen be managed in cotton?",
        "What are the major foliar diseases of cotton?",
        "How should Palmer amaranth be managed in cotton?",
        "What insect management recommendations are given for cotton?",
        "What guidance is given for potassium and phosphorus management in cotton?"
    ]

    st.caption("Example cotton queries:")
    st.code("\n".join(cotton_examples), language="text")

    cotton_query = st.text_input(
        "Ask a cotton farming question:",
        key="cotton_rag_query"
    )

    if cotton_collection is None:
        st.error("Cotton collection is not available.")
        if cotton_collection_error:
            st.info(cotton_collection_error)
        st.info(
            "Ensure `cotton_chroma_db/` and `data/cotton_chunks.parquet` exist in your project folder."
        )
    else:
        if st.button("Search Cotton Advisory", key="search_cotton_button"):
            if not cotton_query.strip():
                st.warning("Please enter a cotton-specific query.")
            else:
                with st.spinner("Retrieving cotton evidence and generating grounded answer..."):
                    result = run_rag_pipeline(
                        query=cotton_query.strip(),
                        top_k=top_k,
                        llm_context_k=llm_context_k,
                        model=embedding_model,
                        collection=cotton_collection
                    )

                render_generated_answer(
                    answer=result["answer"],
                    error_message=result["error_message"]
                )
                render_retrieved_evidence(
                    docs=result["docs"],
                    metas=result["metas"],
                    distances=result["distances"],
                    ids=result["ids"]
                )
