# =========================================================
# VARSTYLE MVP - STREAMLIT APP
# =========================================================

import os

import streamlit as st
import pandas as pd
import numpy as np
import ast

from PIL import Image

from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="VARStyle",

    page_icon="👕",

    layout="wide"
)

# =========================================================
# PATH CONFIG
# =========================================================

BASE_DIR = os.path.dirname(__file__)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images"
)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_products():

    df = pd.read_csv(
        os.path.join(
            DATA_DIR,
            "products.csv"
        )
    )

    # ============================================
    # PARSE LIST COLUMN
    # ============================================

    df["body_type_match"] = df[
        "body_type_match"
    ].apply(ast.literal_eval)

    return df


@st.cache_data
def load_embeddings():

    return np.load(
        os.path.join(
            DATA_DIR,
            "fashion_embeddings.npy"
        )
    )


products = load_products()

embeddings = load_embeddings()

# =========================================================
# HEADER
# =========================================================

st.title("👕 VARStyle")

st.markdown(
    """
    ### Personal AI Outfit Recommender

    Hybrid Recommendation System:
    - Rule-Based Recommendation
    - Visual Similarity AI
    - Context-Aware Recommendation
    - Explainable Recommendation
    """
)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "User Preferences"
)

user_body_type = st.sidebar.selectbox(

    "Body Type",

    [
        "pear",

        "apple",

        "rectangle",

        "hourglass"
    ]
)

user_formality = st.sidebar.selectbox(

    "Activity Type",

    [
        "casual",

        "formal",

        "sports",

        "semi_formal"
    ]
)

user_activity = st.sidebar.radio(

    "Environment",

    [
        "indoor",

        "outdoor"
    ]
)

user_style = st.sidebar.selectbox(

    "Preferred Style",

    [
        "minimalist",

        "sporty",

        "streetwear",

        "smart_casual",

        "casual",

        "ethnic"
    ]
)

available_colors = sorted(

    products[
        "baseColour"
    ].dropna().unique()
)

user_color = st.sidebar.selectbox(

    "Preferred Color",

    available_colors
)

generate_button = st.sidebar.button(
    "Generate Recommendation"
)

# =========================================================
# SCORE FUNCTION
# =========================================================

def calculate_score(row):

    score = 0

    if user_body_type in row[
        "body_type_match"
    ]:

        score += 0.4

    if row[
        "style_group"
    ] == user_style:

        score += 0.25

    if row[
        "formal_level"
    ] == user_formality:

        score += 0.2

    if row[
        "indoor_outdoor"
    ] in [user_activity, "both"]:

        score += 0.1

    if row[
        "baseColour"
    ] == user_color:

        score += 0.05

    return score

# =========================================================
# EXPLANATION FUNCTION
# =========================================================

def generate_explanation(row):

    reasons = []

    if user_body_type in row[
        "body_type_match"
    ]:

        reasons.append(
            "✔ Suitable for your body type"
        )

    if row[
        "formal_level"
    ] == user_formality:

        reasons.append(
            "✔ Matches activity context"
        )

    if row[
        "style_group"
    ] == user_style:

        reasons.append(
            "✔ Consistent with your style"
        )

    if row[
        "baseColour"
    ] == user_color:

        reasons.append(
            "✔ Matches preferred color"
        )

    if row[
        "visual_similarity"
    ] > 0.75:

        reasons.append(
            "✔ High visual similarity"
        )

    return reasons

# =========================================================
# MAIN PROCESS
# =========================================================

if generate_button:

    filtered_df = products.copy()

    # =====================================================
    # FILTERING
    # =====================================================

    filtered_df = filtered_df[

        filtered_df[
            "body_type_match"
        ].apply(

            lambda x:
            user_body_type in x
            or "all" in x
        )
    ]

    # =====================================================
    # FALLBACK STRATEGY
    # =====================================================
    
    if len(filtered_df) < 5:
    
        st.info(
            "Using broader recommendation matching..."
        )
    
        filtered_df = products.copy()
    
        filtered_df = filtered_df[
        
            filtered_df[
                "body_type_match"
            ].apply(
            
                lambda x:
                user_body_type in x
                or "all" in x
            )
        ]
    
    # =================================================
    # RULE SCORE
    # =================================================
    
    filtered_df["score"] = filtered_df.apply(
    
        calculate_score,
    
        axis=1
    )
    
    # =================================================
    # GET EMBEDDINGS
    # =================================================
    
    embedding_indices = filtered_df[
        "embedding_idx"
    ].values
    
    filtered_embeddings = embeddings[
        embedding_indices
    ]
    
    # =================================================
    # CENTROID REFERENCE EMBEDDING
    # =================================================
    
    reference_embedding = np.mean(
        filtered_embeddings,
        axis=0
    )
    
    # =================================================
    # VECTORIZE COSINE SIMILARITY
    # =================================================
    
    similarities = cosine_similarity(
    
        filtered_embeddings,
    
        reference_embedding.reshape(1,-1)
    
    ).flatten()
    
    filtered_df[
        "visual_similarity"
    ] = similarities
    
    # =================================================
    # NORMALIZE SCORES
    # =================================================
    
    filtered_df["score"] = (
    
        filtered_df["score"]
    
        /
    
        filtered_df["score"].max()
    )
    
    # =================================================
    # FINAL SCORE
    # =================================================
    
    filtered_df["final_score"] = (
    
        0.7 * filtered_df["score"]
    
        +
    
        0.3 * filtered_df[
            "visual_similarity"
        ]
    )
    
    # =================================================
    # EXPLANATION
    # =================================================
    
    filtered_df["explanation"] = filtered_df.apply(
    
        generate_explanation,
    
        axis=1
    )
    
    # =================================================
    # SORTING
    # =================================================
    
    filtered_df = filtered_df.sort_values(
    
        by="final_score",
    
        ascending=False
    )
    
    top_df = filtered_df.head(6)
    
    # =================================================
    # SUMMARY
    # =================================================
    
    st.success(
        f"{len(top_df)} recommendations generated"
    )
    
    st.markdown("---")
    
    st.subheader(
        "Recommended Outfits"
    )
    
    # =================================================
    # DISPLAY
    # =================================================
    
    cols = st.columns(3)
    
    for idx, row in enumerate(
        top_df.itertuples()
    ):
    
        with cols[idx % 3]:
        
            image_path = os.path.join(
            
                IMAGE_DIR,
    
                row.image_file
            )
    
            # =========================================
            # SAFE IMAGE LOADING
            # =========================================
    
            if os.path.exists(image_path):
            
                image = Image.open(
                    image_path
                )
    
                st.image(
                    image,
                    use_container_width=True
                )
    
            else:
            
                st.warning(
                    "Image not found"
                )
    
            st.markdown(
                f"### {row.articleType}"
            )
    
            st.markdown(
                f"""
                **Style:** {row.style_group}  
                **Color:** {row.baseColour}  
                **Final Score:** {row.final_score:.2f}  
                **Similarity:** {row.visual_similarity:.2f}
                """
            )
    
            st.markdown(
                "#### Why Recommended?"
            )
    
            for reason in row.explanation:
            
                st.write(reason)
    
            st.markdown("---")
    
# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "VARStyle MVP — Hybrid AI Fashion Recommendation System"
)