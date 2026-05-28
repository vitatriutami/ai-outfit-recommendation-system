# =========================================================
# VARSTYLE MVP - STREAMLIT APP
# =========================================================

import os
import ast

import streamlit as st
import pandas as pd
import numpy as np

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
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #f8f9fa;
}

div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e6e6e6;
    padding: 10px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

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

st.markdown("""
### Personal AI Outfit Recommender

Hybrid Recommendation System:
- Rule-Based Recommendation
- Visual Similarity AI
- Context-Aware Recommendation
- Explainable Recommendation
""")

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
# 🎨 Your Fashion Profile

Personalize your AI outfit recommendation
""")

st.sidebar.markdown("---")

# =========================================================
# BODY TYPE
# =========================================================

body_type_options = {
    "🍐 Pear": "pear",
    "🍎 Apple": "apple",
    "📏 Rectangle": "rectangle",
    "⏳ Hourglass": "hourglass"
}

st.sidebar.markdown("## Body Shape")

selected_body = st.sidebar.radio(
    "",
    options=list(body_type_options.keys())
)

user_body_type = body_type_options[selected_body]

# =========================================================
# ACTIVITY TYPE
# =========================================================

formality_options = {
    "😎 Casual": "casual",
    "🤵 Formal": "formal",
    "🏃 Sports": "sports",
    "🧥 Semi Formal": "semi_formal"
}

st.sidebar.markdown("## Activity Type")

selected_formality = st.sidebar.selectbox(
    "",
    options=list(formality_options.keys())
)

user_formality = formality_options[selected_formality]

# =========================================================
# ENVIRONMENT
# =========================================================

st.sidebar.markdown("## Environment")

is_outdoor = st.sidebar.toggle(
    "🌳 Outdoor Activity"
)

user_activity = (
    "outdoor"
    if is_outdoor
    else "indoor"
)

# =========================================================
# STYLE
# =========================================================

style_options = {
    "✨ Minimalist": "minimalist",
    "🏃 Sporty": "sporty",
    "🛹 Streetwear": "streetwear",
    "💼 Smart Casual": "smart_casual",
    "👕 Casual": "casual",
    "🌸 Ethnic": "ethnic"
}

st.sidebar.markdown("## Fashion Style")

selected_style = st.sidebar.selectbox(
    "",
    options=list(style_options.keys())
)

user_style = style_options[selected_style]

# =========================================================
# COLOR
# =========================================================

available_colors = sorted(
    products[
        "baseColour"
    ].dropna().unique()
)

color_emojis = {
    "Black": "⚫",
    "White": "⚪",
    "Blue": "🔵",
    "Red": "🔴",
    "Green": "🟢",
    "Yellow": "🟡",
    "Pink": "🌸",
    "Purple": "🟣",
    "Brown": "🤎",
    "Grey": "🩶"
}

color_display = []

for color in available_colors:

    emoji = color_emojis.get(
        color,
        "🎨"
    )

    color_display.append(
        f"{emoji} {color}"
    )

st.sidebar.markdown("## Preferred Color")

selected_color_display = st.sidebar.selectbox(
    "",
    options=color_display
)

user_color = selected_color_display.split(
    " ",
    1
)[1]

# =========================================================
# GENERATE BUTTON
# =========================================================

st.sidebar.markdown("---")

generate_button = st.sidebar.button(
    "✨ Generate AI Recommendation",
    use_container_width=True
)

# =========================================================
# SCORE FUNCTION
# =========================================================

def calculate_score(row):

    score = 0

    if user_body_type in row["body_type_match"]:
        score += 0.4

    if row["style_group"] == user_style:
        score += 0.25

    if row["formal_level"] == user_formality:
        score += 0.2

    if row["indoor_outdoor"] in [user_activity, "both"]:
        score += 0.1

    if row["baseColour"] == user_color:
        score += 0.05

    return score

# =========================================================
# EXPLANATION FUNCTION
# =========================================================

def generate_explanation(row):

    reasons = []

    if user_body_type in row["body_type_match"]:
        reasons.append(
            "✔ Suitable for your body type"
        )

    if row["formal_level"] == user_formality:
        reasons.append(
            "✔ Matches activity context"
        )

    if row["style_group"] == user_style:
        reasons.append(
            "✔ Consistent with your style"
        )

    if row["baseColour"] == user_color:
        reasons.append(
            "✔ Matches preferred color"
        )

    if row["visual_similarity"] > 0.75:
        reasons.append(
            "✔ High visual similarity"
        )

    return reasons

# =========================================================
# MAIN PROCESS
# =========================================================

if generate_button:

    with st.spinner(
        "Generating personalized outfits..."
    ):

        filtered_df = products.copy()

        # =================================================
        # FILTERING
        # =================================================

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
        # FALLBACK STRATEGY
        # =================================================

        if len(filtered_df) < 5:

            st.info(
                "Using broader recommendation matching..."
            )

            filtered_df = products.copy()

        # =================================================
        # RULE SCORE
        # =================================================

        filtered_df["score"] = filtered_df.apply(
            calculate_score,
            axis=1
        )

        # =================================================
        # EMBEDDINGS
        # =================================================

        embedding_indices = filtered_df[
            "embedding_idx"
        ].values

        filtered_embeddings = embeddings[
            embedding_indices
        ]

        # =================================================
        # REFERENCE EMBEDDING
        # =================================================

        reference_embedding = np.mean(
            filtered_embeddings,
            axis=0
        )

        # =================================================
        # COSINE SIMILARITY
        # =================================================

        similarities = cosine_similarity(
            filtered_embeddings,
            reference_embedding.reshape(1, -1)
        ).flatten()

        filtered_df[
            "visual_similarity"
        ] = similarities

        # =================================================
        # NORMALIZE SCORE
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

    # =====================================================
    # SUMMARY
    # =====================================================

    st.success(
        f"{len(top_df)} recommendations generated"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Recommendations",
            len(top_df)
        )

    with col2:
        st.metric(
            "Body Type",
            selected_body
        )

    with col3:
        st.metric(
            "Style",
            selected_style
        )

    st.markdown("---")

    st.subheader(
        "✨ Recommended Outfits"
    )

    # =====================================================
    # DISPLAY
    # =====================================================

    cols = st.columns(3)

    for idx, row in enumerate(
        top_df.itertuples()
    ):

        with cols[idx % 3]:

            with st.container(border=True):

                image_path = os.path.join(
                    IMAGE_DIR,
                    row.image_file
                )

                # =========================================
                # IMAGE
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

                # =========================================
                # TITLE
                # =========================================

                st.markdown(
                    f"### {row.articleType}"
                )

                # =========================================
                # BADGE
                # =========================================

                if row.final_score > 0.85:

                    st.success(
                        "🔥 Highly Recommended"
                    )

                elif row.final_score > 0.70:

                    st.info(
                        "✨ Good Match"
                    )

                # =========================================
                # DETAILS
                # =========================================

                st.markdown(
                    f"""
                    **Style:** {row.style_group}  
                    **Color:** {row.baseColour}  
                    **Final Score:** {row.final_score:.2f}  
                    **Similarity:** {row.visual_similarity:.2f}
                    """
                )

                # =========================================
                # EXPLANATION
                # =========================================

                st.markdown(
                    "#### Why Recommended?"
                )

                for reason in row.explanation:

                    st.write(reason)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "VARStyle MVP — Hybrid AI Fashion Recommendation System"
)