
import os
import html
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -------------------- Page --------------------
st.set_page_config(
    page_title="RecipeMatch",
    page_icon="🍽️",
    layout="wide",
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_FILE = os.path.join(ROOT, "models", "best_recommender_model.pkl")
RECIPES_FILE = os.path.join(ROOT, "data", "raw", "RAW_recipes.csv")
TRAIN_FILE = os.path.join(ROOT, "results", "final_train_data.csv")

# -------------------- Style --------------------
st.markdown("""
<style>
.stApp { background:#F6F7FB; }
.block-container { max-width:1400px; padding-top:1.5rem; }
[data-testid="stSidebar"] { background:#172033; }
[data-testid="stSidebar"] * { color:white !important; }

.hero {
    background:linear-gradient(135deg,#172033,#2B3B61);
    color:white; padding:32px; border-radius:24px;
    margin-bottom:22px; box-shadow:0 12px 30px rgba(23,32,51,.14);
}
.hero .eyebrow {
    color:#A7E7D4; font-size:.78rem; font-weight:800;
    letter-spacing:.12em; text-transform:uppercase;
}
.hero h1 { margin:8px 0; font-size:2.25rem; }
.hero p { color:#D8DFEA; max-width:760px; margin:0; }

.kpi {
    background:white; border:1px solid #E7EAF0; border-radius:18px;
    padding:18px; box-shadow:0 6px 18px rgba(23,32,51,.05);
}
.kpi-label { color:#667085; font-size:.8rem; }
.kpi-value { color:#172033; font-size:1.6rem; font-weight:800; margin-top:6px; }

.card {
    background:white; border:1px solid #E7EAF0; border-radius:18px;
    padding:18px; margin-bottom:14px;
    box-shadow:0 6px 18px rgba(23,32,51,.045);
}
.rank {
    display:inline-flex; width:34px; height:34px; border-radius:11px;
    align-items:center; justify-content:center;
    background:#EAF2FF; color:#2864D7; font-weight:800; margin-right:10px;
}
.recipe-title { color:#172033; font-weight:800; }
.muted { color:#667085; font-size:.82rem; }
.pill {
    display:inline-block; background:#F3F5F8; color:#475467;
    border-radius:999px; padding:5px 9px; margin:10px 6px 0 0;
    font-size:.75rem;
}
.score { background:#FFF0E8; color:#B45309; }
.insight {
    background:#EFFAF6; border:1px solid #D4F0E5;
    border-radius:18px; padding:16px 18px; margin-top:8px;
}
.insight b { color:#117A5B; }

.stTabs [data-baseweb="tab-list"] { gap:4px; }
.stTabs [data-baseweb="tab"] { padding:0 14px; }
</style>
""", unsafe_allow_html=True)

# -------------------- Load --------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_csv(path, usecols=None):
    return pd.read_csv(path, usecols=usecols)

required = [MODEL_FILE, RECIPES_FILE, TRAIN_FILE]
if not all(os.path.exists(p) for p in required):
    st.error("Required project files are missing. Check models/, data/raw/, and results/.")
    st.stop()

pkg = load_model(MODEL_FILE)
recipes = load_csv(
    RECIPES_FILE,
    ["id", "name", "minutes", "n_ingredients"]
)
train = load_csv(
    TRAIN_FILE,
    ["user_id", "recipe_id", "date", "rating"]
)

train["user_id"] = pd.to_numeric(train["user_id"], errors="coerce").astype("Int64")
train["recipe_id"] = pd.to_numeric(train["recipe_id"], errors="coerce").astype("Int64")
train["date"] = pd.to_datetime(train["date"], errors="coerce")
recipes["id"] = pd.to_numeric(recipes["id"], errors="coerce").astype("Int64")

indices = np.asarray(pkg["indices"])
similarities = np.asarray(pkg["similarities"])
recipe_to_idx = pkg["recipe_to_index"]
idx_to_recipe = pkg["index_to_recipe"]
neighbors = int(pkg.get("n_neighbors", 100))

# -------------------- Recommendation --------------------
def user_history(user_id):
    return train[train["user_id"] == user_id].sort_values("date", ascending=False)

def recommend(user_id, top_k):
    hist = user_history(user_id)
    seen = set(hist["recipe_id"].dropna().astype(int))
    scores = {}

    for row in hist.itertuples(index=False):
        rid = int(row.recipe_id)
        rating = float(row.rating)

        if rid not in recipe_to_idx:
            continue

        i = recipe_to_idx[rid]
        for n_idx, sim in zip(indices[i][1:], similarities[i][1:]):
            candidate = int(idx_to_recipe[n_idx])
            if candidate not in seen:
                scores[candidate] = scores.get(candidate, 0) + float(sim) * rating

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    out = pd.DataFrame(ranked, columns=["recipe_id", "score"])
    if out.empty:
        return out

    out.insert(0, "rank", range(1, len(out) + 1))
    out = out.merge(
        recipes,
        left_on="recipe_id",
        right_on="id",
        how="left"
    ).drop(columns="id")
    return out

# -------------------- Sidebar --------------------
users = sorted(train["user_id"].dropna().astype(int).unique())

with st.sidebar:
    st.markdown("## 🍽️ RecipeMatch")
    st.caption("Personalized recipe discovery")
    st.markdown("---")

    user_id = st.selectbox("Select User ID", users)
    top_k = st.slider("Recommendations", 5, 20, 10, 5)

    st.markdown("---")
    st.markdown("**Model**")
    st.write("Item-Based Collaborative Filtering")
    st.markdown("**Neighbors**")
    st.write(neighbors)

# -------------------- Header --------------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">Machine Learning Recommendation Demo</div>
    <h1>Find recipes that fit your preferences.</h1>
    <p>
        Recommendations are generated from historical user ratings and
        recipe-to-recipe similarity learned from the interaction data.
    </p>
</div>
""", unsafe_allow_html=True)

hist = user_history(user_id)
recs = recommend(user_id, top_k)

avg_rating = hist["rating"].mean() if not hist.empty else 0
high_rating = (hist["rating"].ge(4).mean() * 100) if not hist.empty else 0

c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Selected User", f"{user_id:,}"),
    (c2, "Historical Ratings", f"{len(hist):,}"),
    (c3, "Average Rating", f"{avg_rating:.2f} / 5"),
    (c4, "High Ratings", f"{high_rating:.0f}%"),
]:
    with col:
        st.markdown(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div></div>',
            unsafe_allow_html=True
        )

st.write("")

tab1, tab2, tab3 = st.tabs(["✨ Recommendations", "📊 User Profile", "🧠 Methodology"])

# -------------------- Recommendations --------------------
with tab1:
    st.subheader("Top Recommended Recipes")
    st.caption(f"Top-{top_k} unseen recipes ranked by similarity-weighted preference.")

    if recs.empty:
        st.warning("No recommendations were generated for this user.")
    else:
        cols = st.columns(2)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 2]:
                name = html.escape(str(row["name"]) if pd.notna(row["name"]) else "Untitled recipe")
                mins = int(row["minutes"]) if pd.notna(row["minutes"]) else None
                ing = int(row["n_ingredients"]) if pd.notna(row["n_ingredients"]) else None

                meta = []
                if mins is not None:
                    meta.append(f"⏱ {mins} min")
                if ing is not None:
                    meta.append(f"🥕 {ing} ingredients")

                pills = "".join(
                    f'<span class="pill">{html.escape(x)}</span>' for x in meta
                )
                pills += f'<span class="pill score">Score {row["score"]:.3f}</span>'

                st.markdown(
                    f"""
                    <div class="card">
                        <div>
                            <span class="rank">#{int(row["rank"])}</span>
                            <span class="recipe-title">{name}</span>
                        </div>
                        <div class="muted">Recipe ID · {int(row["recipe_id"])}</div>
                        <div>{pills}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
        '<div class="insight"><b>Decision-support note</b><br>'
        'Recommendations are ranked suggestions based on historical interaction patterns; '
        'they are not guarantees of user preference.</div>',
        unsafe_allow_html=True
    )

# -------------------- User profile --------------------
with tab2:
    st.subheader("User Rating Profile")

    if hist.empty:
        st.info("No history available.")
    else:
        counts = (
            hist["rating"].value_counts()
            .sort_index()
            .rename_axis("Rating")
            .reset_index(name="Count")
        )

        fig = px.bar(
            counts,
            x="Rating",
            y="Count",
            text="Count",
            title="Historical Rating Distribution"
        )
        fig.update_traces(marker_color="#2864D7")
        fig.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(dtick=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Interactions")
        recent = hist.head(10).merge(
            recipes[["id", "name"]],
            left_on="recipe_id",
            right_on="id",
            how="left"
        )[["date", "recipe_id", "name", "rating"]]

        recent.columns = ["Date", "Recipe ID", "Recipe", "Rating"]
        st.dataframe(recent, use_container_width=True, hide_index=True)

# -------------------- Methodology --------------------
with tab3:
    st.subheader("How the Recommendation Works")

    steps = [
        ("1", "User History", "Read the selected user's previous recipe ratings."),
        ("2", "Recipe Similarity", "Find recipes with similar user-rating patterns."),
        ("3", "Preference Score", "Weight similarity using the user's historical ratings."),
        ("4", "Top-N Ranking", "Rank unseen recipes and return the highest-scoring items."),
    ]

    cols = st.columns(4)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="card"><span class="rank">{num}</span>'
                f'<div style="margin-top:12px"><b>{title}</b></div>'
                f'<div class="muted" style="margin-top:7px">{desc}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("### Model Configuration")
    st.dataframe(
        pd.DataFrame({
            "Component": [
                "Algorithm",
                "Similarity",
                "Neighbors",
                "Output"
            ],
            "Value": [
                "Item-Based Collaborative Filtering",
                "Cosine similarity",
                neighbors,
                f"Top-{top_k} unseen recipes"
            ]
        }),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Built with Python, Streamlit, Pandas, Plotly, scikit-learn, and Item-Based Collaborative Filtering."
)
