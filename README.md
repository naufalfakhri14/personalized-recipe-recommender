# Personalized Recipe Recommendation Using Collaborative Filtering

A machine learning project that builds a personalized recipe recommendation system from historical user–recipe ratings. The project compares a popularity-based baseline, SVD-based matrix factorization, and item-based collaborative filtering, then evaluates the final recommender using Top-N ranking metrics.

## Project Overview

Online recipe platforms provide users with a large number of recipes, making personalized discovery challenging. Users also have different tastes and cooking preferences, so recommending only globally popular recipes may not be sufficient.

This project addresses the following question:

> **How can historical user–recipe interactions be used to recommend relevant recipes to individual users?**

The project is formulated as a **recommendation and ranking problem**, with collaborative filtering as the primary machine learning approach.

### Objective

Build a recommendation system that:

- learns preference patterns from historical user–recipe ratings;
- recommends previously unseen recipes to a user;
- compares personalized collaborative filtering with a simple popularity baseline; and
- evaluates recommendation quality using ranking-oriented metrics.

---

## Dataset

The project uses the **Food.com Recipes and Interactions** dataset, originally collected from Food.com (formerly GeniusKitchen). The dataset contains recipe metadata and historical user interactions and has been used in research on personalized recipe generation. The public dataset is available through Kaggle. [1][2]

### Files Used

#### `interactions.csv`

Contains user–recipe interaction records:

- `user_id` — unique identifier of the user
- `recipe_id` — unique identifier of the recipe
- `date` — date of the interaction
- `rating` — user rating for the recipe
- `review` — written review text

#### `RAW_recipes.csv`

Contains recipe-level metadata:

- `name` — recipe name
- `id` — unique recipe identifier
- `minutes` — preparation time in minutes
- `contributor_id` — identifier of the recipe contributor
- `submitted` — recipe submission date
- `tags` — recipe tags or categories
- `nutrition` — nutritional information
- `n_steps` — number of recipe steps
- `steps` — recipe preparation steps
- `description` — recipe description
- `ingredients` — recipe ingredients
- `n_ingredients` — number of ingredients

### Initial Dataset Size

The loaded raw files contained:

| Dataset | Records | Features |
|---|---:|---:|
| Interactions | 731,927 | 5 |
| Recipes | 83,782 | 12 |

The interaction data contained **188,834 unique users** and **185,913 unique recipe IDs** before cleaning.

---

## Methodology

The project follows the pipeline below:

```text
Raw Data
   ↓
Data Quality Assessment
   ↓
Data Cleaning
   ↓
Metadata Matching
   ↓
5-Core Filtering
   ↓
Exploratory Data Analysis
   ↓
Train / Validation / Test Split
   ↓
Popularity Baseline
   ↓
SVD Collaborative Filtering
   ↓
Item-Based Collaborative Filtering
   ↓
Hyperparameter Tuning
   ↓
Top-N Recommendation
   ↓
Validation & Test Evaluation
```

---

## Data Preprocessing

### 1. Data Quality Assessment

The datasets were inspected for:

- missing values;
- duplicate records;
- data types;
- rating values;
- number of users and recipes; and
- interaction frequency.

No duplicate rows were found in either dataset.

### 2. Rating Validation

The interaction data contained rating values from 0 to 5. A rating of `0` does not represent an actual star rating in this dataset; it indicates that no rating was provided. Therefore, interactions with `rating = 0` were removed from the collaborative filtering data. [1]

```text
Original interactions : 731,927
Rating = 0 removed    : 51,832
Valid interactions   : 680,095
```

Valid ratings retained:

```text
1, 2, 3, 4, 5
```

### 3. Date Processing

The `date` field was converted from string format to datetime format for chronological analysis and temporal train/validation/test splitting.

### 4. Missing Review Values

There were 169 missing values in the `review` column. These records were retained because the review text is not used as a modeling feature in the collaborative filtering approach. The model uses `user_id`, `recipe_id`, and `rating` as its core interaction signals.

### 5. Recipe Metadata Matching

Interaction recipe IDs were matched against recipe IDs in `RAW_recipes.csv` so that recommended recipe IDs could be linked to human-readable recipe names and metadata.

### 6. 5-Core Filtering

The interaction data contained many users and recipes with very few interactions. To reduce sparsity and retain more useful preference histories, an iterative **5-core filtering** strategy was applied:

- each user must have at least 5 valid interactions;
- each recipe must have at least 5 interactions; and
- filtering is repeated until the constraints are stable.

After metadata matching and the final 5-core filtering, the modeling dataset contained:

| Metric | Final Value |
|---|---:|
| Interactions | **51,083** |
| Users | **1,184** |
| Recipes | **6,560** |
| Minimum interactions per user | **5** |
| Minimum interactions per recipe | **5** |

---

## Exploratory Data Analysis

### Rating Distribution

After filtering, ratings remained strongly concentrated toward the upper end:

| Rating | Count | Percentage |
|---:|---:|---:|
| 1 | 1,076 | 0.42% |
| 2 | 2,269 | 0.88% |
| 3 | 7,944 | 3.10% |
| 4 | 39,443 | 15.38% |
| 5 | 205,788 | 80.22% |

**Insight:** 80.22% of observed interactions received a rating of 5, indicating a strong positive-rating bias in the dataset.

### User Interaction Behavior

| Metric | Value |
|---|---:|
| Users | 10,084 |
| Average interactions per user | 25.44 |
| Median interactions per user | 10 |
| Minimum interactions | 5 |
| Maximum interactions | 1,880 |

**Insight:** User activity is highly skewed. Most users have relatively few interactions, while a smaller number of users contribute substantially more ratings.

### Recipe Interaction Behavior

| Metric | Value |
|---|---:|
| Recipes | 21,590 |
| Average interactions per recipe | 11.88 |
| Median interactions per recipe | 7 |
| Minimum interactions | 5 |
| Maximum interactions | 1,267 |

**Insight:** Recipe popularity is also highly uneven, with most recipes receiving relatively few interactions and a small group attracting substantially more feedback.

### Interaction Matrix Sparsity

The filtered user–recipe interaction matrix initially had a shape of **10,084 × 21,590** with approximately **99.8822% sparsity**.

This indicates that most possible user–recipe combinations do not contain an observed rating, motivating the use of collaborative filtering to infer hidden preference patterns.

### Popular Recipes

The most interacted recipes in the final training data included:

| Rank | Recipe | Interactions | Average Rating |
|---:|---|---:|---:|
| 1 | Potato Squashers | 54 | 4.87 |
| 2 | Snickerdoodle French Toast | 48 | 4.85 |
| 3 | Yummy Cheesy Corn | 45 | 4.56 |
| 4 | Lemon Pepper Fish Greek Style | 43 | 4.84 |
| 5 | Deb's Favorite Way to Eat Fresh Fruit | 43 | 4.81 |
| 6 | Baked Maple Oatmeal | 40 | 4.65 |
| 7 | Oven Roasted Potatoes with Garlic and Rosemary | 39 | 4.92 |
| 8 | Strawberry Sweetheart Streusel Muffins | 36 | 4.89 |
| 9 | Peppered Buffalo Ranch Shrimp Pizza | 35 | 5.00 |
| 10 | Our Daily Bread in a Crock – Weekly Make and Bake | 35 | 4.97 |

---

## Train / Validation / Test Split

A per-user temporal holdout strategy was used rather than a standard random split.

For each user:

- earlier interactions → **training**;
- second-latest interaction → **validation**; and
- latest interaction → **test**.

The split produced:

| Dataset | Interactions | Users |
|---|---:|---:|
| Training | 48,715 | 1,184 |
| Validation | 1,184 | 1,184 |
| Testing | 1,184 | 1,184 |

This setup simulates a realistic scenario in which the system learns from previous behavior and attempts to recommend a user's future preferences.

---

## Modeling

### 1. Popularity-Based Recommendation

A popularity-based baseline recommends recipes with the highest historical interaction counts from the training data while excluding recipes already seen by the selected user.

**Purpose:** establish a simple benchmark for personalized recommendation.

### 2. SVD Collaborative Filtering

A Singular Value Decomposition (SVD) model was used as a matrix factorization approach. The model learns latent user and recipe preference representations from historical ratings and estimates predicted ratings for user–recipe pairs.

Initial configuration:

```text
n_factors = 50
n_epochs  = 20
lr_all    = 0.005
reg_all   = 0.02
```

Validation performance of the initial SVD model:

| Metric | Score |
|---|---:|
| RMSE | 0.4930 |
| MAE | 0.3051 |

### 3. Item-Based Collaborative Filtering

An item-based collaborative filtering model was developed using cosine similarity and nearest-neighbor search. For each recipe a user has previously rated, the model identifies similar recipes and aggregates similarity-weighted scores to rank unseen recipes.

Conceptually:

```text
User Rating History
        ↓
Similar Recipes
        ↓
Similarity-Weighted Scoring
        ↓
Rank Unseen Recipes
        ↓
Top-N Recommendations
```

---

## Hyperparameter Tuning

### SVD Tuning

Grid Search with 3-fold cross-validation was used to test:

| Hyperparameter | Values Tested |
|---|---|
| `n_factors` | 50, 100 |
| `n_epochs` | 20, 30 |
| `lr_all` | 0.005, 0.01 |
| `reg_all` | 0.02, 0.05 |

Best cross-validation configuration:

```text
n_factors = 50
n_epochs  = 30
lr_all    = 0.005
reg_all   = 0.05
```

Best 3-fold cross-validation result:

```text
RMSE = 0.4086
MAE  = 0.2490
```

On the held-out validation set, the tuned configuration produced:

```text
RMSE = 0.4911
MAE  = 0.2998
```

The difference between cross-validation and held-out validation performance illustrates that lower cross-validation error does not necessarily translate into a large improvement on unseen validation interactions.

### Item-Based CF Tuning

The number of neighboring recipes was tuned directly against Top-N recommendation metrics:

| `n_neighbors` | Precision@10 | Recall@10 | NDCG@10 |
|---:|---:|---:|---:|
| 20 | 0.0022 | 0.0218 | 0.0129 |
| 50 | 0.0025 | 0.0253 | 0.0149 |
| **100** | **0.0029** | **0.0288** | **0.0169** |

The selected configuration was:

```text
n_neighbors = 100
```

---

## Resampling

**Resampling was not applied.**

This project is formulated as a recommendation and ranking task rather than an imbalanced classification problem. Techniques such as SMOTE are therefore not appropriate for the core modeling objective.

---

## Evaluation

Because the main objective is Top-N recommendation, the evaluation focuses on ranking quality rather than classification accuracy.

### Metrics

- **Precision@10** — proportion of the Top-10 recommendations that are relevant.
- **Recall@10** — proportion of relevant held-out items retrieved within the Top-10 list.
- **NDCG@10** — considers both whether the relevant item was retrieved and where it appeared in the ranking.

For this evaluation, an interaction with **rating ≥ 4** was treated as relevant.

### Validation Model Comparison

| Model | Precision@10 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|
| Popularity Baseline | 0.0007 | 0.0070 | 0.0032 |
| Tuned SVD | 0.0002 | 0.0017 | 0.0005 |
| **Item-Based CF** | **0.0025** | **0.0253** | **0.0149** |

On the validation set, Item-Based Collaborative Filtering produced the highest value for all three Top-N ranking metrics among the evaluated approaches.

### Final Test Performance

After selecting `n_neighbors = 100` using the validation results, the model was retrained on the combined training and validation interactions and evaluated on unseen test data.

| Metric | Test Score |
|---|---:|
| Precision@10 | **0.0021** |
| Recall@10 | **0.0211** |
| NDCG@10 | **0.0109** |

### Interpretation

The final model demonstrates the feasibility of personalized recipe recommendation from collaborative signals, but the low Top-N scores indicate that the current system has substantial room for improvement. In particular, the extremely sparse interaction matrix, limited interaction history for many users, and use of explicit ratings alone make personalization challenging.

---

## Final Recommendation Example

For a selected user, the final system produces a ranked list of previously unseen recipes based on item similarity and the user's historical ratings.

```text
Selected User
      ↓
Historical Recipe Ratings
      ↓
Item-Based Collaborative Filtering
      ↓
Similarity-Weighted Scores
      ↓
Top-10 Unseen Recipes
```

The recipe IDs are joined with `RAW_recipes.csv` so the final output can display human-readable recipe names.

---

## Project Structure

```text
personalized-recipe-recommender/
│
├── data/
│   ├── raw/
│   │   ├── interactions.csv
│   │   └── RAW_recipes.csv
│   │
│   └── processed/
│
├── notebooks/
│   └── recipe_recommender.ipynb
│
├── models/
│   └── best_recommender_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── recommendation_results.csv
│   └── evaluation_results.csv
│
├── dashboard/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- scikit-surprise
- Joblib

### Main Techniques

- Data cleaning
- Exploratory data analysis
- 5-core filtering
- Temporal per-user holdout
- Popularity-based recommendation
- Matrix Factorization with SVD
- Item-Based Collaborative Filtering
- Cosine similarity
- Hyperparameter tuning
- Top-N recommendation evaluation

---

## Installation

Create a Python environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

A typical `requirements.txt` should include:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
scikit-surprise
joblib
streamlit
```

---

## How to Run the Project

1. Place the raw datasets in:

```text
data/raw/
```

2. Open:

```text
notebooks/recipe_recommender.ipynb
```

3. Run the notebook from data loading through final evaluation.

4. The notebook produces the cleaned interaction data, trained recommendation model, evaluation results, and recommendation outputs.

5. The optional Streamlit interface can be launched with:

```bash
python -m streamlit run dashboard/app.py
```

---

## Limitations

### 1. Sparse Interaction Data

Even after 5-core filtering, the user–recipe interaction space remains sparse. Limited historical information makes personalization difficult for many users.

### 2. Explicit Rating Dependence

The current model primarily relies on explicit ratings. Other useful signals such as views, clicks, saves, favorites, or repeated recipe access are not available in the current modeling setup.

### 3. Cold-Start Problem

New users and new recipes with little or no historical interaction data are difficult to recommend using collaborative filtering alone.

### 4. Ranking Evaluation

The current evaluation uses one held-out interaction per user and treats ratings ≥ 4 as relevant. This is useful for a compact offline evaluation, but broader recommendation evaluation could use multiple held-out relevant items and multiple values of K.

### 5. Collaborative Filtering Only

The final model does not directly incorporate recipe content such as ingredients, tags, nutrition, cooking time, or description. This limits its ability to capture content-based similarity and address cold-start cases.

---

## Future Improvements

A stronger version of the system could explore:

1. **Hybrid Recommendation** — combine collaborative filtering with recipe metadata such as ingredients, tags, nutrition, and preparation time.
2. **Implicit Feedback** — incorporate clicks, views, saves, favorites, and repeated interactions.
3. **Cold-Start Handling** — use content-based recommendations when user or recipe interaction history is limited.
4. **Ranking-Oriented Training** — optimize directly for Top-N recommendation quality rather than relying mainly on rating prediction.
5. **Larger Evaluation Protocol** — evaluate multiple held-out relevant recipes per user and compare several values of K.
6. **Model Serving** — deploy the final recommender through an interactive Streamlit application or API.

---

## Key Takeaways

- The project transformed raw Food.com interaction data into a usable collaborative filtering dataset.
- Rating 0 interactions were removed because they represent missing ratings rather than valid zero-star preferences.
- Iterative 5-core filtering reduced sparse users and recipes.
- The final modeling dataset contained **51,083 interactions, 1,184 users, and 6,560 recipes**.
- Item-Based Collaborative Filtering with **100 neighbors** achieved the strongest validation Top-N results among the evaluated approaches.
- Final unseen-test performance was **Precision@10 = 0.0021, Recall@10 = 0.0211, and NDCG@10 = 0.0109**.
- The results demonstrate the feasibility of collaborative recommendation while highlighting the need for additional signals and hybrid methods to improve personalization.

---

## References

[1] Shuyang Li, Bodhisattwa Prasad Majumder, Jianmo Ni, and Julian McAuley, **Food.com Recipes and Interactions**, Kaggle. The dataset contains recipe data and user interactions collected from Food.com and is associated with the paper *Generating Personalized Recipes from Historical User Preferences* (EMNLP 2019).  
https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions

[2] Bodhisattwa Prasad Majumder, Shuyang Li, Jianmo Ni, and Julian McAuley, **Generating Personalized Recipes from Historical User Preferences**, EMNLP 2019.  
https://aclanthology.org/D19-1613/

[3] Dataset background and file structure reference for the Food.com recipe and interaction files.  
https://practicaldsc.org/wn25/final-project/datasets/recipes-and-ratings/
