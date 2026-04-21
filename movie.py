"""
Movie Recommendation System — Linear Algebra Pipeline
UE24MA241B | PES University
Steps: Matrix → RREF → Spaces → Basis → Gram-Schmidt →
       Projection → Least Squares → Eigenvalues → Diagonalization
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────
RATINGS_PATH = r"C:\Users\nagav\OneDrive\Desktop\laa_orange\laa_orange\ml-1m\ratings.dat"
MOVIES_PATH  = r"C:\Users\nagav\OneDrive\Desktop\laa_orange\laa_orange\ml-1m\movies.dat"
MIN_RATINGS  = 100
TOP_N        = 5
LATENT_K     = 5
CF_WEIGHT    = 0.6
CB_WEIGHT    = 0.4

# ── DATA LOADING ──────────────────────────────────────────────
def load_data():
    try:
        ratings = pd.read_csv(
            RATINGS_PATH, sep="::", engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"]
        )
        movies = pd.read_csv(
            MOVIES_PATH, sep="::", engine="python",
            names=["movie_id", "title", "genres"], encoding="latin-1"
        )
        movies["genres"] = movies["genres"].str.split("|")
        print(f"[Data] MovieLens 1M loaded — {len(ratings)} ratings, {len(movies)} movies")

        # Filter movies with enough ratings
        counts  = ratings["movie_id"].value_counts()
        valid   = counts[counts >= MIN_RATINGS].index
        ratings = ratings[ratings["movie_id"].isin(valid)]
        movies  = movies[movies["movie_id"].isin(valid)].reset_index(drop=True)
        print(f"[Data] After filter: {len(movies)} movies (each >= {MIN_RATINGS} ratings)")

    except FileNotFoundError:
        print("[Data] MovieLens files not found — using synthetic data instead.")
        ratings, movies = _synthetic_data()

    return ratings, movies


def _synthetic_data(n_users=80, n_movies=40):
    genres_pool = ["Action","Comedy","Drama","Sci-Fi","Romance","Thriller","Horror"]
    movies = pd.DataFrame({
        "movie_id": range(1, n_movies+1),
        "title"  : [f"Movie_{i}" for i in range(1, n_movies+1)],
        "genres" : [list(np.random.choice(genres_pool, size=np.random.randint(1,3),
                    replace=False)) for _ in range(n_movies)],
    })
    rows = []
    for u in range(1, n_users+1):
        for m in np.random.choice(n_movies, size=max(5, n_movies//6), replace=False):
            rows.append({"user_id": u, "movie_id": int(m)+1,
                         "rating": float(np.random.randint(1, 6))})
    return pd.DataFrame(rows), movies


# ── STEP 1 : USER-MOVIE RATING MATRIX ────────────────────────
def build_rating_matrix(ratings):
    R = ratings.pivot_table(index="user_id", columns="movie_id",
                            values="rating").fillna(0.0)
    sparsity = 100.0 * (R.values == 0).mean()
    print(f"[Step 1] Rating matrix shape={R.shape}  sparsity={sparsity:.1f}%")
    return R


# ── STEP 2 : GAUSSIAN ELIMINATION / RREF ─────────────────────
def rref_rank(A):
    M = A[:min(50, A.shape[0]), :min(50, A.shape[1])].astype(float).copy()
    rows, cols, pivot = M.shape[0], M.shape[1], 0
    for col in range(cols):
        if pivot >= rows:
            break
        idx = np.argmax(np.abs(M[pivot:, col])) + pivot
        if abs(M[idx, col]) < 1e-10:
            continue
        M[[pivot, idx]] = M[[idx, pivot]]
        M[pivot] /= M[pivot, col]
        for r in range(rows):
            if r != pivot:
                M[r] -= M[r, col] * M[pivot]
        pivot += 1
    rank = int(np.sum(np.any(np.abs(M) > 1e-10, axis=1)))
    print(f"[Step 2] RREF — numerical rank={rank}, null-space dim={cols - rank}")
    return rank


# ── STEP 3 : VECTOR SPACES (Row / Column / Null Space) ────────
def analyse_spaces(R):
    U, s, Vt = np.linalg.svd(R, full_matrices=False)
    rank   = int(np.sum(s > 1e-6))
    null_d = R.shape[1] - rank
    print(f"[Step 3] Row-space dim={rank} | Col-space dim={rank} | Null-space dim={null_d}")
    return rank


# ── STEP 4 : BASIS (Linear Independence) ─────────────────────
def independent_basis(R, rank):
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    basis = Vt[:rank]
    print(f"[Step 4] Independent basis: {basis.shape[0]} vectors of dim {basis.shape[1]}")
    return basis


# ── STEP 5 : GRAM-SCHMIDT ORTHOGONALISATION ───────────────────
def gram_schmidt(A):
    Q = []
    for v in A:
        u = v.astype(float).copy()
        for q in Q:
            u -= np.dot(q, u) * q
        n = np.linalg.norm(u)
        if n > 1e-10:
            Q.append(u / n)
    Q = np.array(Q)
    err = np.abs(Q @ Q.T - np.eye(len(Q))).max()
    print(f"[Step 5] Gram-Schmidt: {len(Q)} orthonormal vectors | orthogonality error={err:.2e}")
    return Q


# ── STEP 6 : PROJECTION ───────────────────────────────────────
def project_user(user_vec, Q):
    """Project user rating vector onto orthonormal subspace Q (Step 6)."""
    return sum(np.dot(q, user_vec) * q for q in Q)


# ── STEP 7 : LEAST SQUARES PREDICTION ────────────────────────
def least_squares_predict(R_df, user_id, Q):
    if user_id not in R_df.index:
        return pd.Series(R_df.mean())

    row  = R_df.loc[user_id].values
    mask = row > 0
    y    = row[mask]
    X    = Q[:, mask].T          # (rated_movies x k)

    if len(y) < 2:
        return pd.Series(R_df.mean())

    # x̂ = (AᵀA)⁻¹Aᵀb
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = Q.T @ beta
    print(f"[Step 7] Least-squares prediction fitted for user {user_id}")
    return pd.Series(pred, index=R_df.columns)


# ── STEP 8 : EIGENVALUE ANALYSIS ─────────────────────────────
def eigenvalue_analysis(R):
    C       = np.cov(R.T)
    eigvals = np.linalg.eigvalsh(C)[::-1]
    total   = eigvals.sum()
    print(f"[Step 8] Top-5 eigenvalues : {np.round(eigvals[:5], 2)}")
    print(f"         Variance by top-3 : {100*eigvals[:3].sum()/total:.1f}%")
    return eigvals


# ── STEP 9 : DIAGONALIZATION ──────────────────────────────────
def diagonalize(R, k=LATENT_K):
    C                = np.cov(R.T)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    D   = np.diag(eigvals[idx[:k]])   # k×k diagonal matrix
    P   = eigvecs[:, idx[:k]]         # columns = eigenvectors
    print(f"[Step 9] Diagonalized covariance — kept top {k} principal directions")
    return P, D


# ── CONTENT-BASED SCORE ───────────────────────────────────────
def content_score(user_id, R_df, movies_df, pref_genres):
    mlb  = MultiLabelBinarizer()
    gmat = mlb.fit_transform(movies_df["genres"].tolist()).astype(float)
    pref = np.zeros(gmat.shape[1])
    classes = list(mlb.classes_)
    for g in pref_genres:
        if g in classes:
            pref[classes.index(g)] = 1.0
    genre_sim = gmat @ pref / (np.linalg.norm(pref) + 1e-9)
    return pd.Series(genre_sim, index=movies_df["movie_id"].values)


# ── HYBRID RECOMMENDER ────────────────────────────────────────
def recommend(user_id, R_df, movies_df, Q, pref_genres):
    ls = least_squares_predict(R_df, user_id, Q)
    cb = content_score(user_id, R_df, movies_df, pref_genres)
    cb = cb.reindex(R_df.columns).fillna(0)

    def norm(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo + 1e-9)

    score = CF_WEIGHT * norm(ls) + CB_WEIGHT * norm(cb)
    rated = set(R_df.columns[R_df.loc[user_id] > 0]) if user_id in R_df.index else set()

    result = movies_df.copy()
    result["score"] = result["movie_id"].map(score).fillna(0)
    result = result[~result["movie_id"].isin(rated)]
    top = result.nlargest(TOP_N, "score")[["movie_id","title","genres","score"]]
    top = top.reset_index(drop=True)
    top.index += 1
    return top


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Movie Recommendation System — Linear Algebra Pipeline")
    print("  UE24MA241B | PES University")
    print("=" * 60)

    # Load data
    ratings_df, movies_df = load_data()

    # Step 1 — Rating matrix
    R_df = build_rating_matrix(ratings_df)
    R    = R_df.values

    # Steps 2–5 — Matrix analysis pipeline
    rref_rank(R)
    rank  = analyse_spaces(R)
    basis = independent_basis(R, rank)
    Q     = gram_schmidt(basis[:min(LATENT_K, len(basis))])

    # Steps 8–9 — Eigenvalue & diagonalization
    eigenvalue_analysis(R)
    P, D = diagonalize(R)

    # Interactive recommendation loop
    valid_ids  = sorted(R_df.index.tolist())
    all_genres = sorted({g for gs in movies_df["genres"] for g in gs})

    print(f"\nAvailable user IDs : 1 to {max(valid_ids)}")
    print(f"Available genres   : {', '.join(all_genres)}")

    while True:
        print("\n" + "-" * 60)
        try:
            uid = int(input("Enter user ID (0 to quit): ").strip())
        except ValueError:
            print("Please enter a valid number.")
            continue

        if uid == 0:
            break

        if uid not in valid_ids:
            print(f"User {uid} not found. Try a number between 1 and {max(valid_ids)}.")
            continue

        raw  = input("Preferred genres (comma-separated, e.g. Action,Sci-Fi): ").strip()
        pref = [g.strip() for g in raw.split(",") if g.strip() in all_genres]
        if not pref:
            print("No valid genres entered — defaulting to Action, Drama.")
            pref = ["Action", "Drama"]

        print(f"\nTop {TOP_N} recommendations for user {uid} (genres: {', '.join(pref)}):")
        recs = recommend(uid, R_df, movies_df, Q, pref)
        for rank_i, row in recs.iterrows():
            g = ", ".join(row["genres"]) if isinstance(row["genres"], list) else row["genres"]
            print(f"  {rank_i}. {row['title']:<45} [{g}]  score={row['score']:.3f}")

    print("\nDone. Enjoy your movies!")


if __name__ == "__main__":
    main()