"""FastAPI app for serving music recommendations."""

from fastapi import FastAPI, HTTPException

from music_recommender.recommend import (
    get_similar_artists,
    load_recommender_artifacts,
    recommend_artists_for_user,
)

app = FastAPI(title="Music Recommendation System API")


@app.get("/")
def root() -> dict[str, str]:
    """Return a simple API health message."""
    return {"message": "Music Recommendation System API"}


@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: str, top_k: int = 10) -> dict[str, object]:
    """Return artist recommendations for a user."""
    try:
        model, user_item_matrix, mappings = load_recommender_artifacts()
        recommendations = recommend_artists_for_user(
            model=model,
            user_id=user_id,
            user_item_matrix=user_item_matrix,
            mappings=mappings,
            top_k=top_k,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model first.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {"user_id": user_id, "recommendations": recommendations}


@app.get("/similar-artists/{artist_id}")
def similar_artists(artist_id: str, top_k: int = 10) -> dict[str, object]:
    """Return artists similar to a selected artist."""
    try:
        model, _, mappings = load_recommender_artifacts()
        recommendations = get_similar_artists(
            model=model,
            artist_id=artist_id,
            mappings=mappings,
            top_k=top_k,
        )
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model first.",
        ) from error
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {"artist_id": artist_id, "similar_artists": recommendations}
