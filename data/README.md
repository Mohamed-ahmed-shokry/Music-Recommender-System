# Data

Place raw listening interaction and artist metadata CSV files in `data/raw/`.

Generated or transformed datasets belong in `data/processed/`.

## Interaction contract

Interaction files require these columns:

| Column | Description |
| --- | --- |
| `user_id` | Non-empty listener identifier, loaded as text |
| `artist_id` | Non-empty artist identifier, loaded as text |
| `artist_name` | Stable display name for the artist ID |
| `play_count` | Positive, finite numeric implicit-feedback signal |

Ingestion trims identifier and name whitespace. Repeated rows for the same
`user_id` and `artist_id` are aggregated by summing `play_count`, so raw event
exports and pre-aggregated listening tables are both accepted. One `artist_id`
must not map to multiple names.

## Metadata contract

Artist metadata requires `artist_id`, `artist_name`, `genres`, `mood_tags`,
`country`, and `era`. Genres and moods are semicolon-delimited. Artist IDs must
be unique, and metadata must cover every artist retained from the interaction
file.

The included `sample_interactions.csv` and `sample_artist_metadata.csv` are
small deterministic fixtures intended for local demos and automated tests.
