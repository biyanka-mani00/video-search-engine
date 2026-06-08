# Project Instructions: Video Search Engine

## Philosophy
This project is built for learning AI/ML and Gen AI. Every module MUST be documented with:
- **What** it does.
- **How** it works (the underlying logic).
- **Why** we chose this specific approach/tool.

## Architectural Conventions
- **Ingestion:** Follows a pipeline of `Video -> FFmpeg (Keyframes) -> CLIP (Embeddings) -> Qdrant`.
- **Search:** `Text -> CLIP (Embedding) -> Qdrant (Vector Search)`.
- **Metadata:** All video metadata and frame timestamps are stored in PostgreSQL.
- **Asynchrony:** Use Celery for the ingestion pipeline as it is computationally expensive.

## Code Standards
- Use Python type hints throughout.
- Use `Pydantic` for data validation in the API.
- Use `SQLAlchemy` for PostgreSQL interaction.
- Document every file with a module-level docstring.
