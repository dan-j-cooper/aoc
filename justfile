default:
    @just --list	

run +args:
    uv run python src/solve.py {{args}}
