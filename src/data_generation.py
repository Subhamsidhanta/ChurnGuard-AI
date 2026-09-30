"""Canonical data-generation entry point."""
from pathlib import Path
from data.generate_data import generate_customers


def generate_and_save(output_path="data/churn_data.csv", n=20000, seed=42):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_customers(n=n, seed=seed)
    frame.to_csv(path, index=False)
    return frame


if __name__ == "__main__":
    generate_and_save()
