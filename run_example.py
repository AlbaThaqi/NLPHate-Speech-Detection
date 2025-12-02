"""Small script to verify repository layout"""
import os

REQUIRED = [
    "docs", "data/raw", "data/processed", "src", "notebooks", "experiments"
]

if __name__ == "__main__":
    missing = []
    for p in REQUIRED:
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        print("Missing paths:", missing)
    else:
        print("Repository layout looks OK. Directories present:")
        for p in REQUIRED:
            print(" -", p)
