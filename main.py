from src.traditional_techniques.load_data import load_hatexplain
from src.traditional_techniques.preprocess import TextPreprocessor

df = load_hatexplain("data/dataset.json")

processor = TextPreprocessor()

df["clean_text"] = processor.transform(df["text"])

print(df[["text", "clean_text"]].head())
