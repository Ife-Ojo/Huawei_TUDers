from pathlib import Path
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv(Path(__file__).resolve().parents[2] / "Data" / "train.csv", nrows=20, dtype=str)
print(df)
