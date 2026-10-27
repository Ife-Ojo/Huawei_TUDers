import os
import duckdb

con = duckdb.connect()
con.execute("SET memory_limit='5GB'; SET threads=4; SET preserve_insertion_order=false;")
con.execute("SET temp_directory='duckdb_tmp';")
os.makedirs("shared", exist_ok=True)

for name in ["train", "eval"]:
    print("Converting", name, "...")
    con.execute(f"""
        COPY (SELECT * FROM read_csv('data/{name}.csv', header=true, sample_size=-1))
        TO 'shared/{name}.parquet' (FORMAT parquet, COMPRESSION zstd)
    """)
    n = con.execute(f"SELECT count(*) FROM read_parquet('shared/{name}.parquet')").fetchone()[0]
    print(name, f"{n:,} rows")

print("Done.")