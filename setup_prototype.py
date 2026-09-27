"""One-shot setup: create schema -> generate synthetic data -> load DB -> train models.

    python setup_prototype.py
"""
import time

from src import db, model
from src.generate_data import generate


def main():
    t = time.time()
    print(f"Database: {db.backend_name()}")
    db.create_schema()
    tables = generate()
    for k, v in tables.items():
        print(f"  generated {k:18s} {len(v):>7,d} rows")
    db.load_tables(tables)
    print("Loaded into database.")
    bundle = model.train()
    for k, r in bundle["report"].items():
        g, l = r["gradient_boosting"], r["logistic_regression"]
        print(f"  {r['title']:22s} prevalence {r['prevalence']:5.1%} | GB AUC {g['auc']:.3f}, "
              f"recall {g['recall']:.2f}, precision {g['precision']:.2f} | LogReg AUC {l['auc']:.3f}")
    print(f"Done in {time.time() - t:.1f}s. Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
