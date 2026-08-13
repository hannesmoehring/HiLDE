import pandas as pd

CSV = "/Users/hannesmoehring/Documents/University/SEM8/SHD/dev/SHD/datasets/QM9/qm9.csv"
df = pd.read_csv(CSV, usecols=["smiles"])
s = df["smiles"]
print("rows:", len(s))

# digits appearing inside square brackets are NOT ring-closure markers
inside = s.str.contains(r"\[[^\]]*[1-9][^\]]*\]", regex=True)
print("SMILES with a digit inside [...] :", int(inside.sum()))
print(s[inside].head(20).tolist())

cnt = s.str.count(r"[1-9]")
odd = cnt % 2 == 1
print("SMILES with ODD total digit count:", int(odd.sum()))
print(s[odd].head(20).tolist())

# repo's formula
rings_repo = cnt // 2
print("repo ring distribution:", rings_repo.value_counts().sort_index().to_dict())

# corrected: strip bracket atoms first, then count
stripped = s.str.replace(r"\[[^\]]*\]", "A", regex=True)
rings_fix = stripped.str.count(r"[1-9]") // 2
print("corrected ring distribution:", rings_fix.value_counts().sort_index().to_dict())

mism = rings_repo != rings_fix
print("MOLECULES WHOSE LABEL DIFFERS:", int(mism.sum()), f"({100*mism.mean():.3f}% of {len(s)})")
if mism.any():
    ex = pd.DataFrame({"smiles": s[mism], "repo": rings_repo[mism], "correct": rings_fix[mism]})
    print(ex.head(20).to_string())
