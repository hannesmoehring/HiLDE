import pandas as pd

from src.analysis.dim_reducer import reduce_dimensionality


def main():
    path = "datasets/wine_quality/wine+quality/winequality-red.csv"
    df = pd.read_csv(path, sep=";")
    X = df.to_numpy()

    X_reduced = reduce_dimensionality("PCA", X)
    print(f"Reduced shape: {X_reduced.shape}")


if __name__ == "__main__":
    main()
