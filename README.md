# SHD Interactive Dimensionality Reduction

This project now includes a local browser UI for interactive 2D dimensionality reduction using the hardcoded red wine quality dataset.

## Features

- Choose reduction method: PCA, t-SNE, UMAP.
- Configure method-specific parameters.
- For PCA: select x/y principal components with defaults based on highest explained variance.
- Brush or lasso points in the scatter plot.
- Access selected points in backend state and export them to disk.

## Run

Install dependencies with your preferred tool (for example `uv sync`), then launch:

```bash
streamlit run ui/app.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Workflow

1. Select method and parameters in the sidebar.
2. Click **Run reduction**.
3. Brush/lasso points in the plot.
4. Review selected rows and click **Export selected points to CSV** when needed.

Exports are written to `outputs/selections/`.
