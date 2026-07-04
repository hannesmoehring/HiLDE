// Hierarchical + exploration config controls. Parity with src/ui/components/config.py.
import type { AnalysisConfig, DRMethod } from "../types";

interface Props {
  config: AnalysisConfig;
  maxDims: number;
  onChange: (patch: Partial<AnalysisConfig>) => void;
}

function NumberField(props: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="field">
      <span>{props.label}</span>
      <input
        type="number"
        value={props.value}
        min={props.min}
        max={props.max}
        step={props.step ?? 1}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

function CheckField(props: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="field field--check">
      <input type="checkbox" checked={props.value} onChange={(e) => props.onChange(e.target.checked)} />
      <span>{props.label}</span>
    </label>
  );
}

export function ConfigPanel({ config, maxDims, onChange }: Props) {
  return (
    <div className="config-grid">
      <section className="config-section">
        <h3>Hierarchical clustering</h3>
        <CheckField
          label="Normalize (StandardScaler)"
          value={config.hclust_normalize}
          onChange={(v) => onChange({ hclust_normalize: v, normalize: v })}
        />
        <NumberField
          label="Hierarchical layers"
          value={config.hierarchical_layers}
          min={0}
          max={5}
          onChange={(v) => onChange({ hierarchical_layers: v })}
        />
        {config.hierarchical_layers === 0 && (
          <p className="hint">0 = skip clustering; reduce the whole dataset into a single exploration layer.</p>
        )}
        <NumberField
          label="UMAP pre-reduction components"
          value={config.hclust_umap_n_components}
          min={2}
          max={maxDims}
          onChange={(v) => onChange({ hclust_umap_n_components: v })}
        />
        <NumberField
          label="HDBSCAN min samples"
          value={config.hclust_min_samples}
          min={1}
          onChange={(v) => onChange({ hclust_min_samples: v })}
        />
        <NumberField
          label="HDBSCAN min cluster size"
          value={config.hclust_min_cluster_size}
          min={2}
          onChange={(v) => onChange({ hclust_min_cluster_size: v })}
        />
      </section>

      <section className="config-section">
        <h3>Exploration embedding</h3>
        <label className="field">
          <span>Method</span>
          <select
            value={config.method}
            onChange={(e) => onChange({ method: e.target.value as DRMethod })}
          >
            <option value="PCA">PCA</option>
            <option value="t-SNE">t-SNE</option>
            <option value="UMAP">UMAP</option>
          </select>
        </label>

        {config.method === "t-SNE" && (
          <>
            <NumberField
              label="Perplexity"
              value={config.tsne_perplexity}
              min={1}
              step={1}
              onChange={(v) => onChange({ tsne_perplexity: v })}
            />
            <NumberField
              label="Learning rate"
              value={config.tsne_learning_rate}
              onChange={(v) => onChange({ tsne_learning_rate: v })}
            />
            <NumberField
              label="Random state"
              value={config.tsne_random_state}
              onChange={(v) => onChange({ tsne_random_state: v })}
            />
          </>
        )}

        {config.method === "UMAP" && (
          <>
            <NumberField
              label="n_neighbors"
              value={config.umap_n_neighbors}
              min={2}
              onChange={(v) => onChange({ umap_n_neighbors: v })}
            />
            <NumberField
              label="min_dist"
              value={config.umap_min_dist}
              min={0}
              max={1}
              step={0.05}
              onChange={(v) => onChange({ umap_min_dist: v })}
            />
            <NumberField
              label="Random state"
              value={config.umap_random_state}
              onChange={(v) => onChange({ umap_random_state: v })}
            />
          </>
        )}
      </section>
    </div>
  );
}
