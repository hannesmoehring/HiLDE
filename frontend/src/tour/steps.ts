// The guided tour's script: what each step points at, what it says, and what it
// does to the app on the way in. Targets are `[data-tour="…"]` hooks on real UI,
// so a step whose target is not on screen yet (a build still running) waits.
import type { TreeNode } from "../types";

/** What a step may do to the app when it is entered. */
export interface TourContext {
  /** Load the demo run (Wine quality, 2 layers) and open the rail. */
  startDemo: () => void;
  navigate: (path: number[]) => void;
  setNonFeatureOnly: (on: boolean) => void;
  /** The tree on screen, or null while the run is still building. */
  tree: () => TreeNode | null;
}

export interface TourStep {
  id: string;
  /** `data-tour` hook to spotlight; none = a centred card. */
  target?: string;
  title: string;
  body: string;
  /** Shown instead of the card's body while the target is not on screen. */
  waiting?: string;
  enter?: (ctx: TourContext) => void;
  /** Runs once the target is first on screen — for a step whose target appears
   *  only after a build, when `enter` was too early to act on it. */
  shown?: () => void;
  /** Leave the step out (e.g. no cluster splits again, so there is no layer 2). */
  skip?: (ctx: TourContext) => boolean;
}

/** Click a control if it is on screen. A tour step drives the same buttons the
 *  user would, so the components keep owning their own state. */
export function clickTour(selector: string, text?: string) {
  const els = Array.from(document.querySelectorAll<HTMLElement>(selector));
  const el = text ? els.find((e) => e.textContent?.trim().startsWith(text)) : els[0];
  el?.click();
}

/** The first cluster of the root that splits again, so layer 2 has something to show. */
export function splittingChild(tree: TreeNode | null): number | null {
  const i = tree?.children?.findIndex((c) => (c.children?.length ?? 0) > 0) ?? -1;
  return i >= 0 ? i : null;
}

/** Window event the exploration panel answers by selecting a demo group of points,
 *  so the tour can show a predicate without asking the user to lasso first. */
export const TOUR_SELECT_EVENT = "hilde:tour-select";

const openOverview = () => clickTour('.overview__toggle[aria-expanded="false"]');
const sideTab = (layer: number, tab: string) =>
  clickTour(`[data-tour="layer-${layer}-side"] [role="tab"]`, tab);

export const TOUR_STEPS: TourStep[] = [
  {
    id: "welcome",
    title: "Welcome to HiLDE",
    body:
      "A two-minute tour of the interface on the Wine quality dataset with two hierarchical " +
      "layers. Starting it loads that run, replacing what is on screen. Leave any time with " +
      "Exit or Esc, and replay it from the Tour button in the top bar.",
  },
  {
    id: "dataset",
    target: "dataset",
    title: "Pick a dataset",
    body:
      "Everything starts in the configuration rail. Wine quality has 6,497 wines, each " +
      "described by its chemistry.",
    enter: (ctx) => ctx.startDemo(),
  },
  {
    id: "features",
    target: "features",
    title: "Feature columns",
    body:
      "Ticked columns span the space that gets clustered and projected. target_quality is " +
      "left unticked on purpose: it is a label, held out of the feature space, and reported " +
      "next to the results so you can check what the structure lines up with.",
  },
  {
    id: "layers",
    target: "layers",
    title: "Hierarchical layers",
    body:
      "How many times HDBSCAN splits the data again inside each cluster. The tour uses 2: " +
      "clusters at layer 1, and sub-clusters inside them at layer 2.",
  },
  {
    id: "build",
    target: "build",
    title: "Build & Apply",
    body:
      "Runs clustering, a projection per region and the quality scoring. The tour has " +
      "already pressed it for you.",
  },
  {
    id: "overview",
    target: "overview",
    title: "The treemap overview",
    body:
      "The whole hierarchy at a glance: each tile is a cluster, its area the number of " +
      "points, nested tiles its sub-clusters, hatched tiles HDBSCAN noise. Click a tile to " +
      "zoom in, double-click a leaf or use Explore to open it in the layers below.",
    waiting: "Building the run: reducing, clustering and scoring…",
    shown: () => openOverview(),
  },
  {
    id: "colour",
    target: "colour",
    title: "Colour tiles by",
    body:
      "Recolour the map by structure, projection quality (trustworthiness, continuity) or " +
      "any column's mean. Try target_quality to see which clusters hold the better wines.",
    enter: () => openOverview(),
  },
  {
    id: "layer1",
    target: "layer-1",
    title: "Layer 1: the cluster projection",
    body:
      "The root projected to 2D, coloured by cluster. Click a cluster to drill into it; " +
      "scroll to zoom. Grey crosses are points HDBSCAN left unassigned.",
    enter: (ctx) => ctx.navigate([]),
  },
  {
    id: "side",
    target: "layer-1-side",
    title: "Exploring a cluster",
    body:
      "Selecting a cluster fills this column. The tiles score how faithfully its projection " +
      "keeps neighbourhoods. Characteristics show how far each column's mean and spread sit " +
      "from the rest of the layer, in standard deviations.",
    enter: (ctx) => {
      ctx.setNonFeatureOnly(false);
      ctx.navigate([splittingChild(ctx.tree()) ?? 0]);
      sideTab(1, "Characteristics");
    },
  },
  {
    id: "nonfeature",
    target: "nonfeature",
    title: "Only the target columns",
    body:
      "Tick this to hide the feature columns in every characteristics chart and keep only " +
      "the held-out labels. The tour has ticked it: the chart beside the projection now " +
      "shows just target_quality for this cluster.",
    enter: (ctx) => {
      sideTab(1, "Characteristics");
      ctx.setNonFeatureOnly(true);
    },
  },
  {
    id: "predicate",
    target: "layer-1-side",
    title: "Predicates",
    body:
      "The Predicate tab explains the cluster as feature ranges: a short rule like " +
      "\"alcohol ≥ 11 and density < 0.99\", for example, that separates it from the rest of the layer. " +
      "F1 says how well the rule picks out exactly these points.",
    enter: (ctx) => {
      ctx.setNonFeatureOnly(false);
      sideTab(1, "Predicate");
    },
  },
  {
    id: "layer2",
    target: "layer-2",
    title: "Layer 2: inside the cluster",
    body:
      "The selected cluster gets its own projection, fitted only to its points, and splits " +
      "into sub-clusters of its own. Same controls as layer 1, one level down.",
    enter: (ctx) => {
      sideTab(1, "Characteristics");
      const i = splittingChild(ctx.tree());
      if (i !== null) ctx.navigate([i]);
    },
    skip: (ctx) => splittingChild(ctx.tree()) === null,
  },
  {
    id: "explore-layer",
    target: "explore-layer-1",
    title: "Explore an entire layer",
    body:
      "Instead of drilling into one cluster you can take a layer whole. That is also the " +
      "only way to reach the unassigned points, which belong to no cluster.",
  },
  {
    id: "exploration",
    target: "exploration-plot",
    title: "The exploration panel",
    body:
      "Once you reach a leaf, its points open here. Draw a lasso (or a box) around any " +
      "group of points to select them; Pan and the +/− buttons move the view. Try it now.",
    enter: (ctx) => {
      const tree = ctx.tree();
      const i = splittingChild(tree);
      ctx.navigate(i === null ? [0] : [i, 0]);
    },
  },
  {
    id: "exploration-tabs",
    target: "exploration-tabs",
    title: "Explain your selection",
    body:
      "The tour has selected the left third of the points for you. Predicate now explains " +
      "that selection as feature ranges, against the whole dataset or just this cluster. " +
      "Characteristics compares the selection with the cluster. Ranges works the other way " +
      "round: slide column ranges to make a selection.",
    enter: () => {
      clickTour('[data-tour="exploration-tabs"] [role="tab"]', "Predicate");
      window.dispatchEvent(new Event(TOUR_SELECT_EVENT));
    },
  },
  {
    id: "table",
    target: "exploration-table",
    title: "Selected points",
    body:
      "The raw rows behind your selection, target columns fenced off on the right. Export " +
      "them to CSV from here.",
  },
  {
    id: "done",
    target: "tour-button",
    title: "That's the tour",
    body:
      "Replay it any time from this button. The GLOSH outlier table under each layer and " +
      "Cluster movement, which previews moving one cluster onto another, are worth a look " +
      "next.",
  },
];
