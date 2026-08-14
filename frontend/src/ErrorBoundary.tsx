// React 18 unmounts the whole root on an uncaught render throw, taking the topbar,
// the rail, the error banner and the Build button down with it — recovery is F5,
// losing the tree and the drill-down path. This keeps the failure to a message.
import { Component } from "react";
import type { ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Render failed:", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="panel" style={{ margin: "1rem" }}>
        <h2>Something went wrong rendering this view.</h2>
        <p className="hint">{error.message}</p>
        <button type="button" onClick={() => this.setState({ error: null })}>
          Try again
        </button>
      </div>
    );
  }
}
