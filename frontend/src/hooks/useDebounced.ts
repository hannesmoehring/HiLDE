import { useEffect, useState } from "react";

// Trail a fast-changing value by `ms`. A range slider fires on every pixel of a
// drag; the work behind it (re-scan the node, refetch the selected rows) is far
// too heavy to run at that rate, so the committed value lags the visible one.
export function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);

  return settled;
}
