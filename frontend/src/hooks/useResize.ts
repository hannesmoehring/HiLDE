import { useEffect, useRef, useState } from "react";

// Observe an element's size for responsive D3 charts. Returns a ref to attach and
// the measured { width, height } of that element.
export function useResize<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect;
      setSize({ width: rect.width, height: rect.height });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return { ref, size };
}
