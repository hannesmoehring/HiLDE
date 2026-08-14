// The image behind one point, for datasets whose rows are images.
//
// The server sends raw greyscale pixels, so the canvas is drawn at the image's
// native size and blown up by CSS (`image-rendering: pixelated`) — an 8x8 digit
// stays a grid of 8x8 squares instead of being smoothed into a blur.
import { useEffect, useRef, useState } from "react";
import { fetchPointImage } from "../api";
import type { ImagePixels } from "../types";

interface Props {
  dataset: string;
  rowId: number;
  onClose: () => void;
}

export function PointImage({ dataset, rowId, onClose }: Props) {
  const [image, setImage] = useState<ImagePixels | null>(null);
  const [error, setError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    setImage(null);
    setError(null);
    fetchPointImage(dataset, rowId)
      .then((im) => !cancelled && setImage(im))
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [dataset, rowId]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!image || !ctx) return;
    const buffer = ctx.createImageData(image.width, image.height);
    image.pixels.forEach((v, i) => {
      buffer.data[i * 4] = v;
      buffer.data[i * 4 + 1] = v;
      buffer.data[i * 4 + 2] = v;
      buffer.data[i * 4 + 3] = 255;
    });
    ctx.putImageData(buffer, 0, 0);
  }, [image]);

  return (
    <div className="point-image">
      <div className="point-image__head">
        <strong>Row {rowId}</strong>
        <button type="button" onClick={onClose}>
          Close
        </button>
      </div>
      {error ? (
        <p className="hint point-image__note">Image unavailable.</p>
      ) : image ? (
        <>
          <canvas
            ref={canvasRef}
            width={image.width}
            height={image.height}
            className="point-image__canvas"
          />
          <p className="point-image__note">
            {image.width}×{image.height}
          </p>
        </>
      ) : (
        <p className="hint point-image__note">Loading image…</p>
      )}
    </div>
  );
}
