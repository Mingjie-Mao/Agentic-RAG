import React, { useEffect, useRef, useState } from "react";
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

export type Bbox = {
  l: number;
  r: number;
  t: number;
  b: number;
  coord_origin?: string;
};
export type Provenance = {
  page: number;
  bbox: Bbox;
  charspan?: [number, number];
};
export type PdfLocator = {
  label: string;
  page?: number;
  provenance?: Provenance[];
};

type Rect = { left: number; top: number; width: number; height: number };

/** Highlight boxes come from the parser, so they mark the exact span that was indexed. */
function toRects(spans: Provenance[], viewport: pdfjs.PageViewport): Rect[] {
  return spans.map((span) => {
    const [x1, y1, x2, y2] = viewport.convertToViewportRectangle([
      span.bbox.l,
      span.bbox.b,
      span.bbox.r,
      span.bbox.t,
    ]);
    return {
      left: Math.min(x1, x2),
      top: Math.min(y1, y2),
      width: Math.abs(x2 - x1),
      height: Math.abs(y2 - y1),
    };
  });
}

export default function PdfEvidence({
  url,
  locator,
}: {
  url: string;
  locator: PdfLocator;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const holderRef = useRef<HTMLDivElement | null>(null);
  const [holderWidth, setHolderWidth] = useState(420);
  const [zoom, setZoom] = useState(1);
  const [rects, setRects] = useState<Rect[]>([]);
  const [size, setSize] = useState<{ width: number; height: number } | null>(
    null,
  );
  const [pages, setPages] = useState(0);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const page = locator.page ?? 1;

  useEffect(() => {
    const holder = holderRef.current;
    if (!holder) return;
    const observer = new ResizeObserver((entries) =>
      setHolderWidth(entries[0].contentRect.width),
    );
    observer.observe(holder);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!loading && rects.length)
      holderRef.current
        ?.querySelector(".pdf-mark")
        ?.scrollIntoView({ block: "center", inline: "center" });
  }, [loading, rects]);
  useEffect(() => {
    let cancelled = false;
    let task: pdfjs.PDFDocumentLoadingTask | null = null;
    setLoading(true);
    setError("");
    setRects([]);
    (async () => {
      try {
        // Chinese PDFs embed CID fonts; without the CMap and standard-font tables
        // the page renders blank while still reporting success.
        task = pdfjs.getDocument({
          url,
          withCredentials: true,
          cMapUrl: "/cmaps/",
          cMapPacked: true,
          standardFontDataUrl: "/standard_fonts/",
        });
        const doc = await task.promise;
        if (cancelled) return;
        setPages(doc.numPages);
        const target = Math.min(Math.max(page, 1), doc.numPages);
        const pdfPage = await doc.getPage(target);
        if (cancelled) return;
        const width = Math.max(1, holderWidth) * zoom;
        const base = pdfPage.getViewport({ scale: 1 });
        const scale = width / base.width;
        const viewport = pdfPage.getViewport({ scale });
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ratio = window.devicePixelRatio || 1;
        canvas.width = Math.floor(viewport.width * ratio);
        canvas.height = Math.floor(viewport.height * ratio);
        const context = canvas.getContext("2d");
        if (!context) return;
        context.scale(ratio, ratio);
        await pdfPage.render({ canvasContext: context, viewport }).promise;
        if (cancelled) return;
        setSize({ width: viewport.width, height: viewport.height });
        const spans = (locator.provenance ?? []).filter(
          (item) => item.page === target,
        );
        setRects(toRects(spans, viewport));
        setLoading(false);
      } catch (exception) {
        if (cancelled) return;
        setError(
          exception instanceof Error ? exception.message : "无法打开原文",
        );
        setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
      task?.destroy();
    };
  }, [url, page, locator, holderWidth, zoom]);

  return (
    <div className="pdf-evidence" ref={holderRef} data-testid="pdf-evidence">
      <div className="pdf-caption">
        <span>
          第 {page} 页{pages ? ` / 共 ${pages} 页` : ""}
        </span>
        {rects.length > 0 ? (
          <span className="pdf-hit" data-testid="pdf-highlight-count">
            已标出 {rects.length} 处引用位置
          </span>
        ) : (
          !loading &&
          !error && <span className="pdf-hit muted">本页无坐标信息</span>
        )}
      </div>
      <div className="pdf-zoom">
        <button
          className="secondary"
          aria-label="缩小原文"
          disabled={zoom <= 1}
          onClick={() => setZoom((v) => Math.max(1, v - 0.25))}
        >
          −
        </button>
        <span>{Math.round(zoom * 100)}%</span>
        <button
          className="secondary"
          aria-label="放大原文"
          disabled={zoom >= 2.5}
          onClick={() => setZoom((v) => Math.min(2.5, v + 0.25))}
        >
          ＋
        </button>
        <button className="secondary" onClick={() => setZoom(1)}>
          适合宽度
        </button>
      </div>
      <div
        className="pdf-stage"
        style={size ? { height: size.height } : undefined}
      >
        <canvas
          ref={canvasRef}
          style={size ? { width: size.width, height: size.height } : undefined}
        />
        {rects.map((rect, index) => (
          <span
            key={index}
            className="pdf-mark"
            data-testid="pdf-mark"
            style={{
              left: rect.left,
              top: rect.top,
              width: rect.width,
              height: rect.height,
            }}
          />
        ))}
        {loading && <div className="pdf-state">正在打开原文…</div>}
        {error && <div className="pdf-state pdf-error">{error}</div>}
      </div>
    </div>
  );
}
