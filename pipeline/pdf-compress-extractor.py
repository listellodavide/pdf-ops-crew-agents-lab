#!/usr/bin/env python3
"""Phase 1 of the PDF inbox ingestion pipeline: deterministic extraction and compression.

For every PDF in the project root (or the files/folders given on the command line) it writes
four artifacts to ./pdf-inbox-ingestion/:

    <stem>.txt             normalized UTF-8 text for the AI phases, with [[page N]] markers,
                           repeated headers/footers removed and ruled tables as Markdown
    <stem>.layout.json     page -> blocks -> lines -> spans with bounding boxes and fonts,
                           plus detected tables (schema "pdf-layout-v1")
    <stem>.compressed.pdf  smaller PDF: images downsampled and recompressed, fonts subset,
                           objects deduplicated; scanned pages get an invisible OCR text layer
    <stem>.metadata.json   hashes, sizes, per-page statistics, OCR decisions, warnings

Design rules
    * Native text is never replaced by OCR. Born-digital PDFs (catalogs, reports) already carry
      exact text; OCR would introduce errors in part numbers, units, dates and names.
    * OCR runs only on pages with almost no native text but significant image area (--ocr auto).
    * Re-running is cheap: a PDF whose hash and settings did not change is skipped (--force to redo).

Requires Tesseract (a separate program, not a Python package) only when OCR is needed.
Windows 11:  winget install UB-Mannheim.TesseractOCR

Examples
    python .\\pdf-compress-extractor.py
    python .\\pdf-compress-extractor.py .\\catalog.pdf --mode text
    python .\\pdf-compress-extractor.py --ocr auto --ocr-language "eng+deu+ita+ron"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pymupdf

if hasattr(pymupdf, "no_recommend_layout"):
    pymupdf.no_recommend_layout()          # silence the optional-package hint

SCHEMA_LAYOUT = "pdf-layout-v1"
SCHEMA_METADATA = "pdf-ingestion-metadata-v1"
DEFAULT_OUTPUT = Path("pdf-inbox-ingestion")
HEADER_FOOTER_BAND = 0.08        # top and bottom 8% of the page height
PAGE_NUMBER_KEYS = {"#", "page #", "page # of #", "# / #", "- # -", "# of #"}
MIN_OCR_QUALITY = 0.5            # below this the page is retried sideways and flagged
WINDOWS_TESSDATA = [
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tesseract-OCR" / "tessdata",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Tesseract-OCR" / "tessdata",
]


# --------------------------------------------------------------------------------------- utils

def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def r2(values) -> list[float]:
    return [round(float(v), 2) for v in values]


def color_hex(value: int) -> str:
    return f"#{value:06x}"


def normalize_line(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    return re.sub(r"[ \t\u00a0]+", " ", text).strip()


def boilerplate_key(text: str) -> str:
    """Page numbers and dates change from page to page; compare lines with digits masked."""
    return re.sub(r"\d+", "#", text.lower()).strip()


def ocr_quality(text: str) -> float:
    """Share of characters that belong to plausible words or numbers (0..1).

    A readable scan scores 0.7 to 0.95; a sideways or noisy scan produces fragments such as
    'Oo', 'mM', '=' and scores below 0.4.
    """
    tokens = text.split()
    total = sum(len(t) for t in tokens)
    if not total:
        return 0.0
    good = sum(len(t) for t in tokens
               if re.fullmatch(r"[(\[\"']?[^\W\d_]{3,}[)\]\"'.,;:!?]*", t)
               or re.fullmatch(r"\d+([.,:/-]\d+)*[.,;:]?", t))
    return good / total


def find_tessdata(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    if os.environ.get("TESSDATA_PREFIX"):
        return os.environ["TESSDATA_PREFIX"]
    try:
        found = pymupdf.get_tessdata()
        if found:
            return found
    except Exception:
        pass
    for candidate in WINDOWS_TESSDATA:
        if candidate.is_dir():
            return str(candidate)
    return None


def collect_inputs(inputs: list[str], output_dir: Path) -> list[Path]:
    paths = [Path(p) for p in inputs] or [Path.cwd()]
    pdfs: list[Path] = []
    for p in paths:
        if p.is_dir():
            pdfs.extend(sorted(q for q in p.glob("*.pdf") if q.is_file()))
            pdfs.extend(sorted(q for q in p.glob("*.PDF") if q.is_file() and q.suffix == ".PDF"))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        else:
            log(f"[WARN] skipped, not a PDF or folder: {p}")
    out = output_dir.resolve()
    unique = []
    for p in pdfs:   # never re-ingest our own outputs
        if out not in p.resolve().parents and p.resolve() not in {u.resolve() for u in unique}:
            unique.append(p)
    return unique


# ------------------------------------------------------------------------------ page analysis

def image_coverage(page: pymupdf.Page) -> float:
    """Share of the page area covered by images (0..1, overlaps not subtracted)."""
    area = abs(page.rect) or 1.0
    covered = 0.0
    for info in page.get_image_info():
        covered += abs(pymupdf.Rect(info["bbox"]) & page.rect)
    return min(1.0, covered / area)


def needs_ocr(native_chars: int, coverage: float, mode: str, min_chars: int, min_image: float) -> bool:
    if mode == "never":
        return False
    if mode == "always":
        return True
    return native_chars < min_chars and coverage >= min_image


def ocr_textpage(page: pymupdf.Page, language: str, dpi: int, tessdata: str | None):
    kwargs = {"language": language, "dpi": dpi, "full": True}
    if tessdata:
        kwargs["tessdata"] = tessdata
    return page.get_textpage_ocr(**kwargs)


def ocr_best_orientation(doc, page, args, tessdata, holder: list):
    """OCR the page as displayed; if the result looks like noise, retry turned 90 and 270 degrees.

    Returns (textpage, page object the textpage belongs to, quality, turned?).
    """
    tp = ocr_textpage(page, args.ocr_language, args.ocr_dpi, tessdata)
    best = (tp, page, ocr_quality(page.get_text("text", textpage=tp)), False)
    if best[2] >= MIN_OCR_QUALITY:
        return best
    for extra in (90, 270):
        tmp = pymupdf.open()
        tmp.insert_pdf(doc, from_page=page.number, to_page=page.number)
        turned = tmp[0]
        turned.set_rotation((page.rotation + extra) % 360)
        tp2 = ocr_textpage(turned, args.ocr_language, args.ocr_dpi, tessdata)
        quality = ocr_quality(turned.get_text("text", textpage=tp2))
        holder.append(tmp)
        if quality > best[2] + 0.15:
            best = (tp2, turned, quality, True)
    return best


def to_markdown(cells: list[list[str]]) -> str:
    """Markdown table from extracted cells (PyMuPDF's own to_markdown escapes HTML and can
    drop ligature glyphs such as 'ti' and 'tt')."""
    def cell(value: str) -> str:
        return value.replace("|", "\\|")
    width = max(len(row) for row in cells)
    rows = [[cell(c) for c in row] + [""] * (width - len(row)) for row in cells]
    lines = ["|" + "|".join(rows[0]) + "|", "|" + "|".join(["---"] * width) + "|"]
    lines += ["|" + "|".join(row) + "|" for row in rows[1:]]
    return "\n".join(lines)


def detect_tables(page: pymupdf.Page) -> list[dict]:
    """Ruled tables (spare-part lists, emission tables). Returns bbox, rows and Markdown."""
    tables = []
    try:
        found = page.find_tables()
    except Exception:
        return tables
    for tab in found.tables:
        rows = tab.extract()
        if tab.row_count < 2 or tab.col_count < 2:
            continue
        filled = sum(1 for row in rows for cell in row if cell and str(cell).strip())
        if filled < 4:
            continue
        cleaned = [[normalize_line(str(c).replace("\n", " ")) if c is not None else "" for c in row]
                   for row in rows]
        tables.append({
            "bbox": r2(tab.bbox),
            "rows": tab.row_count,
            "cols": tab.col_count,
            "header": cleaned[0],
            "cells": cleaned,
            "markdown": to_markdown(cleaned),
        })
    return tables


def page_dict(page: pymupdf.Page, textpage=None) -> dict:
    flags = pymupdf.TEXTFLAGS_DICT | pymupdf.TEXT_DEHYPHENATE
    flags &= ~pymupdf.TEXT_PRESERVE_IMAGES      # image metadata only, never the pixel data
    return page.get_text("dict", flags=flags, textpage=textpage)


def layout_for_page(page: pymupdf.Page, raw: dict, source: str, tables: list[dict]) -> dict:
    """Coordinates are converted to the page as displayed (rotation applied), origin top-left."""
    m = page.rotation_matrix

    def box(b) -> list[float]:
        return r2((pymupdf.Rect(b) * m).normalize()) if page.rotation else r2(b)

    def point(p) -> list[float]:
        return r2(pymupdf.Point(p) * m) if page.rotation else r2(p)

    blocks = []
    for b_index, block in enumerate(raw.get("blocks", [])):
        if block.get("type") == 1:
            blocks.append({"block": b_index, "type": "image", "bbox": box(block["bbox"]),
                           "width": block.get("width"), "height": block.get("height"),
                           "ext": block.get("ext")})
            continue
        lines = []
        for l_index, line in enumerate(block.get("lines", [])):
            spans = []
            for s_index, span in enumerate(line.get("spans", [])):
                text = normalize_line(span.get("text", ""))
                if not text:
                    continue
                f = span.get("flags", 0)
                spans.append({
                    "span": s_index, "text": text, "bbox": box(span["bbox"]),
                    "origin": point(span["origin"]), "font": span.get("font"),
                    "size": round(span.get("size", 0), 2), "flags": f,
                    "bold": bool(f & 16), "italic": bool(f & 2), "mono": bool(f & 8),
                    "color": color_hex(span.get("color", 0)),
                })
            if spans:
                lines.append({"line": l_index, "text": " ".join(s["text"] for s in spans),
                              "bbox": box(line["bbox"]), "direction": r2(line.get("dir", (1, 0))),
                              "spans": spans})
        if lines:
            blocks.append({"block": b_index, "type": "text", "bbox": box(block["bbox"]),
                           "text": "\n".join(ln["text"] for ln in lines), "lines": lines})
    return {
        "page": page.number + 1,
        "width": round(page.rect.width, 2),
        "height": round(page.rect.height, 2),
        "rotation": page.rotation,
        "source": source,
        "blocks": blocks,
        "tables": [{k: v for k, v in t.items() if k != "markdown"} for t in tables],
    }


def page_lines(layout_page: dict) -> list[dict]:
    """Flatten text lines with their vertical position, in content order."""
    out = []
    for block in layout_page["blocks"]:
        if block["type"] != "text":
            continue
        for line in block["lines"]:
            out.append({"text": line["text"], "bbox": line["bbox"]})
    return out


def find_boilerplate(pages: list[dict], min_share: float = 0.5) -> set[str]:
    """Lines in the header/footer band that repeat on at least half of the pages (min 3 pages)."""
    if len(pages) < 3:
        return set()
    counts: Counter[str] = Counter()
    for p in pages:
        top, bottom = p["height"] * HEADER_FOOTER_BAND, p["height"] * (1 - HEADER_FOOTER_BAND)
        keys = {boilerplate_key(ln["text"]) for ln in page_lines(p)
                if (ln["bbox"][3] <= top or ln["bbox"][1] >= bottom) and len(ln["text"]) <= 150}
        counts.update(k for k in keys if k and k != "#")
    threshold = max(3, int(len(pages) * min_share + 0.5))
    found = {k for k, c in counts.items() if c >= threshold}
    # bare page numbers ("12", "Page 3 of 40") are boilerplate even if formatted differently
    return found | PAGE_NUMBER_KEYS


def page_text(layout_page: dict, tables: list[dict], boilerplate: set[str], seen: set[str]) -> str:
    """Page text in content order. A repeated header/footer line is kept the first time it
    appears in the document (it often names the section) and dropped on later pages."""
    height = layout_page["height"]
    top, bottom = height * HEADER_FOOTER_BAND, height * (1 - HEADER_FOOTER_BAND)
    table_rects = [pymupdf.Rect(t["bbox"]) for t in tables]
    emitted_tables: set[int] = set()
    out: list[str] = []
    for block in layout_page["blocks"]:
        if block["type"] != "text":
            continue
        lines_out = []
        for line in block["lines"]:
            box = pymupdf.Rect(line["bbox"])
            in_band = box.y1 <= top or box.y0 >= bottom
            key = boilerplate_key(line["text"])
            if in_band and key in boilerplate:
                if key in PAGE_NUMBER_KEYS or key in seen:
                    continue
                seen.add(key)
            centre = pymupdf.Point((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)
            hit = next((i for i, r in enumerate(table_rects) if centre in r), None)
            if hit is not None:            # replace table text with its Markdown, once
                if hit not in emitted_tables:
                    emitted_tables.add(hit)
                    if lines_out:
                        out.append("\n".join(lines_out))
                        lines_out = []
                    out.append(tables[hit]["markdown"])
                continue
            lines_out.append(line["text"])
        if lines_out:
            out.append("\n".join(lines_out))
    for i, t in enumerate(tables):           # tables whose cells had no extractable lines
        if i not in emitted_tables:
            out.append(t["markdown"])
    text = "\n\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ------------------------------------------------------------------------------- compression

def add_invisible_text(page: pymupdf.Page, words: list[tuple]) -> int:
    """Write OCR words as invisible text (render mode 3) over the scanned image."""
    font = pymupdf.Font("helv")
    writer = pymupdf.TextWriter(page.rect)
    to_display = page.rotation_matrix          # OCR words are in unrotated page space
    written = 0
    for x0, y0, x1, y1, word, *_ in words:
        box = (pymupdf.Rect(x0, y0, x1, y1) * to_display).normalize()
        if not word.strip() or box.is_empty:
            continue
        unit = font.text_length(word, fontsize=1) or 1
        size = max(1.0, min(box.height * 0.95, box.width / unit))
        try:
            writer.append((box.x0, box.y1 - box.height * 0.2), word, font=font, fontsize=size)
            written += 1
        except Exception:
            continue
    if written:
        writer.write_text(page, render_mode=3)     # render mode 3 = invisible, still searchable
    return written


def compress(source: Path, target: Path, ocr_words: dict[int, list[tuple]], image_dpi: int,
             jpeg_quality: int, fixed_rotation: dict[int, int]) -> tuple[int, list[str]]:
    warnings: list[str] = []
    doc = pymupdf.open(source)
    try:
        for page_index, rotation in fixed_rotation.items():     # sideways scans display upright
            doc[page_index].set_rotation(rotation)
        for page_index, words in ocr_words.items():
            add_invisible_text(doc[page_index], words)
        try:
            doc.rewrite_images(dpi_threshold=image_dpi + 30, dpi_target=image_dpi,
                               quality=jpeg_quality, lossy=True, lossless=True, bitonal=True,
                               color=True, gray=True)
        except Exception as e:
            warnings.append(f"image rewrite skipped: {e}")
        try:
            doc.subset_fonts()
        except Exception as e:
            warnings.append(f"font subsetting skipped: {e}")
        doc.save(target, garbage=4, clean=True, deflate=True, deflate_images=True,
                 deflate_fonts=True, use_objstms=1)
    finally:
        doc.close()
    original = source.stat().st_size
    if target.stat().st_size >= original and not ocr_words:
        target.write_bytes(source.read_bytes())       # never ship a bigger "compressed" file
        warnings.append("compression did not reduce size; original bytes kept")
    return target.stat().st_size, warnings


# ------------------------------------------------------------------------------- per document

def settings_fingerprint(args: argparse.Namespace) -> dict:
    return {"mode": args.mode, "ocr": args.ocr, "ocr_language": args.ocr_language,
            "ocr_dpi": args.ocr_dpi, "min_chars": args.min_chars, "min_image": args.min_image,
            "image_dpi": args.image_dpi, "jpeg_quality": args.jpeg_quality,
            "tables": args.tables, "keep_boilerplate": args.keep_boilerplate,
            "prettyprint": args.prettyprint}


def process_pdf(pdf: Path, args: argparse.Namespace, tessdata: str | None) -> dict:
    out = Path(args.output)
    stem = pdf.stem
    paths = {k: out / f"{stem}{suffix}" for k, suffix in {
        "text": ".txt", "layout": ".layout.json", "compressed": ".compressed.pdf",
        "metadata": ".metadata.json"}.items()}
    digest = sha256_of(pdf)
    settings = settings_fingerprint(args)

    if not args.force and paths["metadata"].exists():
        try:
            previous = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if previous.get("source", {}).get("sha256") == digest and previous.get("settings") == settings:
                log(f"[SKIP] {pdf.name}: unchanged since {previous.get('processed_at')}")
                return {"file": pdf.name, "status": "skipped"}
        except Exception:
            pass

    started = time.perf_counter()
    log(f"\n[FILE] {pdf.name}")
    warnings: list[str] = []
    doc = pymupdf.open(pdf)
    if doc.needs_pass:
        doc.close()
        raise RuntimeError("password protected; cannot process")

    layout_pages: list[dict] = []
    page_tables: list[list[dict]] = []
    page_stats: list[dict] = []
    ocr_words: dict[int, list[tuple]] = {}
    fixed_rotation: dict[int, int] = {}
    holder: list = []                  # keeps temporary one-page documents alive while in use
    want_text = args.mode in ("all", "text")

    for page in doc:
        native = page.get_text("text").strip()
        native_chars = len(re.sub(r"\s", "", native))
        coverage = image_coverage(page)
        do_ocr = needs_ocr(native_chars, coverage, args.ocr, args.min_chars, args.min_image)
        source, textpage, quality, text_page_obj = "native", None, None, page
        if not do_ocr and native_chars < args.min_chars and coverage >= args.min_image:
            warnings.append(f"page {page.number + 1}: image-only page without text (OCR is off)")
        if do_ocr:
            if tessdata is None:
                warnings.append(f"page {page.number + 1}: OCR needed but Tesseract tessdata not found")
            else:
                try:
                    textpage, text_page_obj, quality, turned = ocr_best_orientation(
                        doc, page, args, tessdata, holder)
                    source = "ocr"
                    if turned:
                        fixed_rotation[page.number] = text_page_obj.rotation
                        log(f"  [OCR] page {page.number + 1} was sideways: turned to {text_page_obj.rotation} degrees")
                    if quality < MIN_OCR_QUALITY:
                        warnings.append(f"page {page.number + 1}: low OCR quality {quality:.2f} "
                                        "(poor scan, handwriting or unsupported language)")
                    if native_chars < args.min_chars:      # only pages without real text get a layer
                        ocr_words[page.number] = text_page_obj.get_text("words", textpage=textpage)
                except Exception as e:
                    warnings.append(f"page {page.number + 1}: OCR failed: {e}")
        tables = detect_tables(page) if (args.tables and source == "native" and want_text) else []
        raw = page_dict(text_page_obj, textpage)
        layout = layout_for_page(text_page_obj, raw, source, tables)
        layout["page"] = page.number + 1
        layout_pages.append(layout)
        page_tables.append(tables)
        page_stats.append({"page": page.number + 1, "native_chars": native_chars,
                           "image_coverage": round(coverage, 3), "source": source,
                           "ocr_quality": round(quality, 3) if quality is not None else None,
                           "rotation_fixed_to": fixed_rotation.get(page.number),
                           "tables": len(tables)})
        if source == "ocr":
            log(f"  [OCR] page {page.number + 1} (native chars {native_chars}, images {coverage:.0%})")

    page_count = doc.page_count
    pdf_meta = {k: v for k, v in (doc.metadata or {}).items() if v}
    for tmp in holder:
        tmp.close()
    doc.close()

    boilerplate = set() if args.keep_boilerplate else find_boilerplate(layout_pages)
    removed = sorted(boilerplate - PAGE_NUMBER_KEYS)
    out.mkdir(parents=True, exist_ok=True)

    text_chars = 0
    if want_text:
        parts, seen = [], set()
        for lp, tables, stat in zip(layout_pages, page_tables, page_stats, strict=True):
            body = page_text(lp, tables, boilerplate, seen)
            stat["text_chars"] = len(body)
            parts.append(f"[[page {lp['page']}]]\n{body}" if body else f"[[page {lp['page']}]]")
        text = "\n\n".join(parts) + "\n"
        text_chars = len(text)
        paths["text"].write_text(text, encoding="utf-8")
        log(f"  [OK] text    {paths['text'].name} ({text_chars:,} chars)")

        layout = {"schema": SCHEMA_LAYOUT, "source_file": pdf.name, "page_count": page_count,
                  "removed_header_footer_lines": removed, "pages": layout_pages}
        indent = 4 if args.prettyprint else None
        separators = None if args.prettyprint else (",", ":")
        paths["layout"].write_text(json.dumps(layout, ensure_ascii=False, indent=indent,
                                              separators=separators), encoding="utf-8")
        log(f"  [OK] layout  {paths['layout'].name}")

    compressed_size = None
    if args.mode in ("all", "compress"):
        compressed_size, w = compress(pdf, paths["compressed"], ocr_words, args.image_dpi,
                                    args.jpeg_quality, fixed_rotation)
        warnings.extend(w)
        log(f"  [OK] pdf     {paths['compressed'].name} "
            f"({pdf.stat().st_size / 1024:,.0f} KB -> {compressed_size / 1024:,.0f} KB)")

    original_size = pdf.stat().st_size
    metadata = {
        "schema": SCHEMA_METADATA,
        "processed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": {"file": pdf.name, "sha256": digest, "bytes": original_size,
                   "page_count": page_count, "pdf_metadata": pdf_meta},
        "outputs": {k: p.name for k, p in paths.items() if p.exists() and k != "metadata"},
        "sizes": {"original_bytes": original_size, "compressed_bytes": compressed_size,
                  "compression_ratio": round(compressed_size / original_size, 3) if compressed_size else None,
                  "text_chars": text_chars or None},
        "ocr": {"mode": args.ocr, "language": args.ocr_language,
                "pages": [s["page"] for s in page_stats if s["source"] == "ocr"],
                "tessdata": tessdata},
        "tables_detected": sum(s["tables"] for s in page_stats),
        "removed_header_footer_lines": removed,
        "pages": page_stats,
        "settings": settings,
        "tool_versions": {"python": sys.version.split()[0], "pymupdf": pymupdf.VersionBind},
        "warnings": warnings,
        "seconds": round(time.perf_counter() - started, 2),
    }
    metadata_indent = 4 if args.prettyprint else 2
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=metadata_indent),
                                 encoding="utf-8")
    for w in warnings:
        log(f"  [WARN] {w}")
    return {"file": pdf.name, "status": "ok", "pages": page_count,
            "ocr_pages": len(metadata["ocr"]["pages"]), "original_kb": original_size // 1024,
            "compressed_kb": compressed_size // 1024 if compressed_size else None,
            "text_chars": text_chars}


# ------------------------------------------------------------------------------------- CLI

def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 1: extract normalized text and layout JSON from PDFs and compress them.")
    p.add_argument("inputs", nargs="*",
                   help="PDF files or folders (default: PDFs in the current folder, the project root)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output folder (default: %(default)s)")
    p.add_argument("--mode", choices=("all", "text", "compress"), default="all",
                   help="all = text + layout + compressed PDF (default); text; compress")
    p.add_argument("--ocr", choices=("never", "auto", "always"), default="auto",
                   help="auto = OCR only pages with little native text and large images (default)")
    p.add_argument("--ocr-language", default="eng",
                   help="Tesseract languages, e.g. 'eng' or 'eng+deu+ita+ron' (default: %(default)s)")
    p.add_argument("--ocr-dpi", type=int, default=300, help="render DPI for OCR (default: %(default)s)")
    p.add_argument("--tessdata", help="Tesseract tessdata folder (auto-detected if omitted)")
    p.add_argument("--min-chars", type=int, default=40,
                   help="pages with fewer native characters are OCR candidates (default: %(default)s)")
    p.add_argument("--min-image", type=float, default=0.3,
                   help="minimum image coverage (0..1) for an OCR candidate page (default: %(default)s)")
    p.add_argument("--image-dpi", type=int, default=150,
                   help="target DPI for images in the compressed PDF (default: %(default)s)")
    p.add_argument("--jpeg-quality", type=int, default=70, help="JPEG quality 1..95 (default: %(default)s)")
    p.add_argument("--no-tables", dest="tables", action="store_false",
                   help="do not convert ruled tables to Markdown")
    p.add_argument("--keep-boilerplate", action="store_true",
                   help="keep repeated header/footer lines in the text")
    p.add_argument("--prettyprint", "--pretty", dest="prettyprint", action="store_true",
                   help="indent all JSON output with 4 spaces (larger files)")
    p.add_argument("--force", action="store_true", help="reprocess files even if unchanged")
    return p.parse_args(argv)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows consoles
    try:
        args = parse_arguments()
        pdfs = collect_inputs(args.inputs, Path(args.output))
        if not pdfs:
            log("No PDF files found. Put PDFs in the project root or pass paths.")
            return 1
        tessdata = find_tessdata(args.tessdata) if args.ocr != "never" else None
        if args.ocr != "never" and tessdata is None:
            log("[WARN] Tesseract not found: scanned pages will have no text. "
                "Install it (winget install UB-Mannheim.TesseractOCR) or pass --tessdata.")
        log(f"[PHASE 1] {len(pdfs)} PDF(s) -> {Path(args.output).resolve()}")
        results, failed = [], 0
        for pdf in pdfs:
            try:
                results.append(process_pdf(pdf, args, tessdata))
            except Exception as e:
                failed += 1
                log(f"  [ERROR] {pdf.name}: {e}")
                results.append({"file": pdf.name, "status": "error", "error": str(e)})
        log("\n[SUMMARY]")
        for r in results:
            if r["status"] == "ok":
                size = f"{r['original_kb']:,} KB -> {r['compressed_kb']:,} KB" if r["compressed_kb"] is not None else ""
                log(f"  {r['file']}: {r['pages']} pages, OCR {r['ocr_pages']}, "
                    f"{r['text_chars']:,} chars {size}")
            else:
                log(f"  {r['file']}: {r['status']} {r.get('error', '')}")
        return 1 if failed else 0
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
