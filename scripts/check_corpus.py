"""Validate the scholarship corpus and its two source inventories."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "hoc-bong"
REQUIRED = {
    "doc_id", "title", "source_url", "retrieved_at", "document_version",
    "audience", "institution", "department", "category", "language",
}
MANIFEST_FIELDS = {
    "doc_id", "file_path", "title", "source_url", "retrieved_at",
    "document_version", "license_or_permission",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_document(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    sections = text.split("---", 2)
    if len(sections) != 3 or sections[0].strip():
        raise ValueError("missing YAML front matter")
    metadata = {}
    for line in sections[1].strip().splitlines():
        key, separator, value = line.partition(": ")
        if not separator or key in metadata:
            raise ValueError(f"invalid metadata line: {line}")
        metadata[key] = value.strip('"')
    return metadata, sections[2].strip()


def main() -> int:
    errors: list[str] = []
    files = sorted(CORPUS.glob("*.md"))
    if len(files) != 7:
        errors.append(f"expected exactly 7 Markdown files, found {len(files)}")

    planned_rows = read_csv(CORPUS / "urls.csv")
    manifest_rows = read_csv(CORPUS / "sources.csv")
    planned = {row["doc_id"]: row for row in planned_rows}
    manifest = {row["doc_id"]: row for row in manifest_rows}
    if len(planned) != len(planned_rows) or len(manifest) != len(manifest_rows):
        errors.append("duplicate doc_id in a CSV")
    if len(files) != len(manifest_rows) or len(files) != len(planned_rows):
        errors.append("Markdown count differs from CSV row count")

    audiences = set()
    for path in files:
        try:
            metadata, body = read_document(path)
        except ValueError as error:
            errors.append(f"{path.name}: {error}")
            continue
        doc_id = metadata.get("doc_id", "")
        missing = REQUIRED - metadata.keys()
        if missing:
            errors.append(f"{path.name}: missing {', '.join(sorted(missing))}")
        if doc_id != path.stem:
            errors.append(f"{path.name}: doc_id does not match filename")
        if len(body) < 300:
            errors.append(f"{path.name}: body is too short")
        if "example.edu" in body or "example.edu" in metadata.get("source_url", ""):
            errors.append(f"{path.name}: contains a sample URL")
        url = metadata.get("source_url", "")
        if urlparse(url).scheme != "https" or not urlparse(url).netloc:
            errors.append(f"{path.name}: invalid source_url")
        try:
            date.fromisoformat(metadata.get("retrieved_at", ""))
        except ValueError:
            errors.append(f"{path.name}: invalid retrieved_at")
        if not metadata.get("document_version"):
            errors.append(f"{path.name}: missing document_version")
        if metadata.get("audience") not in {"student", "faculty", "staff", "all"}:
            errors.append(f"{path.name}: invalid audience")
        audiences.add(metadata.get("audience"))

        source_row = planned.get(doc_id)
        inventory_row = manifest.get(doc_id)
        if source_row is None or inventory_row is None:
            errors.append(f"{path.name}: missing CSV row")
            continue
        for key in ("title", "source_url", "retrieved_at", "document_version"):
            if key in inventory_row and inventory_row[key] != metadata.get(key):
                errors.append(f"{path.name}: sources.csv {key} mismatch")
        for key in ("title", "audience", "institution", "department", "category", "language", "document_version"):
            if source_row.get(key) != metadata.get(key):
                errors.append(f"{path.name}: urls.csv {key} mismatch")
        if source_row.get("url") != url:
            errors.append(f"{path.name}: urls.csv URL mismatch")
        if inventory_row.get("file_path") != path.relative_to(ROOT).as_posix():
            errors.append(f"{path.name}: sources.csv file_path mismatch")
        if not MANIFEST_FIELDS.issubset(inventory_row) or not inventory_row.get("license_or_permission"):
            errors.append(f"{path.name}: incomplete sources.csv row")

    if set(planned) != {path.stem for path in files} or set(manifest) != {path.stem for path in files}:
        errors.append("CSV doc_id set differs from Markdown files")
    if not {"student", "faculty"}.issubset(audiences):
        errors.append("student and faculty documents must be separated")

    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"OK: {len(files)} Markdown files; urls.csv and sources.csv match; audiences: {', '.join(sorted(audiences))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
