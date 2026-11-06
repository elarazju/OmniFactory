"""Reviewed capability rows and immutable evidence feed CLI and README together."""

import hashlib
import json
from functools import lru_cache
from importlib.resources import files

from omnifactory.errors import OmniFactoryError


@lru_cache(maxsize=1)
def catalog():
    root = files("omnifactory") / "evidence"
    result = json.loads((root / "catalog.json").read_text())
    for record in result["records"]:
        if hashlib.sha256((root / record["file"]).read_bytes()).hexdigest() != record["sha256"]:
            raise OmniFactoryError("E_EVIDENCE_HASH", record["id"])
    ids = {record["id"] for record in result["records"]}
    if any(set(row["verification_ids"]) - ids for row in result["rows"]):
        raise OmniFactoryError("E_EVIDENCE_REFERENCE", "capability row refers to missing evidence")
    return result


def verified_scopes(model_key):
    return [r for r in catalog()["records"] if r["model_key"] == model_key]


def render_table(language="en"):
    headers = (
        [
            "Model family",
            "Training inputs",
            "Training outputs",
            "Fine-tuning",
            "Recipes",
        ]
        if language == "en"
        else ["模型族", "训练输入", "训练输出", "微调方式", "配置入口"]
    )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row[language]) + " |" for row in catalog()["readme_rows"])
    return "\n".join(lines)


def update_readme(path, language, *, check=False):
    source = path.read_text()
    start = source.index("<!-- model-capabilities:start -->")
    end = source.index("<!-- model-capabilities:end -->", start)
    block = source[start:end]
    lines = block.splitlines()
    table_lines = [i for i, line in enumerate(lines) if line.startswith("|")]
    lines[table_lines[0] : table_lines[-1] + 1] = render_table(language).splitlines()
    updated = source[:start] + "\n".join(lines) + "\n" + source[end:]
    if check and updated != source:
        raise OmniFactoryError("E_DOC_DRIFT", f"regenerate {path}")
    if not check:
        path.write_text(updated)
