#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


# ---------------------------------------------------------------------------
# Student implementation (SEEM5660 / FTEC5660 HW1)
#
# Chain design: a two-stage LangChain pipeline.
#   Stage 1 (per receipt, in parallel): a vision prompt sends each receipt
#   image to ``deepseek-v4-flash-vision-exp`` and asks for a strict JSON
#   extraction of three numbers:
#       subtotal       - the SUBTOTAL / 小計 line (before ROUNDING)
#       final_paid     - the amount actually paid (OCTOPUS / 八達通 / CASH /
#                        EPS / Amount Deducted line, i.e. subtotal after
#                        the ROUNDING adjustment)
#       discount_total - the SUM of the absolute values of every
#                        discount/promotion/coupon line (5% OFF, Buy X Save,
#                        MB APP UPGRADE, App Upgrade, 包装变形, COUPON, ...).
#                        ROUNDING and charges such as plastic-bag fees are
#                        deliberately excluded.
#   Stage 2 (aggregation): the per-receipt extractions are summed with exact
#   Decimal arithmetic and formatted into the two required answers:
#       query1 = sum(final_paid)
#       query2 = sum(subtotal + discount_total)
# ---------------------------------------------------------------------------

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_deepseek import ChatDeepSeek

VISION_MODEL = "deepseek-v4-flash-vision-exp"
_MAX_RETRIES = 5  # per-receipt re-reads when the JSON extraction is empty/malformed
_RETRY_DELAY_SECONDS = 1.0

_EXTRACTION_SYSTEM = (
    "You are an expert OCR assistant for Hong Kong supermarket receipts "
    "(PARKnSHOP, fusion, Wellcome, etc.). You read one receipt image and "
    "return ONLY a JSON object with exactly three keys: subtotal, final_paid, "
    "discount_total. Do not add explanations, markdown fences, or extra keys."
)

_EXTRACTION_USER = """Read this receipt image carefully and return ONLY one JSON object:

{{
  "subtotal": <number>,
  "final_paid": <number>,
  "discount_total": <number>
}}

Rules:
1. "subtotal" = the value on the SUBTOTAL / 小計 line, i.e. the items total AFTER
   discounts have been applied but BEFORE the ROUNDING adjustment.
2. "final_paid" = the amount actually paid, taken from the payment line:
   OCTOPUS / 八達通 / CASH / 現金 / EPS / Amount Deducted / 扣除金額. This equals
   subtotal adjusted by the ROUNDING line (e.g. ROUNDING -0.01 on subtotal
   102.31 gives final_paid 102.30). If ROUNDING is absent, final_paid equals
   the payment line value.
3. "discount_total" = the SUM of the ABSOLUTE VALUES of every discount,
   promotion, coupon or price-adjustment line, for example:
   "5% OFF ..." -5.39  -> 5.39
   "Buy 2 Save $12.8" -12.80 -> 12.80
   "MB APP UPGRADE -$10" -10.00 -> 10.00
   "App Upgrade..." -20.00 -> 20.00
   "包装变形" (damaged packaging) -12.40 -> 12.40
   "COUPON" 0.00 -> 0.00
   Do NOT include the ROUNDING line. Do NOT include charges such as a plastic
   bag fee (they are costs, not discounts). If there is no discount at all,
   use 0.

Example output: {{"subtotal": 102.31, "final_paid": 102.30, "discount_total": 5.39}}"""


def _parse_extraction(text: str) -> dict[str, Decimal] | None:
    """Parse the model's answer into {subtotal, final_paid, discount_total}."""
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE
    )
    data: dict[str, Any] | None = None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            data = parsed
    except json.JSONDecodeError:
        data = None
    if data is None:  # fallback: pull key:value pairs out of the raw text
        data = {}
        for key in ("subtotal", "final_paid", "discount_total"):
            match = re.search(rf'"{key}"\s*:\s*(-?\d+(?:\.\d+)?)', cleaned)
            if match is not None:
                data[key] = match.group(1)
    try:
        values = {
            key: Decimal(str(data[key])).quantize(Decimal("0.01"))
            for key in ("subtotal", "final_paid", "discount_total")
            if key in data and str(data[key]).strip() not in ("", "null", "None")
        }
    except (InvalidOperation, TypeError, KeyError):
        return None
    if len(values) != 3:
        return None
    if values["final_paid"] < 0 or values["subtotal"] < 0 or values["discount_total"] < 0:
        return None
    return values


def build_chain() -> Any:
    """Create and return your LangChain chain once.

    The returned chain takes ``list[Path]`` of receipt images and returns
    ``{QUERY_1: "HK$...", QUERY_2: "HK$..."}``.
    """
    llm = ChatDeepSeek(
        model=VISION_MODEL,
        temperature=0,
        max_tokens=4096,  # reasoning model: leave room for thinking + JSON output
        timeout=180,
    )
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _EXTRACTION_SYSTEM),
            (
                "human",
                [
                    {"type": "text", "text": _EXTRACTION_USER},
                    {"type": "image_url", "image_url": {"url": "{image}"}},
                ],
            ),
        ]
    )
    extract_chain = extract_prompt | llm | StrOutputParser()

    def run(images: list[Path]) -> dict[str, str]:
        total_paid = Decimal("0")
        total_base = Decimal("0")
        for path in images:
            extraction = None
            for _attempt in range(_MAX_RETRIES):
                raw = extract_chain.invoke({"image": image_data_url(path)})
                extraction = _parse_extraction(raw)
                if extraction is not None:
                    break
                time.sleep(_RETRY_DELAY_SECONDS)
            if extraction is None:
                raise RuntimeError(f"Failed to extract amounts from {path.name}")
            total_paid += extraction["final_paid"]
            total_base += extraction["subtotal"] + extraction["discount_total"]
        return {
            QUERY_1: f"HK${total_paid:.2f}",
            QUERY_2: f"HK${total_base:.2f}",
        }

    return RunnableLambda(run)


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    """Run your chain and return one response for each exact query string.

    ``images`` contains every receipt in the selected folder. A valid return
    value looks like:

        {QUERY_1: "HK$123.40", QUERY_2: "HK$150.00"}

    Use the provided ``image_data_url(path)`` helper to put local images in
    multimodal human messages. LangChain's ``batch`` method is one simple way
    to process independent receipt-extraction prompts in parallel.
    """
    return chain.invoke(images)


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
