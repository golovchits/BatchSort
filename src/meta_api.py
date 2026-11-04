"""Thin client for the Meta Model API. Muse Spark reads labels, SAM 3.1 finds objects."""
import base64
import json
import os
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format

load_dotenv()

BASE_URL = "https://api.meta.ai/v1"
SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"

# The fields we want from every label. Spark must fill each one, with "" when it can't.
LABEL_FIELDS = ["recipient_name", "street", "postcode", "city", "tracking_number"]
# Not on the label, but in the same photo: a soft parcel can be squashed a little to fit.
PACKAGING = {"type": "string", "enum": ["soft", "rigid"]}
LABEL_SCHEMA = {
    "type": "object",
    "properties": {**{field: {"type": "string"} for field in LABEL_FIELDS}, "packaging": PACKAGING},
    "required": LABEL_FIELDS + ["packaging"],
    "additionalProperties": False,
}

LABEL_INSTRUCTIONS = (
    "You read parcel shipping labels. Copy the recipient's details exactly as printed, "
    "spelling mistakes included. Do not correct anything. Ignore the sender. "
    "The name line may start with a short tag and a code before the name: copy the whole line. "
    "The street field holds the street and house number together. "
    "Packaging: soft for a bag, mailer or anything that gives when pressed, rigid for a box. "
    "Use an empty string for anything not printed or not readable."
)


def _headers():
    return {"Authorization": f"Bearer {os.environ['META_API_KEY']}"}


def _data_url(image_path):
    """The API takes images as text: the file's bytes in base64, inside a data URL."""
    data = base64.b64encode(Path(image_path).read_bytes()).decode()
    mime = "image/png" if str(image_path).lower().endswith(".png") else "image/jpeg"
    return f"data:{mime};base64,{data}"


def read_label(image_path):
    """Ask Muse Spark for the label fields. Returns (fields dict, token usage)."""
    body = {
        "model": SPARK_MODEL,
        "reasoning_effort": "minimal",  # low gave the same 40 dev labels in twice the time
        "messages": [
            {"role": "developer", "content": LABEL_INSTRUCTIONS},
            {"role": "user", "content": [
                {"type": "text", "text": "Read this label."},
                {"type": "image_url", "image_url": {"url": _data_url(image_path)}},
            ]},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "label", "schema": LABEL_SCHEMA},
        },
    }
    response = httpx.post(f"{BASE_URL}/chat/completions", headers=_headers(), json=body, timeout=60)
    response.raise_for_status()
    reply = response.json()
    return json.loads(reply["choices"][0]["message"]["content"]), reply["usage"]


def segment(image_path, prompt):
    """Ask SAM 3.1 for every object matching a short noun phrase.

    Returns one dict per object: its box (x1, y1, x2, y2 in image pixels) and its mask, a
    True/False array covering just that box. Meta's own parser (meta-sam-parser) decodes
    SAM's text format, which its docs don't spell out.
    """
    body = {
        "model": SAM_MODEL,
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _data_url(image_path)},
        ]}],
    }
    response = httpx.post(f"{BASE_URL}/responses", headers=_headers(), json=body, timeout=120)
    response.raise_for_status()
    text = "".join(
        part["text"]
        for item in response.json()["output"]
        for part in item.get("content", [])
        if part["type"] == "output_text"
    )
    parser = image_segmentation_format().create_parser()
    parser.push(text + "\n")
    records = parser.finish(CompletedOutcome()).result.records
    objects = {}
    for record in records:
        if record.kind == "box":
            objects.setdefault(record.object_id, {}).update(
                x1=int(record.left), y1=int(record.top), x2=int(record.right), y2=int(record.bottom))
        elif record.kind == "mask":
            raster = np.frombuffer(decode_mask_to_raster(record.mask), np.uint8)
            objects.setdefault(record.object_id, {}).update(
                mask=raster.reshape(record.mask.height, record.mask.width).astype(bool),
                mask_left=int(record.bounds.left), mask_top=int(record.bounds.top))
    return [o for o in objects.values() if "x1" in o and "mask" in o]
