"""Make shipping labels for a fictional carrier, Northwind Parcel, from data/mock/world.yaml.

    python -m tools.make_labels

Writes, all in data/mock/:
  labels_dev.pdf, labels_test.pdf   to print: 7 dev and 6 test labels, A6, 4 to an A4 sheet
  labels_dev.yaml, labels_test.yaml  what each label says, who it is really for, and any error
  label_images/                      40 more dev labels as images only, for tuning the match

Every label goes to the same hub address. The name line is the network's tag, the recipient's
5-letter code and their last name, as a shop prints it: "HVL KRTMS Velthoven".

About 1 in 4 labels carry a deliberate error, as real ones do: a misspelled last name, one
letter of the code mistyped, or no code at all. No code reads labels_test.yaml before the
evaluation at step 11. Each label shows its parcel ID small in a corner, only so we stick it
on the right parcel.
"""
import random
from pathlib import Path

import yaml
from barcode import Code128
from barcode.writer import ImageWriter
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

SEED = 27
OUT = Path("data/mock")
SENDERS = ["Kroonsteen Outdoor, Industrieweg 4, 3011 AB Velsenhof",
           "Lindegaard Books, Postbus 112, 5201 CK Overbrug",
           "Mosvelt Home, Fabriekstraat 18, 7511 JL Dorrebeek"]
ERRORS = ["misspelled_name", "mistyped_code", "missing_code"]
NETWORK_TAG = "HVL"  # Havendam Lockers, the fictional locker network
HUB = {"street": "Kadeweg 1", "town": "1041 HK  Havendam"}
CODE_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"
WIDTH, HEIGHT = 1181, 1654  # 100 x 140 mm at 300 dpi
FONT = {size: ImageFont.load_default(size=size) for size in (34, 46, 64, 80)}


def misspell(name, rng):
    """Swap two neighbouring letters, the commonest typing slip."""
    i = rng.randrange(1, len(name) - 2)
    return name[:i] + name[i + 1] + name[i] + name[i + 2:]


def label_text(recipient, error, rng):
    """The three recipient lines as printed, with the error (if any) applied."""
    code, last = recipient["code"], recipient["last"]
    if error == "misspelled_name":
        last = misspell(last, rng)
    if error == "mistyped_code":
        i = rng.randrange(5)
        code = code[:i] + rng.choice(CODE_LETTERS.replace(code[i], "")) + code[i + 1:]
    name = f"{NETWORK_TAG} {last}" if error == "missing_code" else f"{NETWORK_TAG} {code} {last}"
    return {"name": name, **HUB}


def render(label):
    """One label as a 300 dpi image."""
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH - 1, HEIGHT - 1], outline="black", width=6)
    draw.text((50, 40), "NORTHWIND PARCEL", font=FONT[80], fill="black")
    draw.text((50, 140), "Standard parcel", font=FONT[46], fill="black")
    draw.line([40, 215, WIDTH - 40, 215], fill="black", width=4)
    draw.text((50, 245), f"From: {label['sender']}", font=FONT[34], fill="black")
    draw.text((50, 340), "TO:", font=FONT[46], fill="black")
    for i, line in enumerate(label["text"].values()):
        draw.text((50, 410 + i * 95), line, font=FONT[64], fill="black")
    code = Code128(label["tracking"], writer=ImageWriter()).render({"module_height": 22, "write_text": False, "quiet_zone": 2})
    code = code.resize((WIDTH - 160, 360))
    image.paste(code, (80, 1080))
    draw.text((80, 1460), label["tracking"], font=FONT[64], fill="black")
    draw.text((WIDTH - 140, 1580), label["parcel"], font=FONT[34], fill="black")
    return image


def make_labels(recipients, parcels, errors_wanted, rng, printed=True):
    labels = []
    error_slots = set(rng.sample(range(len(parcels)), errors_wanted))
    for i, (parcel, recipient) in enumerate(zip(parcels, recipients)):
        error = rng.choice(ERRORS) if i in error_slots else None
        labels.append({
            "parcel": parcel,
            "printed": printed,
            "tracking": f"NW{rng.randint(10**9, 10**10 - 1)}",
            "recipient": recipient["id"],
            "sender": rng.choice(SENDERS),
            "text": label_text(recipient, error, rng),
            "error": error,
        })
    return labels


def write_pdf(labels, path):
    """4 labels to an A4 page, each 100 x 140 mm, with room around it for printer margins."""
    page = canvas.Canvas(str(path), pagesize=A4)
    for i, label in enumerate(labels):
        if i and i % 4 == 0:
            page.showPage()
        col, row = i % 2, (i % 4) // 2
        page.drawImage(ImageReader(render(label)), (2.5 + col * 105) * mm, (151 - row * 148.5) * mm, 100 * mm, 140 * mm)
    page.save()


if __name__ == "__main__":
    rng = random.Random(SEED)
    world = yaml.safe_load(open(OUT / "world.yaml"))
    people = rng.sample(world["recipients"], 7 + 40 + 6)  # nobody gets two labels
    dev = make_labels(people[:7], [f"D{i:02d}" for i in range(1, 8)], 2, rng)
    digital = make_labels(people[7:47], [f"X{i:02d}" for i in range(1, 41)], 10, rng, printed=False)
    test = make_labels(people[47:], [f"T{i:02d}" for i in range(1, 7)], 2, rng)

    (OUT / "label_images").mkdir(exist_ok=True)
    for label in digital:
        render(label).save(OUT / "label_images" / f"{label['parcel']}.png")
    write_pdf(dev, OUT / "labels_dev.pdf")
    write_pdf(test, OUT / "labels_test.pdf")
    for name, labels in (("labels_dev.yaml", dev + digital), ("labels_test.yaml", test)):
        with open(OUT / name, "w") as f:
            f.write("# Fictional. Made by tools/make_labels.py with a fixed seed.\n")
            yaml.safe_dump(labels, f, sort_keys=False, allow_unicode=True)
    print(f"Printed dev labels: {len(dev)}, test: {len(test)}, image-only dev labels: {len(digital)}")
    print(f"With errors: dev {sum(l['error'] is not None for l in dev)}, test {sum(l['error'] is not None for l in test)}, "
          f"images {sum(l['error'] is not None for l in digital)}")
