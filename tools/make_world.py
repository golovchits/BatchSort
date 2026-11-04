"""Make the fictional world: 6 locker buildings in the made-up town of Havendam, 200 recipients,
and the compartments, some already holding parcels.

As in a real parcel locker network, a recipient never gives their home address to a shop. They
give the network's hub address, with their own 5-letter code before their last name. The hub
routes every parcel by that code.

    python -m tools.make_world        # writes data/mock/world.yaml

Every name and address is invented from the lists below with a fixed random seed, so the
world is the same on every run. Some names repeat within a building on purpose: a real
network has those, and they should come out amber, never green.
"""
import random

import yaml

SEED = 26
FIRST = ["Mira", "Joris", "Anouk", "Bas", "Lotte", "Teun", "Fenna", "Daan", "Isa", "Ruben", "Noor", "Sem",
         "Elif", "Milan", "Yara", "Thijs", "Sanne", "Koen", "Lina", "Jesse", "Roos", "Stijn", "Amira", "Luuk",
         "Floor", "Hidde", "Saar", "Bram", "Nina", "Mats"]
LAST = ["Velthoven", "Brakel", "Doornbos", "Esvelt", "Haverkamp", "Kooistra", "Lindeboom", "Meerhof",
        "Oosterbaan", "Plasman", "Rietveld", "Schuring", "Terpstra", "Uiterwijk", "Veldkamp", "Wijnhoven",
        "Zandbergen", "Achterberg", "Bosgra", "Dijkhuis", "Grevelink", "Hoekstein", "Kampers", "Leeuwerik"]
STREETS = ["Ooievaarskade", "Zilverreigerlaan", "Brandarisweg", "Kroosgracht", "Wilgenhof", "Meeuwenpad"]
TOWN = "Havendam"
CODE_LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ"  # no I or O, easily misread as 1 or 0
PREFERENCES = ["leave beside the lockers", "hold at hub", "deliver tomorrow"]
# A building is named by its delivery area (a letter) and its number within that area.
BUILDINGS = ["A1", "A2", "B1", "C1", "C2", "D1"]

# Compartments per building, and their inside size in cm. Fictional, matching configs/default.yaml.
COMPARTMENTS = {"XS": (6, [60, 30, 10]), "S": (8, [60, 35, 16]), "M": (6, [60, 40, 22]), "L": (2, [60, 50, 40])}


def make_world():
    rng = random.Random(SEED)
    buildings = []
    for building_id, street in zip(BUILDINGS, STREETS):
        buildings.append({"id": building_id, "name": f"{TOWN} {street}", "street": street, "compartments": []})

    recipients, codes = [], set()
    for r in range(1, 201):
        building = buildings[(r - 1) % 6]
        code = "".join(rng.choices(CODE_LETTERS, k=5))
        while code in codes:
            code = "".join(rng.choices(CODE_LETTERS, k=5))
        codes.add(code)
        recipients.append({
            "id": f"R{r:03d}",
            "code": code,
            "first": rng.choice(FIRST),
            "last": rng.choice(LAST),
            "building": building["id"],
            "preference": rng.choice(PREFERENCES) if rng.random() < 0.1 else None,
        })
    # Same full name twice in one building, told apart only by their codes: 3 pairs on purpose
    # (random picks add a few more).
    for original, copy in [(0, 6), (13, 19), (40, 46)]:
        recipients[copy]["first"], recipients[copy]["last"] = recipients[original]["first"], recipients[original]["last"]

    for building in buildings:
        locals_ = [r for r in recipients if r["building"] == building["id"]]
        for size, (count, inner) in COMPARTMENTS.items():
            for _ in range(count):
                number = len(building["compartments"]) + 1  # plain numbers, as on a real locker wall
                compartment = {"id": f"{building['id']}-{number}", "number": number, "size": size, "inner_cm": list(inner),
                               "occupied_by": None, "contents_cm": []}
                if rng.random() < 0.25:  # a quarter already hold a parcel from an earlier delivery
                    compartment["occupied_by"] = rng.choice(locals_)["id"]
                    compartment["contents_cm"] = [[rng.randint(15, 35), rng.randint(10, 25), rng.randint(3, inner[2] - 2)]]
                building["compartments"].append(compartment)
    return {"town": TOWN, "buildings": buildings, "recipients": recipients}


if __name__ == "__main__":
    world = make_world()
    with open("data/mock/world.yaml", "w") as f:
        f.write("# Fictional. Made by tools/make_world.py with a fixed seed. Do not edit by hand.\n")
        yaml.safe_dump(world, f, sort_keys=False, allow_unicode=True)
    used = sum(c["occupied_by"] is not None for b in world["buildings"] for c in b["compartments"])
    total = sum(len(b["compartments"]) for b in world["buildings"])
    print(f"Wrote data/mock/world.yaml: {len(world['buildings'])} buildings, {len(world['recipients'])} recipients, "
          f"{total} compartments, {used} in use")
