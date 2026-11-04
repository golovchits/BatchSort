"""Book a compartment for a recipient's parcels.

Compartments come from data/mock/world.yaml. Booking one marks it taken in memory, so two
parcels in the same batch never get the same empty compartment.
"""
from src.parcels import COMPARTMENTS, fits

ORDER = list(COMPARTMENTS["sizes"])  # smallest class first


def stack(parcels_cm):
    """Parcels piled in one compartment, each lying on its biggest face, as one block."""
    flat = [sorted(p, reverse=True) for p in parcels_cm]
    return [max(f[0] for f in flat), max(f[1] for f in flat), sum(f[2] for f in flat)]


def book(parcels_cm, recipient_id, building):
    """Book one compartment for all of a recipient's parcels on the mat.

    Returns (compartment, certain, reason). certain is False when the parcels fit only without
    the margin: amber, the worker checks. compartment is None for "too big" or "full": red.
    """
    margin = COMPARTMENTS["margin_cm"]
    # 1. The recipient's own compartment, if everything stacked still fits for certain.
    for c in building["compartments"]:
        if c["occupied_by"] == recipient_id and fits(stack(c["contents_cm"] + parcels_cm), c["inner_cm"], margin):
            c["contents_cm"] += [sorted(p, reverse=True) for p in parcels_cm]
            return c, True, "added to their compartment"
    # 2. The smallest free compartment it fits for certain, else the smallest it fits at all.
    block = stack(parcels_cm)
    free = sorted((c for c in building["compartments"] if c["occupied_by"] is None), key=lambda c: ORDER.index(c["size"]))
    for extra, certain in ((margin, True), (0, False)):
        for c in free:
            if fits(block, c["inner_cm"], extra):
                c["occupied_by"] = recipient_id
                c["contents_cm"] = [sorted(p, reverse=True) for p in parcels_cm]
                return c, certain, "free compartment"
    # 3. Nothing free fits. Too big for every class, or the building is full.
    if not any(fits(block, inner) for inner in COMPARTMENTS["sizes"].values()):
        return None, False, "too big"
    return None, False, "full"
