"""Scoring for the allergen baselines.

A prediction per row is {allergen: 'yes'|'no'|'unsure'}. For P/R/F1, 'yes' is a
positive; 'no' and 'unsure' are both "not flagged" (unsure = abstain = not flagged),
so an abstaining model loses recall rather than being let off the hook. The unsure
rate is reported separately so abstention is visible. Exact-match compares the set of
'yes' allergens against the gold set.
"""

from __future__ import annotations

from dataclasses import dataclass

from contains.labels import ALLERGENS


@dataclass
class PerClass:
    allergen: str
    support: int   # gold positives
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score(golds: list[set[str]], preds: list[dict[str, str]]) -> dict:
    """golds[i] = set of true allergens; preds[i] = {allergen: verdict}."""
    assert len(golds) == len(preds)
    pc = {a: PerClass(a, 0, 0, 0, 0) for a in ALLERGENS}
    exact = 0
    unsure_cells = 0
    n = len(golds)
    for gold, pred in zip(golds, preds):
        pred_yes = {a for a in ALLERGENS if pred.get(a) == "yes"}
        for a in ALLERGENS:
            v = pred.get(a, "unsure")
            if v == "unsure":
                unsure_cells += 1
            in_gold = a in gold
            if v == "yes" and in_gold:
                pc[a].tp += 1
            elif v == "yes" and not in_gold:
                pc[a].fp += 1
            elif v != "yes" and in_gold:
                pc[a].fn += 1
            if in_gold:
                pc[a].support += 1
        if pred_yes == gold:
            exact += 1
    classes = [pc[a] for a in ALLERGENS]
    macro_p = sum(c.precision for c in classes) / len(classes)
    macro_r = sum(c.recall for c in classes) / len(classes)
    macro_f1 = sum(c.f1 for c in classes) / len(classes)
    return {
        "n": n,
        "per_class": classes,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "exact_match": exact / n if n else 0.0,
        "unsure_rate": unsure_cells / (n * len(ALLERGENS)) if n else 0.0,
    }


def format_report(name: str, res: dict, extra: dict | None = None) -> str:
    lines = [f"### {name}  (n={res['n']})"]
    lines.append("```")
    lines.append(f"{'allergen':22s} {'supp':>5} {'prec':>6} {'rec':>6} {'f1':>6}")
    for c in res["per_class"]:
        lines.append(f"{c.allergen:22s} {c.support:>5} {c.precision:>6.3f} "
                     f"{c.recall:>6.3f} {c.f1:>6.3f}")
    lines.append(f"{'MACRO':22s} {'':>5} {res['macro_precision']:>6.3f} "
                 f"{res['macro_recall']:>6.3f} {res['macro_f1']:>6.3f}")
    lines.append("")
    lines.append(f"exact_match={res['exact_match']:.3f}  unsure_rate={res['unsure_rate']:.3f}")
    if extra:
        for k, v in extra.items():
            lines.append(f"{k}={v}")
    lines.append("```")
    return "\n".join(lines)
