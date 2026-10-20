import math
import statistics


def geometric_mean(values):
    if not values or any(v <= 0 for v in values):
        raise ValueError("Geometric means require positive, nonempty observations")
    return math.exp(statistics.mean(math.log(v) for v in values))


def summarize(rows):
    if not rows:
        raise ValueError("No scored prompts")
    rounds = sum(row["draft"]["rounds"] for row in rows)
    if not rounds:
        raise ValueError("No verification rounds")
    tau = sum(row["draft"]["tau"] * row["draft"]["rounds"] for row in rows) / rounds
    ar_ms = statistics.mean(row["ar"]["decode_ms_token"] for row in rows)
    draft_ms = statistics.mean(row["draft"]["decode_ms_token"] for row in rows)
    return {
        "prompts": len(rows),
        "rounds": rounds,
        "tau": tau,
        "speedup": ar_ms / draft_ms,
        "ar_ms_token": ar_ms,
        "draft_ms_token": draft_ms,
    }
