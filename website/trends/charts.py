
import math


def _axis_step(maximum):
    rough_step = max(1, math.ceil(maximum / 4))
    magnitude = 10 ** math.floor(math.log10(rough_step))
    for multiplier in (1, 2, 5, 10):
        step = magnitude * multiplier
        if step >= rough_step:
            return step
    return magnitude * 10


def publication_timeline(years):
    ordered = sorted(years, key=lambda item: int(item["year"]))
    if not ordered:
        return {"points": [], "ticks": [], "line": ""}

    maximum = max((int(item.get("count") or 0) for item in ordered), default=0)
    axis_step = _axis_step(maximum)
    axis_maximum = max(axis_step, math.ceil(maximum / axis_step) * axis_step)
    left, right, top, bottom = 70, 592, 35, 165
    step = (right - left) / max(1, len(ordered) - 1)
    ticks = [
        {
            "value": value,
            "y": round(bottom - value / axis_maximum * (bottom - top)),
        }
        for value in range(0, axis_maximum + 1, axis_step)
    ]
    points = []
    for index, item in enumerate(ordered):
        count = max(0, int(item.get("count") or 0))
        year = int(item["year"])
        points.append({
            "year": year,
            "x": round(left + index * step),
            "y": round(bottom - count / axis_maximum * (bottom - top)),
            "tooltip": (
                f"{'Найдено не менее' if not item.get('exhaustive', True) else 'Найдено'} "
                f"{count} публикаций за {year} год"
            ),
        })
    return {
        "points": points,
        "ticks": ticks,
        "line": " ".join(f"{point['x']},{point['y']}" for point in points),
    }
