"""Canonical sensor layouts shared by training, evaluation, and live demos."""

SENSOR_LAYOUTS = {
    # left wrist, right pocket/thigh, head
    "lw_rp_h": (0, 3, 4),
}

DEFAULT_LAYOUT = "lw_rp_h"


def sensor_ids(layout: str = DEFAULT_LAYOUT) -> list[int]:
    try:
        return list(SENSOR_LAYOUTS[layout])
    except KeyError as error:
        available = ", ".join(sorted(SENSOR_LAYOUTS))
        raise ValueError(f"Unknown sensor layout '{layout}'. Available: {available}") from error
