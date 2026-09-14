"""Bounded video controls for the single-R9700 H3 workflow."""

RESOLUTIONS = {1: (576, 320), 2: (704, 384), 3: (864, 480)}


def video_settings(duration_seconds, resolution, aspect):
    if isinstance(duration_seconds, bool) or int(duration_seconds) != duration_seconds or not 5 <= duration_seconds <= 15:
        raise ValueError("Choose a duration from 5 to 15 whole seconds")
    if resolution not in RESOLUTIONS:
        raise ValueError("Choose resolution level 1, 2 or 3")
    width, height = RESOLUTIONS[resolution]
    if aspect == "Portrait":
        width, height = height, width
    elif aspect == "Square":
        width = height
    elif aspect != "Landscape":
        raise ValueError("Choose Landscape, Portrait or Square")
    requested_frames = int(duration_seconds) * 24
    frames = requested_frames + (5 - requested_frames) % 17
    return width, height, frames
