"""Reject scenery that does not follow independently verified score motion.

Masks only remove observed ink; they never reconstruct notation. Sparse or
stationary evidence leaves the original image intact.
"""

import cv2
import numpy as np

from .clean import observed_staff, stroke_response
from .geometry import staff_groups
from .support import check_cancel


def motion_background_mask(frames, positions, polarity, cancel_event=None):
    """Compare spatially separated observations in a native-resolution atlas.

    Equal screen positions get one vote, regardless of how long playback
    pauses there. Otherwise a stationary guitar can outvote scrolling notes.
    Small registration/antialiasing differences get a one-pixel allowance.
    Atlas values: 0 retain/unknown, 1 ambiguous absence, 2 strict absence.
    """
    positions = np.asarray(positions, dtype=float)
    if (len(frames) < 3 or len(positions) != len(frames)
            or not np.isfinite(positions).all() or np.any(np.diff(positions) < 0)):
        return None
    groups = observed_staff(frames[0], polarity)
    if not groups:
        return None
    space = float(np.median([np.median(np.diff(g)) for g in groups]))
    separation = max(8, space * 4)
    positions = positions - positions[0]
    selected = [0]
    for i in range(1, len(frames)):
        if positions[i] - positions[selected[-1]] >= separation:
            selected.append(i)
    if len(selected) < 3:
        return None
    height, width = frames[0].shape[:2]
    shape = (height, int(np.ceil(width + positions[-1])))
    votes = np.zeros(shape, np.uint16)
    coverage = np.zeros(shape[1], np.uint16)
    kernel = max(3, round(space / 3) | 1)
    for i in selected:
        check_cancel(cancel_event)
        # Repeated observations at the same position can recover briefly
        # occluded notes, but cannot add extra independent votes.
        support = np.zeros((height, width), np.uint8)
        for j in np.where(abs(positions - positions[i]) < 1)[0]:
            response = stroke_response(frames[j], polarity, kernel=kernel)
            support |= (response >= 8).astype(np.uint8)
        support = cv2.dilate(support, np.ones((3, 3), np.uint8))
        start = int(round(positions[i]))
        # Do not treat image padding or a source-edge interpolation fringe
        # as evidence of absence.
        votes[:, start + 2:start + width - 2] += support[:, 2:-2]
        coverage[start + 2:start + width - 2] += 1
    mask = ((coverage[None, :] >= 2)
            & (votes * 2 <= coverage[None, :])).astype(np.uint8)
    # Distinguish an ambiguous majority from unanimous absence in at least
    # two independent views, or absence at >= 75% of at least four positions.
    # Pauses never add votes.
    strict = (((coverage[None, :] >= 2) & (votes == 0))
              | ((coverage[None, :] >= 4) & (votes * 4 <= coverage[None, :])))
    mask[strict] = 2
    return mask


def remove_motion_background(page, mask, offset=0):
    """Apply a verified atlas mask, preserving vulnerable notation strokes.

    offset is the page's horizontal location in upscaled atlas coordinates.
    Input arrays stay unchanged, so failed seam checks are non-destructive.
    """
    if mask is None:
        return page
    scale = page.shape[0] / mask.shape[0]
    transform = np.float32([[scale, 0, (scale - 1) / 2 - offset],
                           [0, scale, (scale - 1) / 2]])
    evidence = cv2.warpAffine(mask, transform, page.shape[::-1],
                             flags=cv2.INTER_NEAREST, borderValue=0)
    remove = evidence > 0
    if not remove.any():
        return page
    groups = staff_groups(page)
    if not groups:
        return page
    ink = (page < 245).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    protect = np.zeros(count, bool)
    notation_zone = np.zeros(page.shape, bool)
    for lines in groups:
        space = float(np.median(np.diff(lines)))
        # Keep fret numbers, articulation dots, ties and the lower rhythm
        # envelope regardless of absence votes. Only detached outer scenery
        # can lose the conservative small-component exemption below.
        a = max(0, round(lines[0] - space * .65))
        b = min(page.shape[0], round(lines[-1] + space * 2.5) + 1)
        notation_zone[a:b] = True
        # Horizontal strings are often faint over light video backgrounds.
        for line in lines:
            y, radius = round(line), max(1, round(scale))
            remove[max(0, y - radius):y + radius + 1] = False
        # Keep entire detached glyphs, dots and short technique marks, even
        # when a transient occlusion makes their motion evidence inconclusive.
        for i in range(1, count):
            x, y, w, h, _ = stats[i]
            if (w <= space * 3 and h <= space * 1.7
                    and y >= lines[0] - 4 * space
                    and y + h <= lines[-1] + 4 * space):
                protect[i] = True
        vertical = cv2.morphologyEx(
            ink, cv2.MORPH_OPEN, np.ones((max(3, round(space * .85)) | 1, 1), np.uint8)
        )
        n, parts, bounds, _ = cv2.connectedComponentsWithStats(vertical, 8)
        keep = np.zeros(n, np.uint8)
        for i in range(1, n):
            y, h = bounds[i, cv2.CC_STAT_TOP], bounds[i, cv2.CC_STAT_HEIGHT]
            keep[i] = y <= lines[-1] + space * .3 and y + h >= lines[0] - space * .3
        radius = max(1, round(scale))
        stems = cv2.dilate(keep[parts], np.ones((radius, radius), np.uint8))
        remove[stems > 0] = False
        stem_guard = max(1, round(space * .35))
        notation_zone |= cv2.dilate(stems, np.ones((2 * stem_guard + 1,
                                                    2 * stem_guard + 1), np.uint8)) > 0
    remove[protect[labels]] = False
    result = page.copy()
    result[remove] = 255
    # Assess whole surviving components; never shave pixels off a letter or
    # remove a mark solely because it is small. One supported edge, proximity
    # to notation, or insufficient independent views leaves it untouched.
    n, pieces, bounds, _ = cv2.connectedComponentsWithStats((result < 245).astype(np.uint8), 8)
    discard = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, _ = bounds[i]
        region = np.s_[y:y + h, x:x + w]
        component = pieces[region] == i
        if (not notation_zone[region][component].any()
                and np.all(evidence[region][component] == 2)):
            discard[i] = True
    result[discard[pieces]] = 255
    return result
