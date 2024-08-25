# from skimage.segmentation import flood_fill
import cv2
import numpy as np
from settings import Settings


# def forall_rotations(f, raw_x):
#     grid, r, c, color, cache = raw_x

#     cache_key = 'rot_grids'
#     if cache_key not in cache['grid']:
#         cache['grid'][cache_key] = [np.rot90(grid), np.rot90(grid, 2), np.rot90(grid, 3)]

#     if f(raw_x):
#         return True
#     for rot_grid in cache['grid'][cache_key]:
#         if f((rot_grid, r, c, color, cache)):
#             return True
#     return False

def out_of_bounds(grid, r, c):
    return (r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]))

def compare_colors(color1_program, color2_program, raw_x):
    return color1_program(raw_x) == color2_program(raw_x)

def get_color_from_action(raw_x):
    _, _, _, color, _ = raw_x
    return color

def get_color_from_cell(cell_program, raw_x):
    r, c = cell_program(raw_x)
    grid = raw_x[0]
    if out_of_bounds(grid, r, c):
        return None
    return grid[r, c]

def candidate_cell(raw_x):
    _, r, c, _, _ = raw_x
    return (r, c)

def mod_candidate_cell(raw_x):
    grid, r, c, _, _ = raw_x
    r = r % grid.shape[0]
    c = c % grid.shape[1]
    return (r, c)

def neighbor_cell(direction, raw_x):
    _, r, c, _, _ = raw_x
    return (r + direction[0], c + direction[1])

def mod_neighbor_cell(direction, raw_x):
    grid, r, c, _, _ = raw_x
    r = r % grid.shape[0]
    c = c % grid.shape[1]
    return (r + direction[0], c + direction[1])

def most_common_color(raw_x):
    # Other than black

    cache_key = 'most_common_color'
    cache = raw_x[-1]['grid']

    if cache_key not in cache or (not Settings.use_cache):
        grid = raw_x[0]
        counts = np.bincount(grid.flat)
        if len(counts) <= 1:
            out = 0
        else:
            max_color = np.argmax(counts[1:]) + 1
            if np.sum(counts[1:] == counts[max_color]) > 1:
                out = 0
            else:
                out = max_color
        cache[cache_key] = out
    return cache[cache_key]

def least_common_color(raw_x):
    # That is still present

    cache_key = 'least_common_color'
    cache = raw_x[-1]['grid']

    if cache_key not in cache or (not Settings.use_cache):
        grid = raw_x[0]
        counts = np.bincount(grid.flat)
        counts[counts == 0] = 1000
        min_color = np.argmin(counts)
        if np.sum(counts[1:] == counts[min_color]) > 1:
            out = 0
        else:
            out = min_color
        cache[cache_key] = out
    return cache[cache_key]

def search_for_color(direction, color_program, raw_x):
    target = color_program(raw_x)

    cache_key = 'search_for_color_{}_{}'.format(str(direction), target)
    cache = raw_x[-1]['cell']

    if cache_key not in cache or (not Settings.use_cache):
        grid, r, c, _, _ = raw_x

        for _ in range(31):
            r, c = (r + direction[0], c + direction[1])

            if out_of_bounds(grid, r, c):
                result = False
                break

            if grid[r, c] == target:
                result = True
                break
        else:
            result = False

        cache[cache_key] = result

    return cache[cache_key]

def inside(color_program, raw_x):
    target = color_program(raw_x)
    grid, r, c, _, _ = raw_x

    if out_of_bounds(grid, r, c):
        return 0

    cache = raw_x[-1]['grid']

    cache_key = 'inside_{}'.format(target)
    if cache_key not in cache or (not Settings.use_cache):
        matches = (grid == target)
        # 1. Find cells with immediate neighbors of target (these are inside)
        solidly_inside = np.zeros_like(matches)
        solidly_inside[1:-1, 1:-1] = matches[:-2, 1:-1] & matches[2:, 1:-1] & matches[1:-1, :-2] & matches[1:-1, 2:]

        # 2. Flood the outside
        origin = np.unravel_index(matches.argmin(), matches.shape)
        result = np.float32(matches)
        try:
            cv2.floodFill(result, None, origin, 1.)[1]
        except:
            # TODO: something weird is going on here
            result = np.ones_like(result)
        result = result.astype(bool)

        # 3. Find insulated zones that were not originally of color (these are inside)
        insulated = ~result & ~matches

        final = solidly_inside | insulated

        cache[cache_key] = final
    return cache[cache_key][r, c]

def row_is_divisible_by(num, cell_program, raw_x):
    r, _ = cell_program(raw_x)
    return (r % num) == 0

def col_is_divisible_by(num, cell_program, raw_x):
    _, c = cell_program(raw_x)
    return (c % num) == 0


# def at_action_cell(local_program, raw_x):
#     return local_program(raw_x)

# def shifted(direction, local_program, raw_x):
#     grid, r, c, color = raw_x
#     new_r = r + direction[0]
#     new_c = c + direction[1]
#     new_raw_x = (grid, new_r, new_c, color)
#     return local_program(new_raw_x)

# def cell_matches_color(raw_x):
#     grid, r, c, color = raw_x
#     if out_of_bounds(grid, r, c):
#         focus = None
#     else:
#         focus = grid[r, c]
#     return (focus == color)

# def color_is_value(value, raw_x):
#     _, _, _, color = raw_x
#     return color == value

# max_timeout = 31
# def scanning(direction, true_condition, false_condition, raw_x):
#     grid, r, c, color = raw_x

#     for _ in range(max_timeout):
#         r, c = (r + direction[0], c + direction[1])

#         if true_condition((grid, r, c, color)):
#             return True

#         if false_condition((grid, r, c, color)):
#             return False

#         # prevent infinite loops
#         if out_of_bounds(grid, r, c):
#             return False

#     return False
