import numpy as np

from matplotlib.lines import Line2D

Y_SCALE = np.sqrt(3) / 2.0

# code adapted from Python-Ternary at https://github.com/marcharper/python-ternary


def normalize_tick_formats(tick_formats):
    if type(tick_formats) == dict:
        return tick_formats
    if tick_formats is None:
        s = '%d'
    elif type(tick_formats) == str:
        s = tick_formats
    else:
        raise TypeError('tick_formats must be a dictionary of strings, a string, or None.')
    return {'b': s, 'l': s, 'r': s}


def draw_ticks(ax, ticks=None, axis='b',
               offset=0.0, fontsize=8, tick_formats=None, clockwise=False, axes_colors=None, **kwargs):

    axis = axis.lower()
    valid_axis_chars = {'l', 'r', 'b'}
    axis_chars = set(axis)
    if not axis_chars.issubset(valid_axis_chars):
        raise ValueError("'axis' must be some combination of 'l', 'r', and 'b'")

    num_ticks = len(ticks)
    step = 1.0 / (num_ticks - 1)
    locations = np.arange(0, 1.0 + step, step)

    tick_formats = normalize_tick_formats(tick_formats)

    # Default color: black
    if axes_colors is None:
        axes_colors = dict()
    for _axis in valid_axis_chars:
        if _axis not in axes_colors:
            axes_colors[_axis] = 'black'

    if 'r' in axis:
        for index, i in enumerate(locations):
            loc1 = (1.0 - i, i, 0)
            if clockwise:
                # Right parallel
                loc2 = (1.0 - i, i + offset, 0)
                text_location = (1.0 - i, i + 2 * offset, 0)
                tick = ticks[-(index + 1)]
            else:
                # Horizontal
                loc2 = (1.0 - i + offset, i, 0)
                text_location = (1.0 - i + 3.1 * offset, i - 0.5 * offset, 0)
                tick = ticks[index]
            draw_line(ax, loc1, loc2, color=axes_colors['r'], **kwargs)
            x, y = project_point(text_location)
            if isinstance(tick, str):
                s = tick
            else:
                s = tick_formats['r'] % tick
            ax.text(x, y, s, horizontalalignment='center',
                    color=axes_colors['r'], fontsize=fontsize)

    if 'l' in axis:
        for index, i in enumerate(locations):
            loc1 = (0, i, 0)
            if clockwise:
                # Horizontal
                loc2 = (-offset, i, 0)
                text_location = (-2 * offset, i - 0.5 * offset, 0)
                tick = ticks[index]
            else:
                # Right parallel
                loc2 = (-offset, i + offset, 0)
                text_location = (-2 * offset, i + 1.5 * offset, 0)
                tick = ticks[-(index + 1)]
            draw_line(ax, loc1, loc2, color=axes_colors['l'], **kwargs)
            x, y = project_point(text_location)
            if isinstance(tick, str):
                s = tick
            else:
                s = tick_formats['l'] % tick
            ax.text(x, y, s, horizontalalignment='center',
                    color=axes_colors['l'], fontsize=fontsize)

    if 'b' in axis:
        for index, i in enumerate(locations):
            loc1 = (i, 0, 0)
            if clockwise:
                # Right parallel
                loc2 = (i + offset, -offset, 0)
                text_location = (i + 3 * offset, -3.5 * offset, 0)
                tick = ticks[-(index + 1)]
            else:
                # Left parallel
                loc2 = (i, -offset, 0)
                text_location = (i + 0.5 * offset, -3.5 * offset, 0)
                tick = ticks[index]
            draw_line(ax, loc1, loc2, color=axes_colors['b'], **kwargs)
            x, y = project_point(text_location)
            if isinstance(tick, str):
                s = tick
            else:
                s = tick_formats['b'] % tick
            ax.text(x, y, s, horizontalalignment='center',
                    color=axes_colors['b'], fontsize=fontsize)


def draw_line(ax, p1, p2, permutation=None, **kwargs):
    pp1 = project_point(p1, permutation=permutation)
    pp2 = project_point(p2, permutation=permutation)
    ax.add_line(Line2D((pp1[0], pp2[0]), (pp1[1], pp2[1]), **kwargs))


def project_point(p, permutation=None):
    permuted = permute_point(p, permutation=permutation)

    a = permuted[0]
    b = permuted[1]
    x = a + b / 2.0
    y = Y_SCALE * b

    return np.array([x, y])


def permute_point(p, permutation=None):
    if not permutation:
        return p
    return [p[int(permutation[i])] for i in range(len(p))]


def set_title(ax, title, **kwargs):
    ax.set_title(title, **kwargs)


def set_label(ax, label, position, rotation, **kwargs):
    transform = ax.transAxes
    x, y = project_point(position)

    # calculate the new angle
    position = np.array([x, y])
    new_rotation = ax.transData.transform_angles(np.array((rotation,)), position.reshape((1, 2)))[0]
    text = ax.text(x, y, label, rotation=new_rotation,
                   transform=transform, horizontalalignment='center',
                   **kwargs)
    text.set_rotation_mode('anchor')


def set_left_axis_label(ax, label, position=None, rotation=60, offset=0.08, **kwargs):
    if not position:
        position = (-offset, 0.6, 0.4)
    set_label(ax, label, position, rotation, **kwargs)


def set_right_axis_label(ax, label, position=None, rotation=-60, offset=0.08, **kwargs):
    if not position:
        position = (0.4 + offset, 0.6, 0)
    set_label(ax, label, position, rotation, **kwargs)


def set_bottom_axis_label(ax, label, position=None, rotation=0, offset=0.08, **kwargs):
    if not position:
        position = (0.5, -offset / 2.0, 0.5)
    set_label(ax, label, position, rotation, **kwargs)


def set_right_corner_label(ax, label, position=None, rotation=0, offset=0.08, **kwargs):
    if not position:
        position = (1.0, offset / 2.0, 0.0)
    set_label(ax, label, position, rotation, **kwargs)


def set_left_corner_label(ax, label, position=None, rotation=0, offset=0.08, **kwargs):
    if not position:
        position = (-offset / 2.0, offset / 2.0, 0.0)
    set_label(ax, label, position, rotation, **kwargs)


def set_top_corner_label(ax, label, position=None, rotation=0, offset=0.16, **kwargs):
    if not position:
        position = (-offset / 2.0, 1.0 + offset, 0.0)
    set_label(ax, label, position, rotation, **kwargs)
