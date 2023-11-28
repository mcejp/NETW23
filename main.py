import heapq
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tilevision.path_util import circle, grid, line, polyline, rectangle_centered, triangle
from tilevision.tilevision import Label, Path as TvPath, TV


@dataclass
class Town:
    id: int
    x: int
    y: int
    pop: int


######

ENABLE_PLOTS = False

W = 50
H = 50
# POP_SCALE = 500
POP_MEAN = 5
POP_SIGMA = 1.4
MAX_TOWNS = 80
MIN_TOWN_DIST = 0
TOWN_DIST_FACTOR = 0.1
K = 1.7     # exponent for distance when weighing city connections;
            #   decrease to favor connections between further cities if they're large enough
            # we found 1 (linear) too little (ignoring close neighbor in favor of far cities)
            # we found 2 (quadratic) too much

# 0 = no change of throughput with volume; 1 = 2x volume -> 2x throughput
# increase to encourage re-use of existing edges
VOLUME_SPEEDUP_EXPONENT = 0.35

BUILD_COST = 0.4
TURN_PENALTY = 0.05

THROUGHPUT_LIMIT = 5000


if ENABLE_PLOTS:
    dir = Path(f'run_{datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}_v2')
    dir.mkdir()
    os.chdir(dir)

    matplotlib.rcParams['figure.figsize'] = (9, 9)

rg = np.random.default_rng(seed=0)

# Spawn a lake
xs = np.arange(0, W)
ys = np.arange(0, H)
yy, xx = np.meshgrid(xs, ys)
height = np.full(shape=(H, W), fill_value=1, dtype=float)

for i in range(3):
    x1 = rg.integers(W)
    y1 = rg.integers(H)

    COUNT = 5

    for j in range(COUNT):
        SPREAD = 4
        x = x1 + rg.normal(0, SPREAD)
        y = y1 + rg.normal(0, SPREAD)

        height = height - SPREAD / COUNT / np.sqrt(np.power(xx - x, 2) + np.power(yy - y, 2))

for i in range(1):
    x1 = rg.integers(W)
    y1 = rg.integers(H)

    COUNT = 12

    for j in range(COUNT):
        SPREAD = 4
        x = x1 + rg.normal(0, SPREAD)
        y = y1 + rg.normal(0, SPREAD)

        height = height + 15 * SPREAD / COUNT / (np.power(xx - x, 2) + np.power(yy - y, 2))

MOUNTAIN_THR = 5

lake_list = []
for y, x in np.ndindex(height.shape):
    if height[y, x] < 0:
        lake_list.append((x, y))

mountain_list = []
for y, x in np.ndindex(height.shape):
    if height[y, x] > MOUNTAIN_THR:
        mountain_list.append((x, y))

"""
- Place towns using Poisson sampling
  - For each, select population from a log-normal distribution
"""

towns = []

for i in range(1000):
    if len(towns) >= MAX_TOWNS:
        break

    x = rg.integers(W)
    y = rg.integers(H)
    pop = math.ceil(rg.lognormal(POP_MEAN, POP_SIGMA))

    if height[y, x] < 0 or height[y, x] > MOUNTAIN_THR:
        continue

    too_close = False

    for t in towns:
        min_dist = MIN_TOWN_DIST + (math.sqrt(t.pop) + math.sqrt(pop)) * TOWN_DIST_FACTOR
        if (t.x - x) ** 2 + (t.y - y) ** 2 < min_dist ** 2:
            too_close = True
            break

    if too_close:
        continue

    towns.append(Town(id=len(towns), x=x, y=y, pop=pop))

pprint(towns)

#######

"""
- Initialize table of connections between pairs of towns incl. traffic volume
  - For each town of origin T1:
    - total traffic = pop(T1)
    - for each other town T2:
      - weight(T1, T2) = pop(T2) * pow(distance(T1, T2), -k)        # k = paremeter 0..inf
    - distribute total traffic according to weights => traffic(T1, T2)
  - For each unordered link(T1, T2)
    - volume(T1, T2) = traffic(T1, T2) + traffic(T2, T1)
"""

traffic_directed = {}

for i, t1 in enumerate(towns):
    weights = {}

    for j, t2 in enumerate(towns):
        if j == i:
            continue

        dist_sq = (t1.x - t2.x) ** 2 + (t1.y - t2.y) ** 2
        weights[j] = t2.pop * math.pow(dist_sq, -K / 2)

    weight_sum = sum(weights.values())

    for j, t2 in enumerate(towns):
        if j == i:
            continue

        traffic = t1.pop * weights[j] / weight_sum
        traffic_directed[(i, j)] = traffic

pprint(traffic_directed)


"""
- While unconnected pairs exist
  - Select unconnected pair at random OR most voluminous first (to experiment)

  - Search for connection using Dijkstra's algorithm
    - For each pre-existing track, set penalty to 1 / (volume(track) + volume(T1, T2))
    - For new track, set penalty to 1 / volume(T1, T2)
    - (later: exclude 90 deg turns)
"""

traffic_volume = {}
unconnected_pairs = []

for i in range(len(towns)):
    for j in range(i + 1, len(towns)):
        vol = traffic_directed[(i, j)] + traffic_directed[(j, i)]
        traffic_volume[(i, j)] = vol
        traffic_volume[(j, i)] = vol
        unconnected_pairs.append((vol, i, j))

unconnected_pairs = sorted(unconnected_pairs, reverse=True)


TOWN_DISPLAY_SCALE = 10

def axes(ax):
    ax.set_xlim(0-W/10, W*1.1)
    ax.set_ylim(0-H/10, H*1.1)
    # ax.grid()
    ax.set_aspect("equal")

def show_towns(ax):
    for i, t1 in enumerate(towns):
        ax.scatter([t1.x], [t1.y], c="k", s=math.sqrt(t1.pop) * TOWN_DISPLAY_SCALE, alpha=0.2)
        ax.annotate(xy=(0,0), xytext=(t1.x, t1.y), text=f"{t1.pop}", fontsize=8, ha="center", va="center")
    #ax.imshow(height < 0, origin="lower")
    ax.scatter(*zip(*lake_list), marker="o", color=(0, 0.2, 0.6), alpha=0.4)
    ax.scatter(*zip(*mountain_list), marker="^", color=(0.3, 0.3, 0.3))

if ENABLE_PLOTS:
    f, ax = plt.subplots()
    show_towns(ax)

    # for i, t1 in enumerate(towns):
    #     for j, t2 in enumerate(towns):
    #         if j == i:
    #             continue
    #
    #         if j > i:
    #             ax.annotate(xytext=(t1.x, t1.y), xy=(t2.x, t2.y), arrowprops=dict(width=1, headwidth=5, color=(0,0,0,0.5)), text="")
    #
    #             ax.annotate(xy=(0,0), xytext=((t1.x + t2.x) / 2, (t1.y + t2.y) / 2), text=f"to: {traffic_directed[i, j]:.2f} "
    #                                                                                       f"back: {traffic_directed[j, i]:.2f} "
    #                                                                                       f"vol: {traffic_volume[j, i]:.2f}",
    #                         fontsize=8
    #                         )

    axes(ax)
    f.savefig("towns.png")

paths = []
traffic_by_segment = {}     # maps (x, y, dx, dy) to total traffic passing

def get_segment_traffic(xx, yy, dx, dy):
    # convention: Y always increasing, unless horizontal and then X increasing
    # force (-1, 0) to (1, 0)
    # force (0, -1) to (0, 1)
    # force (-1, -1) to (1, 1)
    # force (1, -1) to (-1, 1)
    if (dx < 0 and dy == 0) or (dx == 0 and dy < 0) or (dx < 0 and dy < 0) or (dx == 1 and dy == -1):
        yy += dy
        dy = -dy
        xx += dx
        dx = -dx

    assert (dx, dy) in {(0, 1), (1, 0), (1, 1), (-1, 1)}
    return traffic_by_segment.get((xx, yy, dx, dy), 0)

def set_segment_traffic(xx, yy, dx, dy, value):
    if (dx < 0 and dy == 0) or (dx == 0 and dy < 0) or (dx < 0 and dy < 0) or (dx == 1 and dy == -1):
        yy += dy
        dy = -dy
        xx += dx
        dx = -dx

    assert (dx, dy) in {(0, 1), (1, 0), (1, 1), (-1, 1)}
    traffic_by_segment[(xx, yy, dx, dy)] = value

DIRS = [
    (0, 1, 1),
    (0, -1, 1),
    (-1, 0, 1),
    (1, 0, 1),
    (1, 1, math.sqrt(2)),
    (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)),
    (-1, -1, math.sqrt(2)),
]

# return turn angle as a multiple of 45 degrees
def compute_turn_severity(dx1, dy1, dx2, dy2):
    # inefficient!
    a1 = math.atan2(dy1, dx1)
    a2 = math.atan2(dy2, dx2)
    diff = a2 - a1
    if diff < -math.pi: diff += 2*math.pi
    elif diff > math.pi: diff -= 2*math.pi

    return round(abs(diff) / (math.pi * 0.25))

num_steps = len(unconnected_pairs)

class Kernel:
    _step: int = 0

    def __init__(self, args, tv):
        tv.send_hello(w=W, h=H, bg=["#fff" for _ in range(W * H)])

    @property
    def name(self):
        return "NETW23"

    def report(self, f):
        pass

    def step(self, tv):
        if len(unconnected_pairs) == 0:
            raise KeyboardInterrupt()

        step = self._step

        volume, idx1, idx2 = unconnected_pairs.pop(0)

        t1 = towns[idx1]
        t2 = towns[idx2]

        x1, y1 = t1.x, t1.y
        x2, y2 = t2.x, t2.y

        print(f"[{1+step:4d}/{num_steps}] routing track {x1} {y1} --> {x2} {y2}")

        costs = np.full(shape=(H, W), fill_value=np.inf, dtype=float)
        visited = np.zeros(shape=(H, W), dtype=np.uint8)

        # priority queue
        queue = [(0, x2, y2, None)]

        def at(xx, yy):
            if 0 <= xx < costs.shape[1] and 0 <= yy < costs.shape[0]:
                return costs[yy, xx]
            else:
                return np.inf

        def height_at(xx, yy):
            if 0 <= xx < height.shape[1] and 0 <= yy < height.shape[0]:
                return height[yy, xx]
            else:
                return 0

        while len(queue):
            cost, xx, yy, prev = heapq.heappop(queue)
            # print(xx, yy)

            if not (0 <= xx < costs.shape[1] and 0 <= yy < costs.shape[0]):
                continue

            if visited[yy, xx]:
                # already processed
                # TODO: should this ever happen? yes, but not with lower cost than previously assigned
                assert cost >= costs[yy, xx]
                continue

            costs[yy, xx] = cost
            visited[yy, xx] = 1

            if (xx, yy) == (x1, y1):
                # nothing more to do
                break

            for dx, dy, length in DIRS:
                if height_at(xx + dx, yy + dy) < 0 or height_at(xx + dx, yy + dy) > MOUNTAIN_THR:
                    continue

                # valid turn?
                if prev is not None:
                    severity = compute_turn_severity(-prev[0], -prev[1], dx, dy)

                    if severity > 2:
                        continue
                else:
                    severity = 0

                #segment_cost = length / math.pow(volume + get_segment_traffic(xx, yy, dx, dy), VOLUME_SPEEDUP_EXPONENT)
                t = get_segment_traffic(xx, yy, dx, dy)
                if t == 0:
                    build_cost = BUILD_COST * length
                elif t + volume > THROUGHPUT_LIMIT:
                    continue
                else:
                    build_cost = 0
                segment_cost = severity * TURN_PENALTY + build_cost + length / math.pow(t + volume, VOLUME_SPEEDUP_EXPONENT)
                heapq.heappush(queue, (cost + segment_cost, xx + dx, yy + dy, (-dx, -dy, prev)))

            #queue.sort()    # TODO: optimize

        # route it
        xx, yy = x1, y1

        path = [(xx, yy)]

        while prev:
            dx, dy, prev = prev

            set_segment_traffic(xx, yy, dx, dy,
                                get_segment_traffic(xx, yy, dx, dy) + volume)
            xx, yy = xx + dx, yy + dy

            path.append((xx, yy))

        # this is wrong! just because a tile is "cheap" does not mean that any edge leading into it is too
        # while (xx, yy) != (x2, y2):
        #     # look around and pick the best one
        #     neighbor_costs = []
        #     for dx, dy, _ in DIRS:
        #         heapq.heappush(neighbor_costs, (at(xx + dx, yy + dy), dx, dy))
        #     # neighbor_costs.sort()
        #
        #     min_cost, dx, dy = heapq.heappop(neighbor_costs)
        #
        #     # print(f"{xx} {yy} {min_cost}")
        #
        #     # if min_cost >= at(xx, yy):
        #     #     print(f"PATH FINDING ERROR: min({neighbor_costs}) > {at(xx, yy)} (at {xx} {yy})")
        #
        #     assert min_cost < at(xx, yy)
        #
        #     set_segment_traffic(xx, yy, dx, dy,
        #                         get_segment_traffic(xx, yy, dx, dy) + volume)
        #     xx, yy = xx + dx, yy + dy
        #
        #     path.append((xx, yy))

        # print(path)
        paths.append(path)

        tv_labels = []
        tv_paths_bg = []
        tv_paths = []

        TV_SCALE = 0.10

        tv_paths_bg.append(TvPath(grid(W, H, -0.5, -0.5), linewidth=0.01, stroke="rgb(0% 0% 0% / 0.5)"))

        for i, t1 in enumerate(towns):
            x = float(t1.x)
            y = float(t1.y)
            area = math.sqrt(t1.pop) * TOWN_DISPLAY_SCALE       # MPL uses points squared as the unit of size for scatter
            tv_paths.append(TvPath(circle(x, y, r=math.sqrt(area / np.pi) * TV_SCALE), fill="rgba(0,0,0,0.2)"))
            tv_labels.append(Label(x, y, f"{t1.pop}", color="black"))

        for x, y in lake_list:
            tv_paths.append(TvPath(rectangle_centered(x, y, w=1, h=1), fill="rgba(0 20% 60% / 0.4)"))

        for x, y in mountain_list:
            tv_paths.append(TvPath(triangle(x, y, 0.5), fill="rgba(30% 30% 30%)"))

        for (xx, yy, dx, dy), vol in traffic_by_segment.items():
            if vol < 10: color = (0, 0, 1, 0.3)     # blue
            elif vol < 100: color = (0, 0.7, 0, 0.3)    # green
            elif vol < 1e3: color = (1, 1, 0, 0.8)      # yellow
            elif vol < 1e4: color = (1, 0.5, 0, 0.6)    # orange
            elif vol < 1e5: color = (1, 0, 0, 0.5)      # red
            else: color = "black"
            lw = 0.5 + math.log2(vol) * 0.15

            if isinstance(color, tuple):
                r, g, b, a = color
                color = f"rgba({r*100}% {g*100}% {b*100}% / {a})"

            tv_paths_bg.append(TvPath(line(xx, yy, xx + dx, yy + dy), linewidth=(lw + 3) * TV_SCALE, stroke=color))
            tv_paths.append(TvPath(line(xx, yy, xx + dx, yy + dy), linewidth=lw * TV_SCALE, stroke="black"))

        tv_paths.append(TvPath(polyline(path), linewidth=2 * TV_SCALE, stroke="red"))
        tv.send_annotations(labels=tv_labels, paths=tv_paths_bg + tv_paths)

        if step + 1 == num_steps:
            # force always the same filename for final step
            step = 9999

        if ENABLE_PLOTS and (step < 10 or (step < 100 and step % 10 == 0) or (step < 1000 and step % 25 == 0) or step % 100 == 0 or step == 9999):
            f, ax = plt.subplots()

            im = ax.imshow(costs, origin="lower")
            axes(ax)
            f.colorbar(im)
            # f.savefig(f"costs_{step:04d}.png")

            f, ax = plt.subplots()
            show_towns(ax)

            for (xx, yy, dx, dy), vol in traffic_by_segment.items():
                if vol < 10: color = (0, 0, 1, 0.3)     # blue
                elif vol < 100: color = (0, 0.7, 0, 0.3)    # green
                elif vol < 1e3: color = (1, 1, 0, 0.8)      # yellow
                elif vol < 1e4: color = (1, 0.5, 0, 0.6)    # orange
                elif vol < 1e5: color = (1, 0, 0, 0.5)      # red
                else: color = "black"
                lw = 0.5 + math.log2(vol) * 0.15
                ax.plot([xx, xx + dx], [yy, yy + dy], color=color, linewidth=lw + 3, zorder=1) #
                ax.plot([xx, xx + dx], [yy, yy + dy], color="k", linewidth=lw)

            if step != 9999:
                xs, ys = zip(*path)
                # ax.plot(xs, ys, color="red", linewidth=8, alpha=0.25)
                ax.plot(xs, ys, color="red", linewidth=2)

            axes(ax)
            # f.savefig(f"paths_{step:04d}.png", bbox_inches="tight")
        self._step += 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from tilevision.runner import run_kernel
    run_kernel(Kernel)
