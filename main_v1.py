import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


dir = Path(f'run_{datetime.now().strftime("%Y-%m-%dT%H-%M-%s")}_v1')
dir.mkdir()
os.chdir(dir)


@dataclass
class Town:
    id: int
    x: int
    y: int
    pop: int


######

W = 50
H = 50
# POP_SCALE = 500
POP_MEAN = 5
POP_SIGMA = 1.4
MAX_TOWNS = 50
MIN_TOWN_DIST = 7
K = 1.5     # exponent for distance when weighing city connections;
            #   decrease to favor connections between further cities if they're large enough
            # we found 1 (linear) too little (ignoring close neighbor in favor of far cities)
            # we found 2 (quadratic) too much

# 0 = no change of throughput with volume; 1 = 2x volume -> 2x throughput
# increase to encourage re-use of existing edges
VOLUME_SPEEDUP_EXPONENT = 0.6


matplotlib.rcParams['figure.figsize'] = (7, 7)

# xs = np.arange(0, W)
# ys = np.arange(0, H)
# xx, yy = np.meshgrid(xs, ys)

"""
- Place towns using Poisson sampling
  - For each, select population from a log-normal distribution
"""

towns = []

rg = np.random.default_rng(seed=0)

for i in range(1000):
    if len(towns) >= MAX_TOWNS:
        break

    x = rg.integers(W)
    y = rg.integers(H)
    pop = math.ceil(rg.lognormal(POP_MEAN, POP_SIGMA))

    too_close = False

    for t in towns:
        if (t.x - x) ** 2 + (t.y - y) ** 2 < MIN_TOWN_DIST ** 2:
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
    ax.grid()
    ax.set_aspect("equal")

f, ax = plt.subplots()

def show_towns(ax):
    for i, t1 in enumerate(towns):
        ax.scatter([t1.x], [t1.y], c="k", s=math.sqrt(t1.pop) * TOWN_DISPLAY_SCALE, alpha=0.2)
        ax.annotate(xy=(0,0), xytext=(t1.x, t1.y), text=f"pop: {t1.pop}", fontsize=8)

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
    if (dx < 0 and dy == 0) or (dx == 0 and dy < 0) or (dx < 0 and dy < 0) or (dx == 1 and dy == -1):
        yy += dy
        dy = -dy
        xx += dx
        dx = -dx
    # force (1, -1) to (-1, 1)
    # elif dx == 1 and dy == -1:

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

step = 0

num_steps = len(unconnected_pairs)

while len(unconnected_pairs) > 0:
    volume, idx1, idx2 = unconnected_pairs.pop(0)

    t1 = towns[idx1]
    t2 = towns[idx2]

    x1, y1 = t1.x, t1.y
    x2, y2 = t2.x, t2.y

    print(f"[{1+step:4d}/{num_steps}] routing track {x1} {y1} --> {x2} {y2}")

    costs = np.full(shape=(H, W), fill_value=np.inf, dtype=float)
    visited = np.zeros(shape=(H, W))

    # priority queue
    queue = [(0, x2, y2)]

    def at(xx, yy):
        if 0 <= xx < costs.shape[1] and 0 <= yy < costs.shape[0]:
            return costs[yy, xx]
        else:
            return np.inf

    while len(queue):
        cost, xx, yy = queue.pop(0)
        # print(xx, yy)

        if not (0 <= xx < costs.shape[1] and 0 <= yy < costs.shape[0]):
            continue

        if visited[yy, xx]:
            # already processed
            # TODO: should this ever happen? yes, but not with lower cost than previously assigned
            continue

        costs[yy, xx] = cost
        visited[yy, xx] = 1

        for dx, dy, length in DIRS:
            segment_cost = length / math.pow(volume + get_segment_traffic(xx, yy, dx, dy), VOLUME_SPEEDUP_EXPONENT)
            queue.append((cost + segment_cost, xx + dx, yy + dy))

        queue.sort()    # TODO: optimize

    # route it
    xx, yy = x1, y1

    path = [(xx, yy)]

    while (xx, yy) != (x2, y2):
        # look around and pick the best one
        neighbor_costs = []
        for dx, dy, _ in DIRS:
            neighbor_costs.append((at(xx + dx, yy + dy), dx, dy))
        neighbor_costs.sort()

        min_cost, dx, dy = neighbor_costs.pop(0)

        # print(f"{xx} {yy} {min_cost}")

        # if min_cost >= at(xx, yy):
        #     print(f"PATH FINDING ERROR: min({neighbor_costs}) > {at(xx, yy)} (at {xx} {yy})")

        assert min_cost < at(xx, yy)

        set_segment_traffic(xx, yy, dx, dy,
                            get_segment_traffic(xx, yy, dx, dy) + volume)
        xx, yy = xx + dx, yy + dy

        path.append((xx, yy))

    # print(path)
    paths.append(path)

    if step + 1 == num_steps:
        # force always the same filename for final step
        step = 9999

    if step < 10 or (step < 100 and step % 10 == 0) or (step < 1000 and step % 25 == 0) or step % 100 == 0 or step == 9999:
        f, ax = plt.subplots()

        im = ax.imshow(costs, origin="lower")
        axes(ax)
        f.colorbar(im)
        f.savefig(f"costs_{step:04d}.png")

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

        xs, ys = zip(*path)
        # ax.plot(xs, ys, color="red", linewidth=8, alpha=0.25)
        ax.plot(xs, ys, color="red", linewidth=2)

        axes(ax)
        f.savefig(f"paths_{step:04d}.png", bbox_inches="tight")
    step += 1
