import sys, re
import numpy as np
from collections import defaultdict


def parse(filename):
    lines = [l.strip() for l in open(filename, 'r')]

    # parse generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc
    # geeege_prev = 'generator_loss |'
    # geeeges = []
    # for prev, line in zip(lines, lines[1:]):
    #     if prev.startswith(geeege_prev):
    #         geeege = np.array([float(w) for w in line.replace('|', '').split()])
    #         geeeges.append(geeege)
    # print('Parsed %i generator/expert performance entries.' % len(geeeges))
    # geeeges = np.array(geeeges[:1000])

    # parse tabulated values
    tables = defaultdict(list)
    table_open = False
    table_entries = 0
    for i, line in enumerate(lines):
        if line.startswith('-----------------------------'):
            table_open = not table_open
            if table_open:
                table_entries += 1
            continue
        if table_open:
            key, value = tuple(line.replace('|', '').split()[:2])
            tables[key].append(float(value))
    print('Parsed %i tabulated performance entries.' % table_entries)

    # tables['generator_loss'] = geeeges[:, 0]
    # tables['expert_loss'] = geeeges[:, 1]
    # tables['entropy'] = geeeges[:, 2]
    # tables['entropy_loss'] = geeeges[:, 3]
    # tables['generator_acc'] = geeeges[:, 4]
    # tables['expert_acc'] = geeeges[:, 4]

    return tables


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s LOG_FILENAME" % sys.argv[0])
    else:
        parse(sys.argv[1])
