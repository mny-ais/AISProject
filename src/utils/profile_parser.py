import pstats

from os import path

from pstats import SortKey

parent_path = path.dirname(path.dirname(path.abspath(__file__)))
p = pstats.Stats(path.join(parent_path, 'profile.txt'))
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(50)