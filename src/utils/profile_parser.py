import pstats

from pstats import SortKey

p = pstats.Stats('profile.txt')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(50)