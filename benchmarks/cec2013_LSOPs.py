import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cec2013lsgo.cec2013 import Benchmark
from src.utils import cleanup_benchmark_csv, register_csv_cleanup

register_csv_cleanup()

try:
    benchmark = Benchmark()

    print("Number of functions:", benchmark.get_num_functions())

    for i in range(1, benchmark.get_num_functions() + 1):
        print("\nFunction:", i)
        print(benchmark.get_info(i))

        func = benchmark.get_function(i)
        x = np.random.uniform(-100, 100, 1000)
        print("Fitness:", func(x))
finally:
    cleanup_benchmark_csv()
