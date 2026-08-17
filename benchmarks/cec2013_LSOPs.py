import numpy as np
from cec2013lsgo.cec2013 import Benchmark

benchmark = Benchmark()

print("Number of functions:", benchmark.get_num_functions())


for i in range(1, benchmark.get_num_functions() + 1):
    print("\nFunction:", i)
    print(benchmark.get_info(i))

    func = benchmark.get_function(i)
    x = np.random.uniform(-100, 100, 1000)
    print("Fitness:", func(x))
