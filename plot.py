

import matplotlib.pyplot as plt
import math

opts = 6

for i in range(opts):
    paramlist = []
    optrangelower = math.ceil((12.0/opts) *i)
    optrangeupper = math.ceil((12.0/opts) * (i+1))
    
    optrange = list(range(optrangelower,optrangeupper))
    print(optrange)