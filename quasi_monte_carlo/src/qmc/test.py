from qmc.sampler.direction_numbers import load_direction_numbers

v = load_direction_numbers('/Users/celine/Documents/projects/ray-maps/quasi_monte_carlo/src/qmc/sampler/joe_kuo_new_100.txt', max_dim=64)
print(v.shape)          # (64, 32)
print(v[0][:4])         # dim 0: [2147483648, 1073741824, 536870912, 268435456]
print(v[1][:4])         # dim 1: should start with 2147483648 (= 1/2 in fixed point)
print(bin(v[0][0]))     # 0b10000000000000000000000000000000
print(bin(v[0][1]))     # 0b01000000000000000000000000000000