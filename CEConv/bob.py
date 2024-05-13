import numpy as np
import torch

i = 5
neg_sats = i // 2
pos_sats = neg_sats - 1 + i % 2
print(neg_sats, pos_sats)
sat_shifts = np.append(np.linspace(-1, 0, neg_sats+1)[:-1], np.linspace(0, 1, pos_sats+1))
print(sat_shifts[4])
