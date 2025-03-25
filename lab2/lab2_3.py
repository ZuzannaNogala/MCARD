import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from models import Recovering


d = 15
n = 11
Z = torch.randint(0, 10, (n, d), dtype=torch.float32)

# Q3.A
Z_np = Z.detach().numpy()

for r in range(1, 12):
    svd_model = TruncatedSVD(n_components=r)
    W_r = svd_model.fit_transform(Z_np)
    H_r = svd_model.components_
    Z_r = W_r @  H_r
    print(f"Loss for {r}: {np.mean((Z_np - Z_r) ** 2)}")


R1 = Recovering(lr=0.02, n_epochs=1000)
loss_dic_dist2 = {}

print("Dist Frob")
for r in range(1, 12):
    R1.fit(Z, r, dist_pow=2, verbose=False)
    loss_dic_dist2[r] = R1.loss_list[-1]
    print(f" The loss for r = {r} = {R1.loss_list[-1]}")

plt.plot(loss_dic_dist2.keys(), loss_dic_dist2.values(), label="power = 2 (Frob)")
plt.show()


def compute_Recovering_for_chosen_r(chosen_r, dist_pow):
    print(f"Recovering Z matrix with r = {chosen_r} and with distance power = {dist_pow}")
    R = Recovering(lr=0.02, n_epochs=1000)
    R.fit(Z, chosen_r, dist_pow=dist_pow, verbose=False)
    Z_r = R.get_recovered_Z()
    print(Z)
    print(Z_r)
    print(f"The loss for chosen r = {chosen_r} is {R.loss_list[-1]}")


Rdist4 = Recovering(lr=0.02, n_epochs=1000)
loss_dic_dist4 = {}

print("Dist - power 4")
for r in range(1, 12):
    Rdist4.fit(Z, r, dist_pow=4, verbose=False)
    loss_dic_dist4[r] = Rdist4.loss_list[-1]
    print(f" The loss for r = {r} = {Rdist4.loss_list[-1]}")

plt.plot(loss_dic_dist2.keys(), loss_dic_dist2.values(), label="power = 2 (Frob)")
plt.plot(loss_dic_dist4.keys(), loss_dic_dist4.values(), label="power = 4")
plt.legend()
plt.show()

r1 = input("Which r do you choose for dist_pow =2: ")
compute_Recovering_for_chosen_r(int(r1), 2)

r2 = input("Which r do you choose for dist_pow =4: ")
compute_Recovering_for_chosen_r(int(r2), 4)

# Q3.B

R2 = Recovering(lr=0.02, n_epochs=1000)
for r in range(1, 9):
    R2.fit_nonnegativeW(Z, r, dist_pow=2, verbose=False)
    print(f" The loss for r = {r} = {R2.loss_list[-1]}")

print(Z)
print(R2.W_r)
print(R2.get_recovered_Z())

R3 = Recovering(lr=0.02, n_epochs=1000)
for r in range(1, 12):
    R3.fit_H_greater_than_half(Z, r, dist_pow=2, verbose=False)
    print(f" The loss for r = {r} = {R3.loss_list[-1]}")

print(Z)
print(R3.H_r > 0.5)
print(R3.get_recovered_Z())
