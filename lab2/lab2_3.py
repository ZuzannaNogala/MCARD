import torch

d=15
n=11
Z = torch.randint(0,10,(n,d), dtype=torch.float32)
print(Z)
print(Z.shape)


# Q3.A

U, Lmb, V = torch.svd(Z)
print(U.shape)
print(Lmb.shape)
print(V.shape)
