import torch

a = torch.zeros((10,5,3))

b = torch.ones((10,3))
# a[2,0:3] = 2
# a[4,3:4] = 5
# a[7,2:5] = 3
# a[9,0:5] = 5

c = (a - b.unsqueeze(1)).norm(dim=2).sum(dim=1)

# b=torch.unsqueeze(torch.max(a, dim=1)[0], dim=1)
# b = torch.where(b > 2, b, 0)
# c = torch.norm(a, dim=1)
print(a)
# print(b)
print(c)