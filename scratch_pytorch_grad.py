import torch

x = torch.randn(3, requires_grad = True)

print(x)

# y = x + 3

# print(y)
y = x*x

# print(y)

y = y.mean()

print(y)

y.backward()

print(x.grad)