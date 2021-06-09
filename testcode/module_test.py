import torch
from model.module.layer import MultiDimWN, FlexMultiDimWCN, FlexMultiDimDynamicWCN
from model.module.wavelet_func import mexican_hat

import matplotlib.pyplot as plt
from tqdm import tqdm

# Train config
EPCH = 40000
fucking_model = torch.nn.Sequential(
    FlexMultiDimDynamicWCN(1, 1, 17, 3, mexican_hat)
)

#fucking_model = torch.nn.Sequential(torch.nn.Conv1d(1, 1, 17, padding=int(17 / 2)))

fucking_model = torch.nn.Sequential(FlexMultiDimWCN(1, 1, 17, 3, mexican_hat))

optim = torch.optim.Adam(lr=0.0005, params=fucking_model.parameters())
loss = torch.nn.MSELoss()

l_vh = []

for i in tqdm(range(EPCH)):
    optim.zero_grad()
    # x = torch.rand(6, 1, 100)
    x = torch.linspace(-5, 5, 100).expand(1, 1, 100)
    x = torch.sin(4 * x)
    y = 6 * torch.sin(8 * x) + (torch.rand(1, 1, 100))
    y_p = fucking_model(x)
    l_v = loss(y, y_p)
    l_v.backward()
    l_vh.append(l_v.item())
    optim.step()

    # Show the result each epoch?
    if 1 == 0:
        y_np = y.reshape(y.shape[2]).numpy()
        y_p_np = y_p.reshape(y_p.shape[2]).detach().numpy()
        plt.plot(y_np)
        plt.plot(y_p_np)
        plt.plot(x.reshape(x.shape[2]))
        plt.show()

    if i % 10000 == 0:
        print(l_v.item())
        plt.plot(l_vh)
        plt.show()

fucking_model.eval()
# x = torch.rand(1, 1, 100)
x = torch.linspace(-5, 5, 100).reshape(1, 1, -1)
x = torch.sin(4 * x)
y = 6 * torch.sin(8 * x) + (torch.rand(1, 1, 100))
y_p = fucking_model(x)

y_np = y.reshape(y.shape[2]).numpy()
y_p_np = y_p.reshape(y_p.shape[2]).detach().numpy()
plt.plot(y_np)
plt.plot(y_p_np)
plt.plot(x.reshape(x.shape[2]))
plt.show()
