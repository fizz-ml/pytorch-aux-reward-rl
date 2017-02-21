import numpy as np 
import torch 
from torch.autograd import Variable

M = torch.load('models/ddpg_models/mountain_cart_actor.t7')

s = np.random.rand(200,2)
r = np.random.rand(200,1)

a = np.random.rand(200,1)

a = Variable(torch.FloatTensor(a))
opti = torch.optim.Adam(M.parameters(), lr=0.001)

lfun = torch.nn.MSELoss(size_average=False)
for t in range(1000):
	a_pred = M(s,r,[])
	loss = lfun(a_pred,a)
	print(t, loss.data[0])
	opti.zero_grad()
	loss.backward()
	opti.step()