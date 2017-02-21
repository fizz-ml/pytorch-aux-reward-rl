from models.config import model
import torch
from torch.utils.serialization import load_lua
from torch.autograd import Variable

'''example on using generator to save/load model'''


M = model(20, 1)
M.children()
torch.save( M, 'model.t7')
M_loaded = torch.load('model.t7')

tryx = Variable(torch.randn(12,20))
tryy = M_loaded(tryx)

print(tryy)

