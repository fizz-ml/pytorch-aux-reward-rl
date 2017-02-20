import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


SIZE_H1 = 120
SIZE_H2 = 100
SIZE_H3 = 120



class model(torch.nn.Module):
	"""Defines custom model
	Inherits from torch.nn.Module 
	"""
	def __init__(self, dim_input, dim_output):
		
		super(model, self).__init__()
		self._dim_input = dim_input
		lf._dim_output = dim_output
		
		'''Initialize nnet layers'''
		self._l1 = torch.nn.Linear(self._dim_input, SIZE_H1)
		self._l2 = torch.nn.Linear(SIZE_H2, SIZE_H1)
		self._bnlayer = BatchNorm1d(SIZE_H2)
		self._l3 = torch.nn.Linear(SIZE_H3, SIZE_H2)


	def forward(self,x):
		self._l1_out = nn.ReLU(self._l1(x))
		self._l2_out = nn.ReLU(self._l2(self._l1_out))
		self._l3_out = nn.BatchNorm1d(self._l3(self._l2_out))
		self._out = nn.Sigmoid(self._l3_out)

		return self._out

		
