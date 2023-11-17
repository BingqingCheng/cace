import sys
sys.path.append('../')
import torch

from cace.tools import elementwise_multiply_2tensors, elementwise_multiply_3tensors

tensor1 = torch.rand([996, 20])
tensor2 = torch.rand([996, 8])
result = elementwise_multiply_2tensors(tensor1,tensor2)
print(result.shape)
assert torch.equal(result[0,0], tensor2[0] * tensor1[0,0]), "Tensors are not equal."

tensor1 = torch.rand([996, 20])
tensor2 = torch.rand([996, 8])
tensor3 = torch.rand([996, 6])
result = elementwise_multiply_3tensors(tensor1,tensor2,tensor3)
print(result.shape)
# TODO: Fix this test
#assert torch.equal(result[0,0], tensor2[0] * tensor1[0,0] * tensor3[0,0]), "Tensors are not equal."


