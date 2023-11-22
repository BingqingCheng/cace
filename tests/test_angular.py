import sys
sys.path.append('../')
import time
import torch
from cace.modules import AngularComponent, AngularComponent_GPU

vectors = torch.rand(10000, 3)

start_time = time.time()
angular_func = AngularComponent(3)
angular_component = angular_func(vectors)
end_time = time.time()
print(f"Execution time AngularComponent function: {end_time - start_time} seconds")


start_time = time.time()
angular_func_GPU = AngularComponent_GPU(3)
angular_component_GPU = angular_func_GPU(vectors)
end_time = time.time()
print(f"Execution time AngularComponent_GPU function: {end_time - start_time} seconds")

#not supposed to be the same as l_list is different
#print(torch.allclose(angular_component, angular_component_GPU))
