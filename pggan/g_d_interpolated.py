#charge pretrained generator and discriminator  


import os,sys
import torch
from config import config
from torch.autograd import Variable
import utils as utils
import network as net

def recup_nets(config):
    use_cuda = True
    checkpoint_path_g = config.checkpoint_generator
    checkpoint_path_d = config.checkpoint_discriminator
    


    # load trained model.
    model_g = net.Generator(config)
    model_d = net.Discriminator(config)
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        model_g = torch.nn.DataParallel(model_g).cuda(device=0)
        model_d = torch.nn.DataParallel(model_d).cuda(device=0)
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    for resl in range(3, config.start_res+1):
        model_g.module.grow_network(resl)
        model_d.module.grow_network(resl) 
        model_g.module.flush_network()
        model_d.module.flush_network()
    print('generator :')
    print(model_g)
    print('discriminator :')
    print(model_d)
    

    print('load generator from checkpoint  ... {}'.format(checkpoint_path_g))
    print('load discriminator from checkpoint ... {}'.format(checkpoint_path_d))
    checkpoint_g = torch.load(os.path.join('repo/model',checkpoint_path_g))
    checkpoint_d = torch.load(os.path.join('repo/model',checkpoint_path_d))
    print(type(checkpoint_g['state_dict']))
    print(type(checkpoint_d['state_dict']))
    model_g.module.load_state_dict(checkpoint_g['state_dict'],False)
    model_d.module.load_state_dict(checkpoint_d['state_dict'],False)

    return model_g, model_d

'''

self.z1.data.normal_(0.0, 1.0)
self.z2 = torch.FloatTensor(1, config.nz).cuda() if use_cuda else torch.FloatTensor(1,config.nz)
self.z2 = Variable(self.z2)
self.z2.data.normal_(0.0, 1.0)

print
'''
# forward



# save







