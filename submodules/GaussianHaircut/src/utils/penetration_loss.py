import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, 'ext', 'NeuralHaircut', 'NeuS'))
from models.fields import SDFNetwork
import torch
import trimesh


class SdfPenetration:
    def __init__(self,
                 ckpt_name='',
                 device='cuda'):
        
        
        self.device = device 
        self.sdf_network = SDFNetwork(d_out = 257,
                d_in = 3,
                d_hidden = 256,
                n_layers = 8,
                skip_in = [4],
                multires = 6,
                bias = 0.5,
                scale = 1.0,
                geometric_init = True,
                weight_norm = True).to(self.device)

        checkpoint = torch.load(ckpt_name, weights_only=False, map_location=device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.sdf_network.eval()

        
    def query(self, x, EPS=-5e-4):
        
        dists = self.sdf_network.forward(x)[:, :1]

        d_output = torch.ones_like(dists, requires_grad=False, device=dists.device)
        gradients = torch.autograd.grad(
                            outputs=dists,
                            inputs=x,
                            grad_outputs=d_output,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]

        normals = gradients / (gradients.norm(dim=-1, keepdim=True) + 1e-10)

        return dists, normals