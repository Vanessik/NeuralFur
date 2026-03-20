from argparse import ArgumentParser
import yaml
import torch
import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_PROJECT_ROOT, 'ext', 'NeuralHaircut'))
sys.path.append(os.path.join(_PROJECT_ROOT, 'ext', 'NeuralHaircut', 'k-diffusion'))
from src.hair_networks.optimizable_textured_fur import OptimizableTexturedStrands
import numpy as np
import trimesh
import torch.nn.functional as F
from src.hair_networks.strand_prior import Decoder


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--hair_conf_path', default='', type=str)
    parser.add_argument('--model_name', default='', type=str)
    parser.add_argument('--iter', default='10000', type=str)
    parser.add_argument('--n_strands', default=30000, type=int)
    parser.add_argument('--save_pts_per_strand', default=100, type=int)
    parser.add_argument('--data_root', default='', type=str, help="Data root path, replaces DATA_ROOT in the yaml config")
    args, _ = parser.parse_known_args()

    # Load hair config and replace placeholders with actual paths
    with open(args.hair_conf_path, 'r') as f:
        replaced_conf = str(yaml.load(f, Loader=yaml.Loader))
        replaced_conf = replaced_conf.replace('DATASET_TYPE', 'monocular')
        if args.data_root:
            replaced_conf = replaced_conf.replace('DATA_ROOT', args.data_root.rstrip('/'))
        strands_config = yaml.load(replaced_conf, Loader=yaml.Loader)

        
    strands_generator = OptimizableTexturedStrands(
            **strands_config['textured_strands']
        ).cuda()
    
    max_sh_degree = 3
    color_decoder = Decoder(None, dim_hidden=128, num_layers=2, dim_out=3*(max_sh_degree+1)**2 + 1).cuda()
    print('create color decoder')
    
    strands_generator.eval()
    color_decoder.eval()
       
    weights = torch.load(f'{args.model_name}/checkpoints/{args.iter}.pth')

    
    try:
        strands_generator.load_state_dict(weights[0][2])
    except Exception as e:
        print('Failed to load strands generator weights')

    try:
        color_decoder.load_state_dict(weights[0][3])
    except Exception as e:
        print('Failed to load color decoder weights')
        
    
    if strands_generator.strand_length_scale is not None:
        strands_generator.init_length_scale()
        strands_generator.init_strand_groups()
        
    strands_generator.init_pos_enc()
       
    print('scale is', strands_generator.length_param)
    with torch.no_grad():
        p = strands_generator.forward_inference(args.n_strands)[0]
    os.makedirs(f'{args.model_name}/strands', exist_ok=True)
    
    if args.save_pts_per_strand < 100:
    
        # Step 1: Permute to [N, C, P]
        p_perm = p.permute(0, 2, 1)  # [500000, 3, 100]

        # Step 2: Interpolate to 25 points
        p_interp = F.interpolate(p_perm, size=args.save_pts_per_strand, mode='linear', align_corners=True)  # [500000, 3, 25]

        # Step 3: Permute back to [N, 25, 3]
        p_result = p_interp.permute(0, 2, 1) 


        p_npy = p_result.cpu().numpy()
    
    else:
        p_npy = p.cpu().numpy()

    save_path = f'{args.model_name}/strands/{args.iter}_{args.n_strands}_strands.ply'

    xyz = p_npy.reshape(-1, 3)
    n_strands = p_npy.reshape(-1, args.save_pts_per_strand, 3).shape[0]
    cols = torch.cat((torch.rand(n_strands, 3).unsqueeze(1).repeat(1, args.save_pts_per_strand, 1), torch.ones(n_strands, args.save_pts_per_strand, 1)), dim=-1).reshape(-1, 4).cpu()          
    pc = trimesh.points.PointCloud(xyz, colors=cols)
    pc.export(save_path)  