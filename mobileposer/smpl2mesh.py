import articulate as art
from utils import *
from config import paths, joint_set
import torch
import os
import open3d as o3d
import numpy as np
import matplotlib
import trimesh

body_model = art.ParametricModel(paths.smpl_file)

if __name__ == '__main__':
    
    data_dir = '/root/autodl-tmp/dataset/raw/Huawei'
    out_dir = 'data/mesh'
    sub_name = 'xinrui_0407'
    seq_name = '15.pt'
    sub_dir = os.path.join(data_dir, sub_name)
    seq_path = os.path.join(sub_dir, seq_name)
    
    data = torch.load(seq_path)
    pose = data['pose_gt']
    
    _, _, vertex = body_model.forward_kinematics(pose=pose, calc_mesh=True)
    face = body_model.face
        
    output_path = os.path.join(out_dir, sub_name, seq_name.split('.')[0])
    os.makedirs(output_path, exist_ok=True)
        
    # save mesh
    # vertex: [N, 6890, 3]
    # face: [N, 13776, 3]
    vertex_np = vertex.numpy()  # [N, 6890, 3]
    start_index = 200
    end_index = start_index + 200
    for frame_idx in range(start_index, end_index):
        mesh = trimesh.Trimesh(vertices=vertex_np[frame_idx], faces=face, process=False)

        # 保存路径
        mesh_path = os.path.join(output_path, f'{frame_idx:04d}.obj')
        mesh.export(mesh_path)