import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from config import *
from helpers import * 
import articulate as art
from constants import MODULES
from utils.model_utils import load_model
from data import PoseDataset
from models import MobilePoserNet

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([2, 5, 16, 20]), fps=datasets.fps)

    def eval(self, pose_p, pose_t, joint_p=None, tran_p=None, tran_t=None):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        
        if tran_p is not None and tran_t is not None:
            tran_p = tran_p.clone().view(-1, 3)
            tran_t = tran_t.clone().view(-1, 3)
        else:
            tran_p = torch.zeros(pose_p.shape[0], 3, device=pose_p.device)
            tran_t = torch.zeros(pose_t.shape[0], 3, device=pose_t.device)
            
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)

        errs = self._eval_fn(pose_p, pose_t, tran_p=tran_p, tran_t=tran_t)
        return torch.stack([errs[9], errs[3], errs[9], errs[0]*100, errs[7]*100, errs[1]*100, errs[4] / 100, errs[6]])

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Masked Angular Error (deg)',
                                  'Positional Error (cm)', 'Masked Positional Error (cm)', 'Mesh Error (cm)', 
                                  'Jitter Error (100m/s^3)', 'Distance Error (cm)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i, 0], errors[i, 1]))
    @staticmethod
    def print_single(errors, file=None):
        names = [
            'Angular Error (deg)',
            'Mesh Error (cm)',
        ]
        max_len = max(len(n) for n in names)
        outs = []
        for i, n in enumerate([
            'SIP Error (deg)',
            'Angular Error (deg)',
            'Masked Angular Error (deg)',
            'Positional Error (cm)',
            'Masked Positional Error (cm)',
            'Mesh Error (cm)',
            'Jitter Error (100m/s^3)',
            'Distance Error (cm)',
        ]):
            if n in names:
                outs.append(f"{n:<{max_len}}: {errors[i,0]:.2f}")
        print(" | ".join(outs), file=file)

@torch.no_grad()
def evaluate_pose(model, dataset, num_past_frame=20, num_future_frame=5, 
                  save_dir=None):
    # specify device
    device = model_config.device

    # load data
    xs, ys = zip(*[(imu.to(device), (pose.to(device), tran)) for imu, pose, joint, tran in dataset])

    # setup Pose Evaluator
    evaluator = PoseEvaluator()

    # track errors
    pose_errs = []
    
    model.eval()
    with torch.no_grad():
        for idx, (x, y) in enumerate(zip(xs, ys)):
            # if idx > 1:
            #     break
            
            print(f"Evaluating sample {idx+1}/{len(xs)}...")
            model.reset()

            pose_t, tran_t = y
            pose_t = art.math.r6d_to_rotation_matrix(pose_t)

            # results = [model.forward_online(f) for f in torch.cat((x, x[-1].repeat(num_future_frame, 1)))]
            # pose_p, joint_p_online, tran_p, _ = [torch.stack(_)[num_future_frame:] for _ in zip(*results)]
            pose_p = []
            for i in tqdm(range(x.shape[0])):
                pose = model.forward_frame(x[i])
                pose_p.append(pose)
            pose_p = torch.stack(pose_p)

            pose_errs.append(evaluator.eval(pose_p, pose_t))
            
            # save the results
            out_path = save_dir / f"{idx+1}.pt"
            torch.save(
                {
                    'pose_p': pose_p.cpu(),
                    'pose_t': pose_t.cpu(),
                }, 
                out_path
            )

    # print joint errors
    print('============== online ================')
    evaluator.print(torch.stack(pose_errs).mean(dim=0))

    log_path = save_dir / "log.txt"
    with open(log_path, "w") as f:
        for i, e in enumerate(pose_errs):
            evaluator.print_single(e, file=f)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dip')
    parser.add_argument('--combo', type=str, default='lw_rw_lp_rp_h')
    args = parser.parse_args()

    # load model 
    model = load_model(args.model)

    # load dataset
    if args.dataset not in datasets.test_datasets:
        raise ValueError(f"Test dataset: {args.dataset} not found.")
    dataset = PoseDataset(fold='test', evaluate=args.dataset)

    save_dir = Path("data") / "eval" / args.dataset / args.combo / "origin_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    # evaluate pose
    print(f"Starting evaluation: {args.dataset.capitalize()}")
    evaluate_pose(model, dataset, save_dir=save_dir)
