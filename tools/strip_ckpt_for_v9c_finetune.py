"""Strip shape-mismatched keys from shu_wei's VAD_tiny_e2e.pth so it can
be load_from'd into a v9c model (h4f12, fut_ts=12, with speed head).

Drops:
  pts_bbox_head.traj_branches.*.4.{weight,bias}   ([12,512] -> [24,512])
  pts_bbox_head.ego_fut_decoder.4.{weight,bias}   ([36,512] -> [72,512])
"""
from __future__ import annotations

import argparse
import torch


SHAPE_MISMATCH_KEYS = (
    'pts_bbox_head.traj_branches.0.4.weight',
    'pts_bbox_head.traj_branches.0.4.bias',
    'pts_bbox_head.ego_fut_decoder.4.weight',
    'pts_bbox_head.ego_fut_decoder.4.bias',
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='/media/user/data1/shu_wei/VAD/ckpts/VAD_tiny_e2e.pth')
    p.add_argument('--dst', default='ckpts/shu_wei_stripped_for_v9c.pth')
    args = p.parse_args()

    print(f'[load] {args.src}')
    ckpt = torch.load(args.src, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    print(f'  {len(sd)} state_dict keys')

    dropped = []
    for k in SHAPE_MISMATCH_KEYS:
        if k in sd:
            dropped.append((k, list(sd[k].shape)))
            del sd[k]
    print(f'[drop] {len(dropped)} shape-mismatched keys:')
    for k, sh in dropped:
        print(f'  {k} {sh}')

    ckpt['state_dict'] = sd
    print(f'[save] {args.dst}  ({len(sd)} keys remain)')
    torch.save(ckpt, args.dst)
    print('[done]')


if __name__ == '__main__':
    main()
