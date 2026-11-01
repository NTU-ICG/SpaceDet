"""Run the SpaceDet YOLO detector + BoT-SORT tracker on raw images and keep what OD needs.

Reuses YOLOv8Trainer.predict (tiled inference + box merging) and ObjectTracker.track_objects
from codes/scripts unchanged, but unlike main.py it also saves the predicted class (0 = LEO) and
the boxes in pixels, so that LEO tracks can be selected and the angle extraction chosen later.

Outputs (in --out):
    detections.csv  raw merged detections: t, cls, conf, x, y, w, h   (pixels, full image)
    tracks.csv      tracker output:        t, track_id, cls, x, y, w, h

Usage (from the repository root):
    python codes/orbit_determination/detect_track.py --images data/SpaceNet-100/images
"""
import argparse
import glob
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'codes'))             # bundled (modified) ultralytics
sys.path.insert(1, os.path.join(ROOT, 'codes', 'scripts'))

import pandas as pd
import torch

# The repository's checkpoint is a full pickled model, which torch>=2.6 refuses by default.
# Allow it here only: the weights ship with the SpaceDet repository itself.
_torch_load = torch.load
torch.load = lambda *args, **kw: _torch_load(*args, **{'weights_only': False, **kw})

from ObjectTracker import Args, ObjectTracker
from YOLOv8Trainer import ConfigLoader, YOLOv8Trainer
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import DEFAULT_CFG


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', default='data/SpaceNet-100/images')
    ap.add_argument('--model', default='codes/scripts/models/SpaceNet-100_YOLOv8n.pt')
    ap.add_argument('--tracker-config', default='codes/ultralytics/cfg/botsort.yaml')
    ap.add_argument('--out', default='results/orbit_determination/detections')
    ap.add_argument('--max-images', type=int, default=None)
    ap.add_argument('--device', default=None, help="e.g. 'cpu' or '0'; default: GPU if available")
    a = ap.parse_args()
    os.chdir(ROOT)
    # default.yaml asks for CUDA device 0; the detector and the tracker's re-ID model both read it.
    DEFAULT_CFG.device = a.device or ('0' if torch.cuda.is_available() else 'cpu')

    cfg = ConfigLoader('codes/ultralytics/cfg/default.yaml')
    W = cfg.get('width_default')
    work = os.path.join(a.out, 'work')
    trainer = YOLOv8Trainer(work, os.path.join(work, 'runs'), cfg.get('picture_base_name'), a.images, 'predict',
                            cfg.get('distance_threshold'), cfg.get('angle_threshold'), cfg.get('Iou_threshold'),
                            cfg.get('conf_threshold'), cfg.get('track_low_thresh'), a.tracker_config, None,
                            a.model, Focal_length=cfg.get('focal_length') * 1e-6, width=W, height=cfg.get('height_default'),
                            width_mm=cfg.get('width_mm_default'), height_mm=cfg.get('height_mm_default'),
                            cutting_size=cfg.get('cutting_size'), overlap=cfg.get('overlap'))
    tracker = ObjectTracker(a.model, a.images, work, a.tracker_config, W)
    args = Args(tracker.load_yaml_config())
    bot = BOTSORT(args, frame_rate=1) if args.tracker_type == 'botsort' else BYTETracker(args, frame_rate=1)

    images = sorted(glob.glob(os.path.join(a.images, 'Raw_Observation*.tiff')))[:a.max_images]
    dets, tracks = [], []
    start = time.time()
    for frame_id, path in enumerate(images, start=1):
        path = path.replace('\\', '/')
        t = int(os.path.basename(path)[len('Raw_Observation'):][:4])
        merged = trainer.predict(path, os.path.join(work, 'predict'))
        if merged == 1:
            continue
        dets += [[t, int(b[1]), b[6], b[2] * W, b[3] * W, b[4] * W, b[5] * W] for b in merged]
        for _, tid, cls, x, y, w, h in tracker.track_objects(merged, path, frame_id, bot):
            tracks.append([t, int(tid), int(cls), x, y, w, h])
        print(f'[{frame_id}/{len(images)}] t={t}s  detections={len(merged)}  '
              f'elapsed={time.time() - start:.0f}s', flush=True)

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(dets, columns=['t', 'cls', 'conf', 'x', 'y', 'w', 'h']).to_csv(
        os.path.join(a.out, 'detections.csv'), index=False)
    pd.DataFrame(tracks, columns=['t', 'track_id', 'cls', 'x', 'y', 'w', 'h']).to_csv(
        os.path.join(a.out, 'tracks.csv'), index=False)
    print(f'Saved {len(dets)} detections and {len(tracks)} track points to {a.out}')


if __name__ == '__main__':
    main()
