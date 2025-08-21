import motmetrics as mm
import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=None: np.asarray(a, dtype=float if dtype is None else dtype)


def calculate_metrics(gt_file: str, res_file: str) -> None: 

    acc = mm.MOTAccumulator(auto_id=True)

    # Load ground truth and hypothesis
    gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=1)
    res = mm.io.loadtxt(res_file, fmt="mot15-2D")

    # Compare
    mh = mm.metrics.create()
    acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.7)

    # events = acc.mot_events  # DataFrame with FrameId, Type, OId (GT), HId (tracker), D (distance = 1-IoU)
    # matches = events[events['Type'] == 'MATCH'][['FrameId','OId','HId','D']]
    # matches['IoU'] = 1.0 - matches['D']
    # print(matches.head())

    # Compute summary
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1', 'precision', 'recall'], name='tracker')

    print(f"[INFO]: Calculated metrics")
    return summary