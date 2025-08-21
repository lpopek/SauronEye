import os
import argparse
import configparser
import tqdm
from ultralytics import YOLO
from metrics.benchmark import calculate_metrics
import datetime


MOTA_DIR = "/workspace/evaluation/MOTA/MOT15"
RESULT_DIR = "/workspace/output"

def parse_args():
    parser = argparse.ArgumentParser(description="Script for YOLO inference and calculating MOT15 resulst")
    parser.add_argument("--mode", type=str, default='train', help="<Mode: 'train' or 'test'")
    parser.add_argument("--seq", type=str, default=None, help="Path to seq to be calculated")
    parser.add_argument("--model", type=str, help="YOLO model path", required=True)
    parser.add_argument("--device", type=str, default="cuda", help="Device: 'cpu' or 'cuda'")
    parser.add_argument("--conf", type=float, default=0.14, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.2, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--classes", type=int, default=[0], help="List of class IDs to detect")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Type of tracker or path to .yaml file")
    parser.add_argument("--metrics", type=bool, default=True, help="Flag for calculating metrics")

    return parser.parse_args()

def initilize_process():
    args = parse_args()
    now = datetime.datetime.now()
    print("[INFO]: Launching new calculation")
    result_main_dir = os.path.join(RESULT_DIR, f"job_{now.strftime('%Y%m%d%H%M%S')}")
    os.mkdir(result_main_dir)
    eval_main_dir = os.path.join(MOTA_DIR, args.mode)
    if args.seq is not None:
        seq_dir = os.path.join(result_main_dir, os.path.basename(args.seq))
        os.mkdir(seq_dir)
    elif args.mode == 'train' or args.mode == 'test':
        seqs = os.listdir(eval_main_dir)
        for dir in seqs:
            os.mkdir(os.path.join(result_main_dir, dir))
    else:
        raise ValueError('[ERROR]: Wrong mode choosen. Please make sure if right type of parameter added.')
    print("[INFO]: Calculation directory initialised.")

    return result_main_dir, eval_main_dir

def read_seqinfo(project_dir: str):
    """
    Reads MOTChallenge-style seqinfo.ini and returns a dict with lowercase keys.
    Expected section: [Sequence]
    """
    seqinfo_path = os.path.join(project_dir, 'seqinfo.ini')
    if not os.path.isfile(seqinfo_path):
        raise FileNotFoundError(f"seqinfo file not found at: {seqinfo_path}")

    cfg = configparser.ConfigParser()
    cfg.read(seqinfo_path)

    seq_dict = {}
 
    for k, v in cfg["Sequence"].items():
        key = k.lower()
        try:
            seq_dict[key] = int(v)
        except ValueError:
            seq_dict[key] = v
    seq_dict["imdir"] = os.path.join(project_dir, seq_dict["imdir"])
    seq_dict["gt"] = os.path.join(project_dir, 'gt/gt.txt')

    return seq_dict

def make_inference(imdir: str, results_dir: str, MOTA_dir_name: str):

    args = parse_args()
    if args.model: 
        model = YOLO(args.model)

        print(f"[INFO]: Running YOLO tracking for {MOTA_dir_name}")
        results = model.track(
            source=imdir,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            stream=True,
            classes=args.classes,
            agnostic_nms=True,
            tracker=args.tracker,
            persist=True
        )
        tracked_objects = []
        frame_id = 1
        for result in results:
            det = result.boxes.cpu().numpy()
            if len(det) == 0:
                continue

            objects_in_frame = getattr(result, "boxes", None)
            if len(objects_in_frame) == 0:
                continue

            tracked_objects.append([frame_id, objects_in_frame])
            frame_id += 1
        
        pred_file_path = os.path.join(results_dir, f'pred.txt')
        with open(pred_file_path, "w") as f:
            for frame_id, boxes in tqdm.tqdm(tracked_objects):
                if boxes:
                    for box in boxes:
                        if box.id is not None:
                            f.write(f"{frame_id},{int(box.id)},{int(box.xywh[0][0] - box.xywh[0][3]/2)},{int(box.xywh[0][1] - box.xywh[0][3]/2)},{int(box.xywh[0][2])},{int(box.xywh[0][3])},{round(float(box.conf), 2)},-1,-1,-1\n")
                        else:
                            f.write(f"{frame_id},0,{int(box.xywh[0][0] - box.xywh[0][3]/2)},{int(box.xywh[0][1] - box.xywh[0][3]/2)},{int(box.xywh[0][2])},{int(box.xywh[0][3])},{round(float(box.conf), 2)},-1,-1,-1\n")
        
        print(f"[INFO]: Data saved in file pred.txt")
        
        return pred_file_path


    

def main():
    result_main_dir, eval_main_dir = initilize_process()
    sequences = [f for f in os.listdir(result_main_dir) if not f.startswith(".")]
    MOTA_results = {}
    for seq in sequences:
        path_to_seq = os.path.join(eval_main_dir, seq)
        seqinfo = read_seqinfo(path_to_seq)
        imdir = seqinfo["imdir"]
        sequence_name = seqinfo["name"]
        # frame_rate = seqinfo["framerate"]
        gt_path = seqinfo["gt"]
        pred_path = make_inference( imdir=imdir, 
                                    results_dir=os.path.join(result_main_dir, seq),
                                    MOTA_dir_name=sequence_name
        )
        MOTA_results[seq] = calculate_metrics(gt_file=gt_path, res_file=pred_path)

    with open(os.path.join(result_main_dir, 'results_MOTA.txt'), 'w') as f:
        for key, val in MOTA_results.items():
            f.write(key + '\n')
            title_line, val_line = '', ''
            for metric in val.columns:
                title_line += metric + '\t'
                val_line += str(round(val[metric].iloc[0], 4)) + '\t'
            f.write(title_line + '\n')
            f.write(val_line + '\n')

if __name__ == "__main__":
    main()