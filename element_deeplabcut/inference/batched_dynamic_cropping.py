import numpy as np
from pathlib import Path
import cv2 as cv
from tqdm import tqdm
import math
from pathlib import Path
import torch
import time
import pickle
import time
from deeplabcut.pose_estimation_pytorch.data import DLCLoader


SINGLE_ANIMAL_POSE_HEAD = "bodypart"
# -----------------------------
# Crop utilities
# -----------------------------
def clamp_box(x1:int, y1:int, x2:int, y2:int, W:int, H:int)->tuple[int,int,int,int]:
    w, h = x2 - x1, y2 - y1
    x1 = max(0, min(x1, W - w))
    y1 = max(0, min(y1, H - h))
    return int(x1), int(y1), int(x1 + w), int(y1 + h)


def init_crops(frame_shape:tuple[int,int], crop_size:int)->tuple[int,int,int,int]:
    H, W = frame_shape[:2]
    cx, cy = W // 2, H // 2
    x1 = cx - crop_size // 2
    y1 = cy - crop_size // 2
    x2 = cx + crop_size // 2
    y2 = cy + crop_size // 2
    return clamp_box(x1, y1, x2, y2, W, H)


def update_crop_from_preds(
    full_kpts:np.ndarray, 
    crop_size:int, 
    frame_shape:tuple[int,int], 
    conf_thresh:float=0.5
)->tuple[int,int,int,int]|None:
    H, W = frame_shape[:2]

    xs = full_kpts[:, 0]
    ys = full_kpts[:, 1]
    conf = full_kpts[:, 2]

    mask = conf > conf_thresh
    if mask.sum() == 0:
        return None

    weights = conf[mask]
    cx = np.sum(xs[mask] * weights) / np.sum(weights)
    cy = np.sum(ys[mask] * weights) / np.sum(weights)

    half = crop_size // 2
    x1 = int(cx - half)
    y1 = int(cy - half)
    x2 = int(cx + half)
    y2 = int(cy + half)

    return clamp_box(x1, y1, x2, y2, W, H)

def crop_image(im:np.ndarray, bbox:tuple[int,int,int,int])->np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    return im[y1:y2, x1:x2]

def extract_keypoints(raw_pred: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    if SINGLE_ANIMAL_POSE_HEAD not in raw_pred:
        available = list(raw_pred.keys())
        raise ValueError(
            f"Expected a '{SINGLE_ANIMAL_POSE_HEAD}' head in model output; "
            f"got {available}. This pipeline only supports single-animal "
            f"PyTorch models with a bodypart head."
        )
    poses = raw_pred[SINGLE_ANIMAL_POSE_HEAD]["poses"]  # (B, N, K, 3)
    if poses.shape[1] != 1:
        raise ValueError(
            f"Multi-instance output detected (N={poses.shape[1]}). "
            "This pipeline only supports single-animal models."
        )
    return poses[:, 0]

# -----------------------------
# Main pipeline
# -----------------------------
def run_batch_dynamic_cropping(
    video_path: str|Path,
    config: str|Path|dict,
    shuffle: int=1, 
    trainingsetindex:int=0, 
    snapshotindex:int|str='best',
    batchsize:int=8,
    crop_size:int=256,
    device:str|None='cuda',
    low_conf_frac_thresh=0.5,
    conf_thresh=0.5,
    start_frame=0,
    end_frame=None,
    destfolder=None,
    save_as_csv=True,
):
    from deeplabcut.pose_estimation_pytorch.apis.videos import create_df_from_prediction
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _check_model_compatibility(loader: DLCLoader) -> None:
        assert not loader.project_cfg["multianimalproject"], (
            "This pipeline only supports single-animal projects."
        )

        engine = loader.model_cfg.get("engine", "pytorch")
        if engine != "pytorch":
            raise ValueError(
                f"Model engine is '{engine}'. This pipeline targets the PyTorch "
                f"engine. TensorFlow models are not supported."
            )

        heads = loader.model_cfg.get("model", {}).get("heads", {})
        if SINGLE_ANIMAL_POSE_HEAD not in heads:
            raise ValueError(
                f"Model config does not define a '{SINGLE_ANIMAL_POSE_HEAD}' head. "
                f"Found: {list(heads.keys())}. Only single-animal bodypart-head "
                f"models are supported."
            )

    def _get_model(dlc_loader:DLCLoader, snapshotindex:int|str='best', device:str='cuda'):
        from deeplabcut.pose_estimation_pytorch.models.model import PoseModel
        from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots


        model = PoseModel.build(dlc_loader.model_cfg['model'])
        model.to(device)

        snapshot = get_model_snapshots(
            snapshotindex, dlc_loader.model_folder, dlc_loader.pose_task
        )[0]

        snapshot_dict = torch.load(
            snapshot.path,
            map_location=device,
            weights_only=dlc_loader.model_cfg['runner'].get("load_weights_only", None)
        )

        model.load_state_dict(snapshot_dict['model'])
        model.eval()
        return model, snapshot

    video_path = Path(video_path)
    loader = DLCLoader(
        config,
        trainset_index=trainingsetindex,
        shuffle=shuffle,
    )
    _check_model_compatibility(loader)
    nbp = len(loader.model_cfg["metadata"]["bodyparts"])
    model, snapshot = _get_model(loader, snapshotindex=snapshotindex, device=device)


    cap = cv.VideoCapture(video_path.as_posix())
    fps = cap.get(cv.CAP_PROP_FPS)
    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    H, W = (int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))

    if end_frame is None:
        end_frame = n_frames

    start_frame = max(0, start_frame)
    end_frame = min(n_frames, end_frame)

    n_window = end_frame - start_frame
    n_batches = math.ceil(n_window / batchsize)

    cap.set(cv.CAP_PROP_POS_FRAMES,start_frame)
    results = []
    low_conf_n_thresh = int(nbp * low_conf_frac_thresh)
    print(f"max. # allowable low conf est.={low_conf_n_thresh}/{nbp}")

    start_time = time.time()
    try:
        with tqdm(total=n_window) as pbar:
            current_crop = init_crops((H,W), crop_size)
            for ii in range(n_batches):

                full_frames = []
                crop_frames = []
                current_batchsize = 0

                # -----------------
                # Read + crop
                # -----------------
                for b in range(batchsize):
                    ret, im = cap.read()
                    if not ret:
                        assert ii == (n_batches - 1), "Failed to reach the end of the video"
                        break

                    im = im[:, :, ::-1]  # ✅ convert to RGB once

                    full_frames.append(im)

                    crop = crop_image(im, current_crop)
                    crop = cv.resize(crop, (crop_size, crop_size))
                    crop_frames.append(crop.transpose(2, 0, 1))
                    current_batchsize += 1

                # safer tensor creation
                crop_cpu = torch.from_numpy(np.stack(crop_frames)).float()
                crop_inp = crop_cpu.to(device)
                del crop_cpu

                # -----------------
                # Inference (cropped)
                # -----------------
                with torch.inference_mode():
                    raw_pred = model.get_predictions(model(crop_inp))

                kpts_np = extract_keypoints(raw_pred).cpu().numpy()  # (B, K, 3)
                x1, y1, x2, y2 = current_crop
                scale_x = (x2 - x1) / crop_size
                scale_y = (y2 - y1) / crop_size

                kpts_np[:,:,0] = kpts_np[:,:,0] * scale_x + x1
                kpts_np[:,:,1] = kpts_np[:,:,1] * scale_y + y1

                # -----------------
                # Confidence check (FIXED INDEXING)
                # -----------------
                bad = np.array([
                    (kpts_np[b, :, 2] < conf_thresh).sum()
                    for b in range(current_batchsize)
                ])
                med_min_confs = np.median(kpts_np[:,:,2].min(axis=1))
                med_max_confs = np.median(kpts_np[:,:,2].max(axis=1))
                fallback = bad > low_conf_n_thresh


                # -----------------
                # Full-frame fallback (only where needed)
                # -----------------
                # kpts_np is now in full-frame coords via crop transform.
                # Fallback predictions are already in full-frame coords (no crop applied),
                # so we can directly overwrite.
                if fallback.any():
                    full_cpu = torch.from_numpy(
                        np.stack(full_frames)[fallback].transpose(0,3,1,2)
                    ).float()
                    full_inp = full_cpu.to(device)
                    del full_cpu

                    with torch.inference_mode():
                        raw_pred_full = model.get_predictions(model(full_inp))

                    kpts_np[fallback] = extract_keypoints(raw_pred_full).cpu().numpy()

                # -----------------
                # Convert + update
                # -----------------

                results.extend({"bodyparts": i[None]} for i in kpts_np)

                new_crop = update_crop_from_preds(
                    kpts_np[-1],
                    crop_size,
                    (H,W),
                    conf_thresh=conf_thresh
                )

                if new_crop is not None:
                    current_crop = new_crop

                pbar.update(current_batchsize)
                pbar.set_description(f"med. (min, max) conf.: ({med_min_confs:.2f},{med_max_confs:.2f}); (min,med,max) # low conf. preds: ({bad.min()}, {np.median(bad)}, {bad.max()})\t")
    finally:
        cap.release()
    stop_time = time.time()

    dlc_scorer = loader.scorer(snapshot)
    if destfolder is None:
        output_path = video_path.parent
    else:
        output_path = Path(destfolder)
    output_prefix = video_path.stem + dlc_scorer
    
    create_df_from_prediction(
        predictions=results,
        multi_animal=False,
        model_cfg=loader.model_cfg,
        dlc_scorer=dlc_scorer,
        output_path=output_path,
        output_prefix=output_prefix,
        save_as_csv=save_as_csv,
    )

    project_cfg = loader.project_cfg
    model_cfg = loader.model_cfg
    train_fraction = project_cfg["TrainingFraction"][trainingsetindex]

    metadata = {
        "data": {
            "start": start_time,
            "stop": stop_time,
            "run_duration": stop_time - start_time,
            "Scorer": dlc_scorer,
            "pytorch-config": model_cfg,
            "fps": fps,
            "batch_size": batchsize,           # nominal; doesn't affect labeled video
            "frame_dimensions": (W,H),
            "nframes": n_frames,
            "iteration (active-learning)": project_cfg["iteration"],
            "training set fraction": train_fraction,
            "cropping": False,
            "cropping_parameters": [0, W, 0, H],
            "individuals": model_cfg["metadata"]["individuals"],
            "bodyparts": model_cfg["metadata"]["bodyparts"],
            "unique_bodyparts": model_cfg["metadata"]["unique_bodyparts"],
        }
    }

    pkl_path = output_path / f"{output_prefix}_meta.pickle"
    with open(pkl_path, "wb") as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

    return results