import torch
import torch.nn.functional as F
from torchvision.io import read_video, write_video
import cv2
import os
from argparse import ArgumentParser
import argparse, os, re, shutil
from pathlib import Path
from datetime import datetime

# read video as tensor: (T, C, H, W)
CAM_INDEX = {"right": 0, "left": 1, "wrist": 2}
FOLDER_RE = re.compile(r"^video_(\d{4})_(\d{2})_(\d{2})_(\d{2}):(\d{2}):(\d{2})$")

def parse_args():
    p = argparse.ArgumentParser(description="Reorganize Ctrl-World videos into IDed trajectories.")
    p.add_argument("--dataset", default="demos")
    # Temporarily allow source/dest to be None
    p.add_argument("--source", default=None)
    p.add_argument("--dest",   default=None)

    # First parse to get dataset value
    args, remaining = p.parse_known_args()

    # Now compute default paths
    base = "/n/fs/irom-testing/world_models/Ctrl-World/dataset_example"
    default_source = f"{base}/{args.dataset}"
    default_dest   = f"{base}/{args.dataset}/videos"

    # Set defaults dynamically
    p.set_defaults(source=default_source, dest=default_dest)

    # p.add_argument("--source",
    #                default="/n/fs/irom-testing/world_models/Ctrl-World/dataset_example/pi0_demos",
    #                help="Directory containing video_YYYY_MM_DD_HH:MM:SS subfolders.")
    # p.add_argument("--dest",
    #                default="/n/fs/irom-testing/world_models/Ctrl-World/dataset_example/pi0_demos/videos",
    #                help="Destination base for trajectory folders.")
    p.add_argument("--pad", type=int, default=2, help="Zero-padding for trajectory IDs (e.g., 2 => 00, 01, ...).")
    p.add_argument("--start-id", type=int, default=0, help="Starting numeric ID offset.")
    p.add_argument("--copy", action="store_true", help="Copy instead of move.")
    p.add_argument("--skip-resized", action="store_true", help="Do not collect resized_* videos.")
    p.add_argument("--move-poses", action="store_true", help="Move robot_pose_*.h5 into <dest>/annotation/<ID>.h5")
    p.add_argument("--dry-run", action="store_true", default=False, help="Show planned operations without changing files.")
    return p.parse_args()

def find_timestamped_dirs(source: Path):
    items = []
    for p in source.iterdir():
        if not p.is_dir():
            continue
        m = FOLDER_RE.match(p.name)
        if not m:
            continue
        y, mo, d, H, M, S = map(int, m.groups())
        ts = datetime(y, mo, d, H, M, S)
        items.append((ts, p))
    items.sort(key=lambda x: x[0])  # chronological
    return items

def ensure_dir(path: Path, dry: bool):
    if dry:
        print(f"[DRY] mkdir -p {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)

def do_transfer(src: Path, dst: Path, copy: bool, dry: bool):
    action = "cp" if copy else "mv"
    if not src.exists():
        return False
    if dry:
        print(f"[DRY] {action} '{src}' -> '{dst}'")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)
    return True

def torch_resize():
    # read video as tensor: (T, C, H, W)
    video_dir = "/n/fs/irom-testing/world_models/Ctrl-World/dataset_example/irom_subset/videos"
    video_path=f"{video_dir}/01/1.mp4"
    video, _, info = read_video(video_path, output_format="TCHW")  # (T, C, H, W)
    fps = info["video_fps"]

    # resize to 144x256
    video_resized = F.interpolate(video, size=(144, 256), mode='bilinear', align_corners=False)

    # If your tensor is float in [0,1], convert to uint8 in [0,255]
    if video_resized.dtype == torch.float32 and video_resized.max() <= 1.0:
        video_uint8 = (video_resized * 255).to(torch.uint8)
    else:
        video_uint8 = video_resized.to(torch.uint8)

    # TorchVision expects shape (T, H, W, C)
    video_uint8 = video_uint8.permute(0, 2, 3, 1).cpu()  # move to CPU if on GPU

    # Save to disk
    write_video(out_video_path, video_uint8, fps=int(fps))


def cv_resize(video_path, out_video_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (320, 192))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (320, 192))
        out.write(resized)

    cap.release()
    out.release()

def main(args):
    
    source = Path(args.source).resolve()
    dest = Path(args.source).resolve() / "videos"

    print(f"Source: {source}")
    print(f"Dest:   {dest}")
    print(f"Mode:   {'COPY' if args.copy else 'MOVE'}; Resized={'SKIP' if args.skip_resized else 'COLLECT'}; Dry-run={args.dry_run}")
    timestamped = find_timestamped_dirs(source)
    if not timestamped:
        print("No timestamped subfolders found. Exiting.")
        return

    # Optional poses destination
    poses_dest = dest.parent / "annotation"
    if args.move_poses:
        ensure_dir(poses_dest, args.dry_run)

    plan_summary = []

    for idx, (_, folder) in enumerate(timestamped, start=args.start_id):
        traj_id = f"{idx:0{args.pad}d}"
        traj_dir = dest / traj_id
        resized_dir = traj_dir / "resized_cv"

        ensure_dir(traj_dir, args.dry_run)
        if not args.skip_resized:
            ensure_dir(resized_dir, args.dry_run)

        # For each camera kind, gather raw and resized paths
        cams_found = []
        for cam, cam_idx in CAM_INDEX.items():
            raw_name = f"{cam}.mp4"
            raw_src = folder / raw_name
            raw_dst = traj_dir / f"{cam_idx}.mp4"

            if do_transfer(raw_src, raw_dst, args.copy, args.dry_run):
                cams_found.append(cam)
            else:
                # Not fatal — some folders may lack right.mp4 or similar
                pass

            if not args.skip_resized:
                rname = f"resized_{cam}.mp4"
                rsrc = folder / rname
                rdst = resized_dir / f"{cam_idx}.mp4"
                do_transfer(rsrc, rdst, args.copy, args.dry_run)

        # Move pose file if requested
        pose_moved = False
        if args.move_poses:
            # Prefer the single .h5 in folder (matches your pattern robot_pose_*.h5)
            for item in folder.iterdir():
                if item.suffix == ".h5" and item.name.startswith("robot_pose_"):
                    pose_dst = poses_dest / f"{traj_id}"/f"{traj_id}.h5"
                    pose_moved = do_transfer(item, pose_dst, args.copy, args.dry_run)
                    break

        plan_summary.append({
            "id": traj_id,
            "src": str(folder),
            "cams": ",".join(cams_found) if cams_found else "-",
            "pose": "yes" if pose_moved else "no"
        })

        # If we MOVED everything out and the directory is empty, remove it
        if not args.copy and not args.dry_run:
            try:
                # Might still contain files if some not moved; ignore errors
                if not any(folder.iterdir()):
                    folder.rmdir()
            except Exception:
                pass

    print("\n=== Summary ===")
    for row in plan_summary:
        print(f"ID {row['id']}: {row['src']}  cams[{row['cams']}]  pose_moved={row['pose']}")

if __name__== "__main__":
    # parser = ArgumentParser()
    # parser.add_argument('--video_dir', type=str, default=None)
    # video_dir = "/n/fs/irom-testing/world_models/Ctrl-World/dataset_example/irom_subset/videos"
    # video_path=f"{video_dir}/01/0.mp4"
    # os.makedirs(f"{video_dir}/resized_cv", exist_ok=True)
    # out_video_path = f"{video_dir}/resized_cv/0.mp4"
    # cv_resize()

    ## Transfer videos
    args = parse_args()
    main(args)

    # Manual resize
    video_dir = args.dest
    
    for folder in os.listdir(video_dir):
        files = [vid_file for vid_file in os.listdir(os.path.join(video_dir, folder)) if vid_file.endswith(".mp4")]
        for file in files:
            video_path=f"{video_dir}/{folder}/{file}"
            out_folder = f"{video_dir}/{folder}/resized_wm"
            os.makedirs(f"{video_dir}/{folder}/resized_wm", exist_ok=True)
            out_video_path = f"{video_dir}/{folder}/resized_wm/{file}"
            cv_resize(video_path, out_video_path)