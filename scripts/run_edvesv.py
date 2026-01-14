from __future__ import annotations
import argparse

from edvesv.device import pick_device
from edvesv.cropping import FullFrameCropper, FixedCropper, YoloBestBoxCropper
from edvesv.runner import RunConfig, run_echonet_csv, run_uab_pkl

def build_cropper(args):
    if args.crop_mode == "full":
        return FullFrameCropper()
    if args.crop_mode == "fixed":
        if args.fixed_crop is None or len(args.fixed_crop) != 4:
            raise SystemExit("--fixed_crop requires 4 ints: X1 Y1 X2 Y2")
        x1,y1,x2,y2 = args.fixed_crop
        return FixedCropper(x1=x1, y1=y1, x2=x2, y2=y2)
    if args.crop_mode == "yolo":
        if not args.yolo_weights:
            raise SystemExit("--yolo_weights is required for crop_mode=yolo")
        return YoloBestBoxCropper(args.yolo_weights)
    raise SystemExit(f"Unknown crop_mode: {args.crop_mode}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["echonet_csv", "uab_pkl"], required=True)

    # common
    p.add_argument("--crop_mode", choices=["yolo", "full", "fixed"], default="yolo")
    p.add_argument("--fixed_crop", nargs=4, type=int, default=None, help="X1 Y1 X2 Y2 for fixed crop")
    p.add_argument("--yolo_weights", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    p.add_argument("--prominence", type=float, default=0.6)
    p.add_argument("--rpca_max_iter", type=int, default=1000)
    p.add_argument("--rpca_tol", type=float, default=1e-6)
    p.add_argument("--start_idx", type=int, default=None)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--output_csv", type=str, required=True)

    # echonet_csv
    p.add_argument("--video_dir", type=str, default=None)
    p.add_argument("--input_csv", type=str, default=None)

    # uab_pkl
    p.add_argument("--npy_base_dir", type=str, default=None)
    p.add_argument("--input_pkl", type=str, default=None)
    p.add_argument("--id_col", type=str, default=None)
    p.add_argument("--ed_col", type=str, default=None)
    p.add_argument("--es_col", type=str, default=None)
    p.add_argument("--npy_col", type=str, default=None)

    args = p.parse_args()

    device = pick_device(args.device)
    cropper = build_cropper(args)

    cfg = RunConfig(
        device=device,
        crop_mode=args.crop_mode,
        fixed_crop=tuple(args.fixed_crop) if args.fixed_crop else None,
        yolo_weights=args.yolo_weights,
        prominence=args.prominence,
        rpca_max_iter=args.rpca_max_iter,
        rpca_tol=args.rpca_tol,
    )

    print(f"Using device: {device}")
    print(f"Crop mode: {args.crop_mode}")

    if args.dataset == "echonet_csv":
        if not args.video_dir or not args.input_csv:
            raise SystemExit("--video_dir and --input_csv are required for dataset=echonet_csv")
        run_echonet_csv(
            video_dir=args.video_dir,
            input_csv=args.input_csv,
            output_csv=args.output_csv,
            cropper=cropper,
            cfg=cfg,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )
        print(f"Saved: {args.output_csv}")
        return

    if args.dataset == "uab_pkl":
        if not args.npy_base_dir or not args.input_pkl:
            raise SystemExit("--npy_base_dir and --input_pkl are required for dataset=uab_pkl")
        run_uab_pkl(
            npy_base_dir=args.npy_base_dir,
            input_pkl=args.input_pkl,
            output_csv=args.output_csv,
            cropper=cropper,
            cfg=cfg,
            id_col=args.id_col,
            ed_col=args.ed_col,
            es_col=args.es_col,
            npy_col=args.npy_col,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )
        print(f"Saved: {args.output_csv}")
        return

if __name__ == "__main__":
    main()
