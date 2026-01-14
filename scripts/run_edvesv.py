from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Optional
import tempfile
import csv

from edvesv.device import pick_device
from edvesv.cropping import FullFrameCropper, FixedCropper, YoloBestBoxCropper
from edvesv.runner import RunConfig, run_echonet_csv, run_uab_pkl


def build_cropper(args):
    if args.crop_mode == "full":
        return FullFrameCropper()
    if args.crop_mode == "fixed":
        if args.fixed_crop is None or len(args.fixed_crop) != 4:
            raise SystemExit("--fixed_crop requires 4 ints: X1 Y1 X2 Y2")
        x1, y1, x2, y2 = args.fixed_crop
        return FixedCropper(x1=x1, y1=y1, x2=x2, y2=y2)
    if args.crop_mode == "yolo":
        if not args.yolo_weights:
            raise SystemExit("--yolo_weights is required for crop_mode=yolo")
        return YoloBestBoxCropper(args.yolo_weights)
    raise SystemExit(f"Unknown crop_mode: {args.crop_mode}")


def _normalize_filenames(names: List[str]) -> List[str]:
    """Accept names with or without extension; normalize to FileName stem (no .avi)."""
    out: List[str] = []
    for n in names:
        n = str(n).strip()
        if not n:
            continue
        n = os.path.basename(n)
        stem, ext = os.path.splitext(n)
        if ext.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
            out.append(stem)
        else:
            out.append(n)
    return out


def _read_file_list(file_list_path: str) -> List[str]:
    p = Path(file_list_path)
    if not p.exists():
        raise SystemExit(f"--file_list not found: {file_list_path}")
    lines: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
    return _normalize_filenames(lines)


def _write_temp_echonet_csv(file_names: List[str], out_path: Optional[str] = None) -> str:
    """
    Create a temporary EchoNet-style CSV with minimal required columns.
    EDV/ESV are left blank since these modes are inference-only.
    """
    if out_path is None:
        fd, tmp_path = tempfile.mkstemp(prefix="edvesv_filenames_", suffix=".csv")
        os.close(fd)
        out_path = tmp_path

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["FileName", "EDV", "ESV"])
        for fn in file_names:
            w.writerow([fn, "", ""])
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        choices=["echonet_csv", "uab_pkl", "single_avi", "echonet_list"],
        required=True,
        help=(
            "echonet_csv: AVI + CSV with FileName,EDV,ESV (labeled/eval)\n"
            "uab_pkl: NPY + PKL table (labeled/eval)\n"
            "single_avi: single AVI inference (no labels)\n"
            "echonet_list: batch AVI inference from a text list of filenames (no labels)"
        ),
    )

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
    p.add_argument("--video_dir", type=str, default=None, help="Directory containing videos (e.g., *.avi)")
    p.add_argument("--input_csv", type=str, default=None, help="CSV with FileName,EDV,ESV (and optionally more columns)")

    # uab_pkl
    p.add_argument("--npy_base_dir", type=str, default=None)
    p.add_argument("--input_pkl", type=str, default=None)
    p.add_argument("--id_col", type=str, default=None)
    p.add_argument("--ed_col", type=str, default=None)
    p.add_argument("--es_col", type=str, default=None)
    p.add_argument("--npy_col", type=str, default=None)

    # single_avi
    p.add_argument("--video_path", type=str, default=None, help="Path to a single .avi file (for dataset=single_avi)")
    p.add_argument("--case_id", type=str, default=None, help="Optional ID to store as FileName in output (single_avi)")

    # echonet_list
    p.add_argument(
        "--file_list",
        type=str,
        default=None,
        help="Text file with one FileName per line (with or without .avi). Lines starting with # are ignored.",
    )

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

    # Inference-only modes reuse run_echonet_csv by generating a temporary CSV with FileName + empty EDV/ESV.
    tmp_csv: Optional[str] = None
    try:
        if args.dataset == "single_avi":
            if not args.video_path:
                raise SystemExit("--video_path is required for dataset=single_avi")
            vp = Path(args.video_path)
            if not vp.exists():
                raise SystemExit(f"--video_path not found: {args.video_path}")

            video_dir = str(vp.parent)
            file_name = args.case_id.strip() if args.case_id else vp.stem
            tmp_csv = _write_temp_echonet_csv([file_name])

            run_echonet_csv(
                video_dir=video_dir,
                input_csv=tmp_csv,
                output_csv=args.output_csv,
                cropper=cropper,
                cfg=cfg,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
            )
            print(f"Saved: {args.output_csv}")
            return

        if args.dataset == "echonet_list":
            if not args.video_dir or not args.file_list:
                raise SystemExit("--video_dir and --file_list are required for dataset=echonet_list")
            file_names = _read_file_list(args.file_list)
            if len(file_names) == 0:
                raise SystemExit("--file_list contained no filenames after filtering comments/blank lines")

            tmp_csv = _write_temp_echonet_csv(file_names)

            run_echonet_csv(
                video_dir=args.video_dir,
                input_csv=tmp_csv,
                output_csv=args.output_csv,
                cropper=cropper,
                cfg=cfg,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
            )
            print(f"Saved: {args.output_csv}")
            return

    finally:
        if tmp_csv and os.path.exists(tmp_csv):
            try:
                os.remove(tmp_csv)
            except Exception:
                pass


if __name__ == "__main__":
    main()
