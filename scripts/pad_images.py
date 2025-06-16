from pathlib import Path
from functools import partial
import multiprocessing as mp
import traceback
from PIL import Image, ImageOps


def pad_to_rectangle(img, target_width=480, target_height=480, fill=(255, 255, 255)):

    w, h = img.size
    scale_w = target_width / w if w > target_width else 1.0
    scale_h = target_height / h if h > target_height else 1.0
    scale = min(scale_w, scale_h)

    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    pad_w = (target_width - img.width) // 2
    pad_h = (target_height - img.height) // 2

    return ImageOps.expand(
        img,
        border=(
            pad_w,
            pad_h,
            target_width - img.width - pad_w,
            target_height - img.height - pad_h,
        ),
        fill=fill,
    )


def process_one(src_file, dst_dir, target_width, target_height):

    try:
        with Image.open(src_file) as im:
            if im.mode != "RGB":
                im = im.convert("RGB")
            padded = pad_to_rectangle(
                im, target_width=target_width, target_height=target_height
            )
            out_path = dst_dir / src_file.name.lower()
            padded.save(out_path, format="PNG", optimize=True)
        return True, str(src_file)
    except Exception:
        return False, traceback.format_exc(limit=1)


def process_folder(src_dir, dst_dir, target_width=256, target_height=128, workers=8):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in exts]
    if not files:
        print(f"[skip] no images in {src_dir}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    worker_fn = partial(
        process_one,
        dst_dir=dst_dir,
        target_width=target_width,
        target_height=target_height,
    )
    ok = 0

    with mp.Pool(processes=min(workers, 8)) as pool:
        for i, (success, _) in enumerate(pool.imap_unordered(worker_fn, files), 1):
            if success:
                ok += 1
            if i % 200 == 0 or i == len(files):
                print(f"  {i}/{len(files)} done ({ok} ok)")

    print(f"[{src_dir.name}] completed: {ok}/{len(files)} succeeded")


def main():
    source = Path("data")
    target = Path("data_padded")

    target_width = 480
    target_height = 480

    for split in ["train", "validation", "test"]:
        src_imgs = source / split / "images"
        dst_imgs = target / split / "images"
        process_folder(
            src_imgs, dst_imgs, target_width=target_width, target_height=target_height
        )

        label_src = source / split / "labels.json"
        if label_src.exists():
            dst_split = target / split
            dst_split.mkdir(parents=True, exist_ok=True)
            (dst_split / "labels.json").write_text(label_src.read_text())


if __name__ == "__main__":

    mp.set_start_method("fork", force=True)
    main()
