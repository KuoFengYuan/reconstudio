"""pipeline.fusion: staging two matted passes of one subject into one COLMAP input.

The failure this module exists to prevent is silent: COLMAP resolves a mask by
the image's path *relative to image_path*, so if the staged mask folder does not
mirror the staged image folder name, every image reports `Mask: No` and the
background is reconstructed anyway — a plausible-looking model of the wrong
thing. So the mirroring, and the input forms that must resolve to it, are
pinned here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from pipeline import fusion

Image = pytest.importorskip("PIL.Image", reason="the size scan reads image headers")


def _pass_on_disk(root: Path, n: int = 3, size=(64, 48), masks: int | None = None) -> Path:
    """A matted capture: `<root>/images_2560/*.jpg` + `.../no_bg/{masks,cutout}`.
    Returns the `no_bg` folder — what the 去背 job reports and the form takes."""
    images = root / "images_2560"
    no_bg = images / "no_bg"
    for d in (images, no_bg / "masks", no_bg / "cutout"):
        d.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", size, (90, 60, 40)).save(images / f"IMG_{i:04d}.jpg", quality=80)
        if masks is None or i < masks:
            Image.new("L", size, 255).save(no_bg / "masks" / f"IMG_{i:04d}.png")
    return no_bg


# --- input resolution ------------------------------------------------------- #
def test_resolves_the_no_bg_folder(tmp_path):
    no_bg = _pass_on_disk(tmp_path / "up")
    got, images, masks = fusion.resolve_pass(no_bg)
    assert got == no_bg
    assert images == tmp_path / "up" / "images_2560"
    assert masks == no_bg / "masks"


@pytest.mark.parametrize("suffix", ["", "/masks", "/cutout"])
def test_resolves_the_forms_people_actually_paste(tmp_path, suffix):
    """The image folder above it, and the two children the review pages show,
    all name the same capture — accepting only one of them would just be a
    paper cut with a confusing error message."""
    no_bg = _pass_on_disk(tmp_path / "up")
    given = (no_bg.parent if suffix == "" else Path(str(no_bg) + suffix))
    assert fusion.resolve_pass(given)[0] == no_bg


def test_a_folder_that_was_never_matted_is_rejected(tmp_path):
    plain = tmp_path / "photos"
    plain.mkdir()
    with pytest.raises(ValueError, match="去背"):
        fusion.resolve_pass(plain)


def test_matte_without_the_masks_output_is_rejected(tmp_path):
    # 輸出 = "只要 cutout/" leaves no single-channel masks, and those are the
    # ones feature extraction needs.
    no_bg = _pass_on_disk(tmp_path / "up")
    for f in (no_bg / "masks").iterdir():
        f.unlink()
    (no_bg / "masks").rmdir()
    with pytest.raises(FileNotFoundError, match="masks"):
        fusion.resolve_pass(no_bg)


# --- inspect ---------------------------------------------------------------- #
def test_inspect_names_passes_after_their_dataset_folder(tmp_path):
    a = _pass_on_disk(tmp_path / "vase_up", n=3)
    b = _pass_on_disk(tmp_path / "vase_down", n=2)
    passes = fusion.inspect([str(a), str(b)])
    assert [p.name for p in passes] == ["vase_up", "vase_down"]
    assert [(p.n_images, p.n_masks) for p in passes] == [(3, 3), (2, 2)]


def test_inspect_needs_at_least_two_passes(tmp_path):
    a = _pass_on_disk(tmp_path / "up")
    with pytest.raises(ValueError, match="兩組"):
        fusion.inspect([str(a)])


def test_inspect_reports_missing_masks_as_a_warning_not_a_crash(tmp_path):
    a = _pass_on_disk(tmp_path / "up", n=4, masks=2)      # half matted
    b = _pass_on_disk(tmp_path / "down", n=2)
    passes = fusion.inspect([str(a), str(b)])
    assert passes[0].unmatched == 2
    _, warnings = fusion.summarize(passes)
    assert any("沒有對應的遮罩" in w for w in warnings)


def test_mixed_sizes_warn_against_one_shared_camera(tmp_path):
    a = _pass_on_disk(tmp_path / "up", size=(64, 48))
    b = _pass_on_disk(tmp_path / "down", size=(48, 64))
    _, warnings = fusion.summarize(fusion.inspect([str(a), str(b)]))
    assert any("per_folder" in w for w in warnings)


def test_same_dataset_name_twice_still_gets_two_folders(tmp_path):
    a = _pass_on_disk(tmp_path / "x" / "vase")
    b = _pass_on_disk(tmp_path / "y" / "vase")
    names = [p.name for p in fusion.inspect([str(a), str(b)])]
    assert len(set(names)) == 2, names


# --- staging ---------------------------------------------------------------- #
def test_stage_mirrors_the_image_folder_names_under_masks(tmp_path):
    a = _pass_on_disk(tmp_path / "up", n=2)
    b = _pass_on_disk(tmp_path / "down", n=2)
    passes = fusion.inspect([str(a), str(b)])
    images_root, masks_root, ws = fusion.stage(passes, tmp_path / "fusion")

    assert ws == tmp_path / "fusion" / "colmap"
    for p in passes:
        img_link, msk_link = images_root / p.name, masks_root / p.name
        assert img_link.is_symlink() and msk_link.is_symlink()
        assert Path(os.readlink(img_link)) == p.images_dir
        assert Path(os.readlink(msk_link)) == p.masks_dir
        # the contract: images/<pass>/X.jpg must resolve to masks/<pass>/X.png
        for img in img_link.iterdir():
            if img.suffix == ".jpg":
                assert (msk_link / f"{img.stem}.png").is_file()


def test_stage_links_rather_than_copies(tmp_path):
    a, b = _pass_on_disk(tmp_path / "up"), _pass_on_disk(tmp_path / "down")
    images_root, _, _ = fusion.stage(fusion.inspect([str(a), str(b)]), tmp_path / "f")
    assert all(p.is_symlink() for p in images_root.iterdir())


def test_stage_is_idempotent_and_repoints_a_stale_link(tmp_path):
    a, b = _pass_on_disk(tmp_path / "up"), _pass_on_disk(tmp_path / "down")
    out = tmp_path / "fusion"
    passes = fusion.inspect([str(a), str(b)])
    fusion.stage(passes, out)
    fusion.stage(passes, out)                      # again: must not raise
    # point one link somewhere stale, then re-stage
    link = out / "images" / passes[0].name
    link.unlink()
    link.symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    fusion.stage(passes, out)
    assert Path(os.readlink(link)) == passes[0].images_dir


def test_stage_refuses_to_replace_a_real_directory(tmp_path):
    a, b = _pass_on_disk(tmp_path / "up"), _pass_on_disk(tmp_path / "down")
    passes = fusion.inspect([str(a), str(b)])
    out = tmp_path / "fusion"
    real = out / "images" / passes[0].name
    real.mkdir(parents=True)
    (real / "someones_data.txt").write_text("x")
    with pytest.raises(FileExistsError):
        fusion.stage(passes, out)
    assert (real / "someones_data.txt").is_file()      # never deleted


def test_default_out_dir_is_a_fusion_folder_beside_the_two_datasets(tmp_path):
    a = _pass_on_disk(tmp_path / "proj" / "up")
    b = _pass_on_disk(tmp_path / "proj" / "down")
    got = fusion.default_out_dir(fusion.inspect([str(a), str(b)]))
    assert got == (tmp_path / "proj").resolve() / "fusion"
