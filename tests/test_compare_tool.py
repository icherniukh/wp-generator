"""tools/compare.py: variant parsing, sheet layout, shared palettes and diff."""

import numpy as np
import pytest
from PIL import Image

from tools import compare


def test_parses_numbers_bools_strings_and_null():
    overrides = compare.parse_variant("preset=aura,noise=0.15,glass=true,complexity=4,halftone=null,engine=blobs")
    assert overrides == {"preset": "aura", "noise": 0.15, "glass": True, "complexity": 4, "halftone": None, "engine": "blobs"}


def test_dotted_keys_build_nested_stage_params():
    overrides = compare.parse_variant("stage_params.shading.tint=0.6,stage_params.shading.angle=40,stage_params.bloom.rays=2")
    assert overrides == {"stage_params": {"shading": {"tint": 0.6, "angle": 40}, "bloom": {"rays": 2}}}


def test_plus_makes_lists_and_json_lists_survive_commas():
    assert compare.parse_variant("warps=noise+liquify") == {"warps": ["noise", "liquify"]}
    assert compare.parse_variant("warps=noise") == {"warps": ["noise"]}
    assert compare.parse_variant("fractal_center=[0.5,-0.3],noise=0.1") == {"fractal_center": (0.5, -0.3), "noise": 0.1}
    assert compare.parse_variant("colors=#112233+#445566") == {"colors": [(17, 34, 51), (68, 85, 102)]}


def test_empty_variant_is_the_defaults():
    assert compare.parse_variant("") == {}


@pytest.mark.parametrize("text", ["nosuchkey=1", "noise", "seed=3", "width=10", "noise.sub=1", "stage_params.shading=1"])
def test_bad_variants_fail_clearly(text):
    with pytest.raises(compare.VariantError):
        compare.parse_variant(text)


def test_unknown_key_message_lists_valid_fields():
    with pytest.raises(compare.VariantError) as info:
        compare.parse_variant("nosuchkey=1")
    message = str(info.value)
    assert "nosuchkey" in message and "field_grain" in message and "stage_params" in message


def test_size_and_seed_parsing():
    assert compare.parse_size("640x360", "--res") == (640, 360)
    assert compare.parse_seeds("1,2,5-7") == [1, 2, 5, 6, 7]
    with pytest.raises(compare.VariantError):
        compare.parse_size("640", "--res")
    with pytest.raises(compare.VariantError):
        compare.parse_seeds("a")


def test_wrap_text_stays_inside_the_width():
    font = compare.load_font()
    lines = compare.wrap_text("preset=aura,field_grain=0.15,stage_params.shading.tint=0.6," * 3, font, 120, max_lines=4)
    assert 1 < len(lines) <= 4
    assert all(font.getlength(line) <= 120 for line in lines)
    assert lines[-1].endswith("...")


def test_two_by_two_sheet_matches_the_layout():
    variants = ["preset=aura", "preset=aura,noise=0"]
    parsed = [compare.parse_variant(v) for v in variants]
    grid = compare.render_cells(parsed, [1, 2], 64, 40)
    sheet, layout = compare.make_sheet(grid, variants, [1, 2], 64, 40)
    assert sheet.size == layout.size
    assert len(layout.boxes) == 4
    for (vi, si), tiles in layout.boxes.items():
        left, top, right, bottom = tiles["image"]
        assert (right - left, bottom - top) == (64, 40)
        assert right <= sheet.width and bottom <= sheet.height
        assert np.array_equal(np.asarray(sheet.crop(tiles["image"])), np.asarray(grid[vi][si].image))
    # label column + two cells + gaps + padding; header, two rows, footer.
    assert sheet.width == compare.PAD * 2 + compare.ROW_LABEL_WIDTH + 2 * 64 + compare.GAP
    assert sheet.height > 2 * 40 + compare.GAP


def test_crop_and_diff_add_tiles_and_transpose_swaps_axes():
    variants = ["noise=0.1", "noise=0.2"]
    parsed = [compare.parse_variant(v) for v in variants]
    grid = compare.render_cells(parsed, [1, 2, 3], 64, 40)
    plain, _ = compare.make_sheet(grid, variants, [1, 2, 3], 64, 40)
    full, layout = compare.make_sheet(grid, variants, [1, 2, 3], 64, 40, crop=(30, 20), diff=True)
    assert layout.boxes[(0, 0)]["crop"][2] - layout.boxes[(0, 0)]["crop"][0] == 30
    assert full.width > plain.width and full.height > plain.height
    tr, tr_layout = compare.make_sheet(grid, variants, [1, 2, 3], 64, 40, transpose=True)
    assert tr_layout.boxes[(1, 2)]["image"][1] > tr_layout.boxes[(1, 0)]["image"][1]  # seeds run down
    assert tr_layout.boxes[(1, 0)]["image"][0] > tr_layout.boxes[(0, 0)]["image"][0]  # variants run across


def test_palette_is_shared_across_rows_unless_the_variant_fixes_it():
    # Nothing fixes colors, so these would otherwise sample different random palettes.
    parsed = [compare.parse_variant(v) for v in ("engine=simplex", "engine=blobs,complexity=3,noise=0")]
    grid = compare.render_cells(parsed, [3, 4], 48, 32)
    for si in range(2):
        assert grid[0][si].colors == grid[1][si].colors
        assert grid[1][si].palette_shared and not grid[0][si].palette_shared
    assert grid[0][0].colors != grid[0][1].colors

    unshared = compare.render_cells(parsed, [3, 4], 48, 32, share_palette=False)
    assert any(unshared[0][si].colors != unshared[1][si].colors for si in range(2))

    own = [compare.parse_variant(v) for v in ("engine=simplex", "colors=#ff0000+#0000ff")]
    grid = compare.render_cells(own, [3], 48, 32)
    assert not grid[1][0].palette_shared
    assert "#ff0000" in grid[1][0].colors[0] or grid[1][0].colors != grid[0][0].colors


def test_diff_of_identical_variants_is_black():
    variants = ["preset=aura,noise=0.1", "preset=aura,noise=0.1"]
    parsed = [compare.parse_variant(v) for v in variants]
    grid = compare.render_cells(parsed, [1, 2], 64, 40)
    sheet, layout = compare.make_sheet(grid, variants, [1, 2], 64, 40, diff=True, crop=(20, 20))
    for tiles in layout.boxes.values():
        assert not np.asarray(sheet.crop(tiles["diff"])).any()
        assert not np.asarray(sheet.crop(tiles["diff_crop"])).any()


def test_diff_amplifies_differences():
    a = Image.new("RGB", (4, 4), (10, 10, 10))
    b = Image.new("RGB", (4, 4), (13, 10, 10))
    out = np.asarray(compare.diff_image(a, b, gain=8))
    assert out[0, 0].tolist() == [24, 0, 0]


def test_cli_writes_the_sheet_and_prints_the_path(tmp_path, capsys):
    out = tmp_path / "nested" / "sheet.png"
    code = compare.main(["--variant", "noise=0.1", "--variant", "noise=0", "--seeds", "1,2", "--res", "48x32", "--out", str(out), "--workers", "2"])
    assert code == 0
    assert capsys.readouterr().out.strip() == str(out.resolve())
    assert Image.open(out).size[0] > 96


def test_cli_json_reports_every_cell(tmp_path, capsys):
    out = tmp_path / "s.png"
    compare.main(["--variant", "noise=0.1", "--variant", "noise=0", "--seeds", "1,2", "--res", "48x32", "--out", str(out), "--json"])
    import json

    report = json.loads(capsys.readouterr().out)
    assert len(report["cells"]) == 4
    assert report["cells"][0]["resolved"]["noise"] == 0.1


def test_cli_reports_bad_input_without_a_traceback(tmp_path, capsys):
    assert compare.main(["--variant", "nosuchkey=1", "--out", str(tmp_path / "x.png")]) == 2
    assert "Valid keys" in capsys.readouterr().err
    assert compare.main(["--variant", "preset=nosuchpreset", "--res", "32x32", "--out", str(tmp_path / "x.png")]) == 2
