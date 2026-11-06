from qsa.mm_data import make_specs


def test_specs_are_deterministic():
    a = make_specs(0)
    b = make_specs(0)
    assert [s["png_sha256"] for s in a] == [s["png_sha256"] for s in b]
    assert len(a) == 3


def test_options_contain_answer():
    for spec in make_specs(0):
        assert len(spec["options"]) == 4
        assert 0 <= spec["answer"] < 4
        assert spec["image_size"] == [640, 448]


def test_kinds_present():
    kinds = {spec["kind"] for spec in make_specs(0)}
    assert {"code", "table", "shapes"} <= kinds
