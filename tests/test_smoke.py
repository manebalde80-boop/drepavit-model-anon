def test_import():
    import importlib
    m = importlib.import_module("drepavit.core")
    assert m is not None
