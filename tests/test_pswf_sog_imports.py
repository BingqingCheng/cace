import importlib


def test_pswf_sog_modules_importable():
    module_names = [
        "cace.modules.pswf",
        "cace.modules.pswf1e3",
        "cace.modules.pswf_qfield",
        "cace.modules.sog",
    ]
    for name in module_names:
        mod = importlib.import_module(name)
        assert mod is not None
