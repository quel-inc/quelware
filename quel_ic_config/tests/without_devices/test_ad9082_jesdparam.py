import pytest
from pydantic import ValidationError

from quel_ic_config import Ad9082JesdParam


def test_jesdparam_normal():
    dict0 = {
        "l": 101,
        "f": 102,
        "m": 103,
        "s": 104,
        "hd": 105,
        "k": 106,
        "n": 107,
        "np": 108,
        "cf": 109,
        "cs": 110,
        "did": 111,
        "bid": 112,
        "lid0": 113,
        "subclass": 114,
        "scr": 115,
        "duallink": 116,
        "jesdv": 117,
        "mode_id": 118,
        "mode_c2r_en": 119,
        "mode_s_sel": 120,
    }

    obj = Ad9082JesdParam(**dict0)
    dict1 = obj.model_dump()
    assert dict0 == dict1

    cmsobj = obj.as_cpptype()
    for k, v in dict0.items():
        assert getattr(cmsobj, k) == dict0[k]

    attrs = {k for k in dir(cmsobj) if not k.startswith("_")}
    assert attrs == set(dict0.keys())

    # Notes: the following test case is not required since mypy detects via static analysis.
    # with pytest.raises(ValueError) as _:
    #    obj.non_existent = 500


def test_jesdparam_lacking():
    """testing if I use pydantic in a correct way or not."""
    dict0_lacking = {
        "l": 101,
        "f": 102,
        "m": 103,
        "s": 104,
        "hd": 105,
        "k": 106,
        "n": 107,
        "np": 108,
        "cf": 109,
        "cs": 110,
        "bid": 112,
        "lid0": 113,
        "subclass": 114,
        "scr": 115,
        "duallink": 116,
        "jesdv": 117,
        "mode_id": 118,
        "mode_c2r_en": 119,
        "mode_s_sel": 120,
    }

    with pytest.raises(ValidationError) as _:
        _ = Ad9082JesdParam(**dict0_lacking)


def test_jesdparam_extra():
    """testing if I use pydantic in a correct way or not."""
    dict0_extra = {
        "l": 101,
        "f": 102,
        "m": 103,
        "s": 104,
        "hd": 105,
        "k": 106,
        "n": 107,
        "np": 108,
        "cf": 109,
        "cs": 110,
        "did": 111,
        "bid": 112,
        "lid0": 113,
        "subclass": 114,
        "scr": 115,
        "duallink": 116,
        "jesdv": 117,
        "mode_id": 118,
        "mode_c2r_en": 119,
        "mode_s_sel": 120,
        "hogehoge": 600,
    }

    with pytest.raises(ValidationError) as _:
        _ = Ad9082JesdParam(**dict0_extra)
