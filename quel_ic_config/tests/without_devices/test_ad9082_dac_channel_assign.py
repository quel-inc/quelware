from quel_ic_config.ad9082 import Ad9082ChannelAssignConfig, _Ad9082LaneConfigEnum


def test_dac_channel_assign_normal():
    dict0 = {
        "dac0": [_Ad9082LaneConfigEnum(i) for i in [2]],
        "dac1": [_Ad9082LaneConfigEnum(i) for i in [1]],
        "dac2": [_Ad9082LaneConfigEnum(i) for i in [5, 4, 0]],
        "dac3": [_Ad9082LaneConfigEnum(i) for i in [7, 6, 3]],
    }

    obj = Ad9082ChannelAssignConfig(**dict0)
    dict1 = obj.model_dump()
    assert dict0 == dict1
    assert obj.as_cpptype() == [0b0000_0100, 0b0000_0010, 0b0011_0001, 0b1100_1000]


def test_dac_channel_assign_wrong_order():
    dict0 = {
        "dac0": [_Ad9082LaneConfigEnum(i) for i in [2]],
        "dac1": [_Ad9082LaneConfigEnum(i) for i in [1]],
        "dac2": [_Ad9082LaneConfigEnum(i) for i in [0, 5, 4]],
        "dac3": [_Ad9082LaneConfigEnum(i) for i in [7, 6, 3]],
    }

    dict0_fixed = {
        "dac0": [_Ad9082LaneConfigEnum(i) for i in [2]],
        "dac1": [_Ad9082LaneConfigEnum(i) for i in [1]],
        "dac2": [_Ad9082LaneConfigEnum(i) for i in [5, 4, 0]],
        "dac3": [_Ad9082LaneConfigEnum(i) for i in [7, 6, 3]],
    }

    obj = Ad9082ChannelAssignConfig(**dict0)
    dict1 = obj.model_dump()
    assert dict0_fixed == dict1
    assert obj.as_cpptype() == [0b0000_0100, 0b0000_0010, 0b0011_0001, 0b1100_1000]
