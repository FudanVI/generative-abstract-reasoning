import json
import types


class ArgsParser:
    def __init__(self):
        self.__param_list__ = {}

    def add_argument(self, args, default=None, type=None):
        assert type is None or type in [int, float, str, list, bool]
        self.__param_list__[args] = {'default': default, 'type': type}

    def get_args(self, args_list, pre_args=None):
        if pre_args is None:
            pre_args = types.SimpleNamespace()
        for k, v in self.__param_list__.items():
            setattr(pre_args, k[2:], v['default'])
        args_list = args_list[1:]
        pos = 0
        while pos < len(args_list):
            key = args_list[pos]
            assert key[:2] == '--'
            if key in self.__param_list__.keys():
                key_config = self.__param_list__[key]
                value = args_list[pos + 1]
                if key_config['type'] is None or key_config['type'] in [list]:
                    value = json.loads(value)
                else:
                    value = key_config['type'](value)
                setattr(pre_args, key[2:], value)
            pos += 2
        return pre_args
