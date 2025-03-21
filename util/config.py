import os
import json
support_types = ('str', 'int', 'bool', 'float', 'none')
def convert_param(original_lists):
    '''
    传过来的参数是一个列表，至少包含，参数的类型和参数，比如是 "epochs"   : ["int",   "1"]
    加个断言，对于错误的类型和不符合要求的参数抛出异常
    '''
    assert isinstance(original_lists, list), 'The type is not right : {:}'.format(original_lists)
    ctype, value = original_lists[0], original_lists[1]
    assert ctype in support_types, 'Ctype={:}, support={:}'.format(ctype, support_types)
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == 'int': x = int(x)
        elif ctype == 'str': x = str(x)
        elif ctype == 'bool': x = bool(int(x))
        elif ctype == 'float': x = float(x)
        elif ctype == 'none':
            assert x == 'None', 'for none type, the value must be None instead of {:}'.format(x)
            x = None
        else:
            raise TypeError('Does not know this type : {:}'.format(ctype))
        outs.append(x)
    '''
    原始的不是一个列表的话再将其转化为单个值，比如 bool 类型的 True 和 False
    '''
    if not is_list:
        outs = outs[0]
    return outs
'''
加载配置文件，不存在的话报错，否则直接以 json 的方式打开当前的 path 地址
并且将其内容解构为一个 python 字典
'''
def load_config(path):
    path = str(path)
    assert os.path.exists(path), 'Can not find {:}'.format(path)
    with open(path, 'r') as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    return content

if __name__ == '__main__':
    config = load_config('../config/optim_config')
    print('Finish!')
