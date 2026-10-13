import copy
import torch

models = {}

def register(name):
    """模型注册装饰器：把模型类按名字登记到 models 字典（如 'nerv_enc'/'nerv_model'），
    供 make() 根据配置中的 model.name 实例化。"""
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

# Registration function to associate model names with classes/functions
def register_model(name, model_class):
    """按名字直接注册模型类（非装饰器形式的辅助接口）。"""
    models[name] = model_class


def make(model_spec, args=None, load_sd=False):
    """根据 {'name': ..., 'args': {...}} 配置实例化模型；args 可覆盖部分默认参数；
    load_sd=True 时从 model_spec['sd'] 载入权重（宽松匹配）。"""
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    # except:
    #     print(model_spec)
    if load_sd:
        model.load_state_dict(model_spec['sd'], strict=False)
    return model


def compare_models(model1, model2):
    """逐键比较两个 state_dict 的张量是否完全相等（用于调试/验证权重一致性）。"""
    for k,v in model1.items():
        if torch.equal(v, model2[k].to(v.device)):
            continue
        else:
            return False
    return True

