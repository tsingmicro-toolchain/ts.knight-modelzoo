import os
import re
import sys
import inspect
from importlib import import_module

current_path = os.path.dirname(__file__)
modules = os.listdir(current_path)
func_name_dict = {}

def load_function(module, function_name):
    import_module(module)
    all_func = inspect.getmembers(sys.modules[module], inspect.isfunction)
    for name, func in all_func:
        if name == function_name and name in func_name_dict:
            print('\033[31m{}\033[0m'.format(("Load infer functions failed, function {} is already registered."
                                              .format(name))))
            sys.exit(-1)
        func_name_dict[name] = func

def dynamic_import_infer_functions(function_name):
    # 加载一级目录
    for module in modules:
        relative_path = 'custom_models.infer.' + module
        relative_file = os.path.join('custom_models', 'infer', module)
        if re.match(r"[^_][\w]{1,50}\.py$", module):
            module = relative_path[:-3]
            load_function(module, function_name)
        elif os.path.isdir(relative_file):
            relative_subdir = relative_file.split('/')[-1]
            # 加载二级目录
            sub_modules = os.listdir(relative_file)
            for sub_module in sub_modules:
                if re.match(r"[^_][\w]{1,50}\.py$", sub_module):
                    sub_module = sub_module.split('.')[0]
                    sub_module = 'custom_models.infer.' + relative_subdir + '.' + sub_module
                    load_function(sub_module, function_name)
    return func_name_dict
