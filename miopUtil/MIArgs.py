from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import shlex

# Mapping of flags to normalized names (both short and long forms)
FLAG_MAPPING = {
    # Operation type
    '-F': 'forw', '--forw': 'forw',
    # Input tensor
    '-n': 'batchsize', '--batchsize': 'batchsize',
    '-c': 'in_channels', '--in_channels': 'in_channels',
    '-H': 'in_h', '--in_h': 'in_h',
    '-W': 'in_w', '--in_w': 'in_w',
    '-!': 'in_d', '--in_d': 'in_d',
    # Output channels
    '-k': 'out_channels', '--out_channels': 'out_channels',
    # Kernel
    '-y': 'fil_h', '--fil_h': 'fil_h',
    '-x': 'fil_w', '--fil_w': 'fil_w',
    '-@': 'fil_d', '--fil_d': 'fil_d',
    # Padding
    '-p': 'pad_h', '--pad_h': 'pad_h',
    '-q': 'pad_w', '--pad_w': 'pad_w',
    '-$': 'pad_d', '--pad_d': 'pad_d',
    # Stride
    '-u': 'conv_stride_h', '--conv_stride_h': 'conv_stride_h',
    '-v': 'conv_stride_w', '--conv_stride_w': 'conv_stride_w',
    '-#': 'conv_stride_d', '--conv_stride_d': 'conv_stride_d',
    # Dilation
    '-l': 'dilation_h', '--dilation_h': 'dilation_h',
    '-j': 'dilation_w', '--dilation_w': 'dilation_w',
    '-^': 'dilation_d', '--dilation_d': 'dilation_d',
    # Groups
    '-g': 'group_count', '--group_count': 'group_count',
    # Spatial dimension
    '-_': 'spatial_dim', '--spatial_dim': 'spatial_dim',
    # Solver and data type
    '-S': 'solution', '--solution': 'solution',
    '-t': 'time', '--time': 'time',
    '-V': 'verify', '--verify': 'verify',
    '-m': 'mode', '--mode': 'mode',
    # Data loading
    '-d': 'in_data', '--in_data': 'in_data',
    '-e': 'weights', '--weights': 'weights',
    '-D': 'dout_data', '--dout_data': 'dout_data',
    # Bias
    '-b': 'bias', '--bias': 'bias',
    '-a': 'in_bias', '--in_bias': 'in_bias',
    # Additional parameters
    '-i': 'iter', '--iter': 'iter',
    '-z': 'pad_mode', '--pad_mode': 'pad_mode',
    '-f': 'fil_layout', '--fil_layout': 'fil_layout',
    '-I': 'in_layout', '--in_layout': 'in_layout',
    '-O': 'out_layout', '--out_layout': 'out_layout',
    '-~': 'gpubuffer_check', '--gpubuffer_check': 'gpubuffer_check',
    '-w': 'wall', '--wall': 'wall',
    '-P': 'printconv', '--printconv': 'printconv',
    '-r': 'pad_val', '--pad_val': 'pad_val',
    '-o': 'dump_output', '--dump_output': 'dump_output',
    '-s': 'search', '--search': 'search',
    '-C': 'verification_cache', '--verification_cache': 'verification_cache',
    '-G': 'gpualloc', '--gpualloc': 'gpualloc',
    # Shape format
    '--shapeformat': 'shapeformat',
    '--trace': 'trace',
    '--event': 'event',
    '--gpu': 'gpu',
    '--dbshape': 'dbshape',
    '--warmup': 'warmup',
    '--cpu' : 'cpu',
    '--save_db': "save_db"
}

CONVERTERS = {
    'forw': int, 'batchsize': int, 'in_channels': int, 'in_h': int, 'in_w': int,
    'in_d': int, 'out_channels': int, 'fil_h': int, 'fil_w': int, 'fil_d': int,
    'pad_h': int, 'pad_w': int, 'pad_d': int, 'conv_stride_h': int, 'conv_stride_w': int,
    'conv_stride_d': int, 'dilation_h': int, 'dilation_w': int, 'dilation_d': int,
    'group_count': int, 'spatial_dim': int, 'solution': int, 'time': int, 'verify': int,
    'bias': int, 'iter': int, 'gpubuffer_check': int, 'wall': int, 'printconv': int,
    'pad_val': int, 'dump_output': int, 'search': int,
    # Strings (keep as-is)
    'mode': str, 'in_data': str, 'weights': str, 'dout_data': str, 'in_bias': str,
    'pad_mode': str, 'fil_layout': str, 'in_layout': str, 'out_layout': str,
    'verification_cache': str, 'shapeformat': str, 'trace': str, 'event': str,
    'gpu': int, 'dbshape': int, 'warmup': int, 'cpu': int, 'gpualloc': int, 'save_db': int
}

class MiopenDataType(Enum):
    miopenHalf = 0  # 16-bit floating point (Fully supported)
    miopenFloat = 1  # 32-bit floating point (Fully supported)
    miopenInt32 = 2  # 32-bit integer (Partially supported)
    miopenInt8 = 3  # 8-bit integer (Partially supported)
    # miopenInt8x4 = 4  # Pack of 4x Int8 in NCHW_VECT_C format (Support discontinued)
    miopenBFloat16 = 5  # 16-bit binary floating point (8-bit exponent, 7-bit fraction) (Partially supported)
    miopenDouble = 6  # 64-bit floating point (Partially supported)
    miopenFloat8_fnuz = 7
    miopenBFloat8_fnuz = 8
    miopenInt64 = 9

def get_data_type_name(data_type):
    return {
        MiopenDataType.miopenHalf:          "FP16",
        MiopenDataType.miopenFloat:         "FP32",
        MiopenDataType.miopenInt32:         "INT32",
        MiopenDataType.miopenInt8:          "INT8",
        MiopenDataType.miopenBFloat16:      "BF16",
        MiopenDataType.miopenDouble:        "FP64",
        MiopenDataType.miopenFloat8_fnuz:   "FP8",
        MiopenDataType.miopenBFloat8_fnuz:  "BF8",
        MiopenDataType.miopenInt64:         "INT64"
    }.get(data_type, f"Unknown({data_type})")

def get_direction_str(forw):
    if forw == 1:
        return "F"
    elif forw == 2:
        return "B"
    elif forw == 4:
        return "W"
    else:
        return "unknown"

def ParseRunList(file_path):
    convRunList = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        command_line = line.strip()
        if command_line:
            try:
                args_list = shlex.split(command_line)
                args = MIArgs.ParseParam(args_list[1:])
                convRunList.append((args, args.in_data_type))

            except Exception as e:
                print(f"Error parsing command line: {command_line}\n{e}")
    return convRunList


@dataclass
class MIArgs:
    # shape param
    forw: int
    batchsize: int
    in_channels: int
    in_h: int
    in_w: int
    in_d: int = 1
    out_channels: int = 0
    fil_h: int = 0
    fil_w: int = 0
    fil_d: int = 1
    pad_h: int = 0
    pad_w: int = 0
    pad_d: int = 0
    conv_stride_h: int = 1
    conv_stride_w: int = 1
    conv_stride_d: int = 0
    dilation_h: int = 1
    dilation_w: int = 1
    dilation_d: int = 0
    group_count: int = 1
    spatial_dim: int = 2
    
    # run param
    solution: int = -1
    time: int = 0
    verify: int = 1
    mode: str = 'conv'
    in_data: str = ''
    weights: str = ''
    dout_data: str = ''
    bias: int = 0
    in_bias: str = ''
    iter: int = 10
    pad_mode: str = 'default'
    fil_layout: str = ''
    in_layout: str = ''
    out_layout: str = ''
    gpubuffer_check: int = 0
    wall: int = 0
    printconv: int = 1
    pad_val: int = 0
    dump_output: int = 0
    search: int = 0
    verification_cache: str = ''
    gpualloc: int = 0  # GPU allocation mode, 0 for default, 1 for specific GPU

    # private fields
    trace: str = ''
    event: str = ''
    warmup: int = 3
    gpu: int = 0
    dbshape: int = 0
    cpu: int = 0 # 1 use CPU for verification
    save_db: int = 0

    in_data_type: MiopenDataType = MiopenDataType.miopenHalf
    shapeformat: str = 'vs'

    @staticmethod
    def ParseParam(args_list):
        command_name = args_list[0] if len(args_list) > 1 else "conv"
        if "bfp16" in command_name.lower():
            in_data_type = MiopenDataType.miopenBFloat16
        elif "fp16" in command_name.lower():
            in_data_type = MiopenDataType.miopenHalf
        elif "int8" in command_name.lower():
            in_data_type = MiopenDataType.miopenInt8
        elif "conv" == command_name.lower():
            in_data_type = MiopenDataType.miopenFloat
        elif "fp64" in command_name.lower():
            in_data_type = MiopenDataType.miopenDouble
        else:
            print(f"Unknown {command_name} in command name: {args_list}")

        args={}
        idx = 1

        # default value
        args['in_d']=1
        args['fil_d']=1
        args['pad_h']=0
        args['pad_w'] = 0
        args['pad_d'] = 0
        args['conv_stride_h'] = 1
        args['conv_stride_w'] = 1
        args['conv_stride_d'] = 0
        args['dilation_h'] = 1
        args['dilation_w'] = 1
        args['dilation_d'] = 0
        args['group_count'] = 1
        args['spatial_dim'] = 2
        args['solution'] = -1
        args['time'] = 0
        args['verify'] = 1
        args['mode'] = 'conv'
        args['in_data'] = ''
        args['weights'] = ''
        args['dout_data'] = ''
        args['bias'] = 0
        args['in_bias'] = ''
        args['iter'] = 10
        args['pad_mode'] = 'default'
        args['fil_layout'] = None
        args['in_layout'] = None
        args['out_layout'] = None
        args['gpubuffer_check'] = 0
        args['wall'] = 0
        args['printconv'] = 1
        args['pad_val'] = 0
        args['dump_output'] = 0
        args['search'] = 0
        args['verification_cache'] = ''
        args['gpualloc'] = 0  # GPU allocation mode, 0 for default, 1 for specific GPU
        args['shapeformat'] = 'solver'  # default shape format
        args['trace'] = ''
        args['event'] = ''
        args['gpu'] = 0
        args['dbshape'] = 0
        args['warmup'] = 3
        args['cpu'] = 0  # Use CPU for verification
        args['save_db'] = 0 # Should the metadata of the current shape be saved to the database
        
        while idx < len(args_list):
            if args_list[idx].startswith('-'):
                key = args_list[idx]
                value = args_list[idx + 1]

                norm_name = FLAG_MAPPING.get(key)
                convert = CONVERTERS.get(norm_name, str)
                args[norm_name] = convert(value)
                idx += 1

            idx += 1

        if args['in_layout'] == None:
            if args['spatial_dim'] == 2:
                args['in_layout'] = 'NCHW'
            else:
                args['in_layout'] = 'NCDHW'
        if args['fil_layout'] == None:
            args['fil_layout'] = args['in_layout']
        if args['out_layout'] == None:
            args['out_layout'] = args['in_layout']

        if 'forw' not in args:
            print(args_list)

        miargs = MIArgs(
            forw=args["forw"],
            batchsize=args["batchsize"],
            in_channels=args["in_channels"],
            in_h=args["in_h"],
            in_w=args["in_w"],
            in_d=args["in_d"],
            out_channels=args["out_channels"],
            fil_h=args["fil_h"],
            fil_w=args["fil_w"],
            fil_d=args["fil_d"],
            pad_h=args["pad_h"],
            pad_w=args["pad_w"],
            pad_d=args["pad_d"],
            conv_stride_h=args["conv_stride_h"],
            conv_stride_w=args["conv_stride_w"],
            conv_stride_d=args["conv_stride_d"],
            dilation_h=args["dilation_h"],
            dilation_w=args["dilation_w"],
            dilation_d=args["dilation_d"],
            group_count=args["group_count"],
            spatial_dim=args["spatial_dim"],
            solution=args["solution"],
            time=args["time"],
            verify=args["verify"],
            mode=args["mode"],
            in_data=args["in_data"],
            weights=args["weights"],
            dout_data=args["dout_data"],
            bias=args["bias"],
            in_bias=args["in_bias"],
            iter=args["iter"],
            pad_mode=args["pad_mode"],
            fil_layout=args["fil_layout"],
            in_layout=args["in_layout"],
            out_layout=args["out_layout"],
            gpubuffer_check=args["gpubuffer_check"],
            wall=args["wall"],
            printconv=args["printconv"],
            pad_val=args["pad_val"],
            dump_output=args["dump_output"],
            search=args["search"],
            verification_cache=args["verification_cache"],
            gpualloc=args["gpualloc"],

            # private fields
            shapeformat=args["shapeformat"],
            trace=args["trace"],
            event=args["event"],
            gpu=args["gpu"],
            dbshape=args["dbshape"],
            warmup=args["warmup"],
            cpu=args["cpu"],
            save_db=args["save_db"],
            in_data_type=in_data_type
        )

        return miargs