from html import parser
import torch
import time
import torch.nn.functional as F
import argparse
import sys
import os
import numpy as np
import concurrent.futures
import itertools
from torch.profiler import profile, record_function, ProfilerActivity
from enum import Enum
import miopUtil.shapeConvert as shapeConvert
from miopUtil.MIArgs import *
import MIOpenDriver_Ref
import miopUtil.DataHash as DataHash
import miopUtil.PrintStat as PrintStat
import threading

database_lock = threading.Lock()
class ConvolutionRunner:
    def __init__(self, device, args, in_data_type, golden_database, gpu_idx, test_idx=0):
        self.device = device
        self.args = args
        self.in_data_type = in_data_type
        self.golden_database = golden_database
        self.gpu_idx = gpu_idx
        self.test_idx = test_idx
        
        self._setup_device()
        self._setup_data_type()
        self._setup_dimensions()
        self._setup_tensors()
        
        self.creation_time = 0
        self.execution_time = 0
        
    def _setup_device(self):
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            self.stream = torch.cuda.Stream(device=self.device)
        else:
            self.stream = None
            
        if self.args.solution >= 0:
            os.environ["MIOPEN_DEBUG_FIND_ONLY_SOLVER"] = str(self.args.solution)
    
    def _setup_data_type(self):
        dtype_map = {
            MiopenDataType.miopenHalf: (torch.float16, "fp16"),
            MiopenDataType.miopenBFloat16: (torch.bfloat16, "bfp16"),
            MiopenDataType.miopenInt8: (torch.int8, "int8"),
        }
        self.dtype, self.data_type_str = dtype_map.get(self.in_data_type, (torch.float32, ""))
    
    def _setup_dimensions(self):
        self.is_3d = self.args.spatial_dim == 3
        self.conv_fn = F.conv3d if self.is_3d else F.conv2d
        self.tensor_dim = 5 if self.is_3d else 4
        
        self._calculate_output_dimensions()
        
    def _calculate_output_dimensions(self):
        def calc_output_size(in_size, pad, kernel, stride, dilation=1):
            return (in_size + 2*pad - dilation*(kernel - 1) - 1) // stride + 1
        
        if self.is_3d:
            self.out_depth = calc_output_size(
                self.args.in_d, self.args.pad_d, self.args.fil_d, 
                self.args.conv_stride_d, self.args.dilation_d
            )
        else:
            self.out_depth = 1
            
        self.out_height = calc_output_size(
            self.args.in_h, self.args.pad_h, self.args.fil_h,
            self.args.conv_stride_h, self.args.dilation_h
        )
        self.out_width = calc_output_size(
            self.args.in_w, self.args.pad_w, self.args.fil_w,
            self.args.conv_stride_w, self.args.dilation_w
        )
    
    def _setup_tensors(self):
        start_time = time.time()
        
        self._calculate_tensor_shapes()
        
        self.input_tensor = self._create_tensor(self.input_shape, self.args.in_data)
        self.weight = self._create_tensor(self.weight_shape, self.args.weights)
        self.grad_output = self._create_tensor(self.grad_output_shape, self.args.dout_data)
        
        self.bias = None
        if self.args.bias:
            bias_shape = (self.args.out_channels,)
            self.bias = self._create_tensor(bias_shape, self.args.in_bias)
        
        self.creation_time = (time.time() - start_time) * 1000
        # print(f"Test {self.test_idx}: Tensor creation took {self.creation_time:.2f} ms")
    
    def _calculate_tensor_shapes(self):
        if self.is_3d:
            self.input_shape = (self.args.batchsize, self.args.in_channels, 
                              self.args.in_d, self.args.in_h, self.args.in_w)
        else:
            self.input_shape = (self.args.batchsize, self.args.in_channels, 
                              self.args.in_h, self.args.in_w)
        
        in_channels_per_group = self.args.in_channels // self.args.group_count
        if self.is_3d:
            self.weight_shape = (self.args.out_channels, in_channels_per_group,
                               self.args.fil_d, self.args.fil_h, self.args.fil_w)
        else:
            self.weight_shape = (self.args.out_channels, in_channels_per_group,
                               self.args.fil_h, self.args.fil_w)
        
        if self.is_3d:
            self.grad_output_shape = (self.args.batchsize, self.args.out_channels,
                                    self.out_depth, self.out_height, self.out_width)
        else:
            self.grad_output_shape = (self.args.batchsize, self.args.out_channels,
                                    self.out_height, self.out_width)
    
    def _create_tensor(self, shape, filename=""):
        if filename and os.path.exists(filename):
            data = np.fromfile(filename, dtype=np.float32)
            return torch.tensor(data.reshape(shape), dtype=self.dtype, 
                              device=self.device, requires_grad=True)
        else:
            return self._generate_fixed_seed_input(shape)
    
    def _generate_fixed_seed_input(self, shape):
        if self.args.verify:
            torch.manual_seed(12345678)
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                input_data = torch.randn(shape, dtype=self.dtype, device=self.device, requires_grad=True)
                # input_data = input_data.to(self.device, non_blocking=True)
                # self.stream.synchronize()
        else:
            input_data = torch.randn(shape, dtype=self.dtype, device=self.device, requires_grad=True)
        return input_data
    
    def _get_conv_args(self):
        return {
            'stride': (self.args.conv_stride_d, self.args.conv_stride_h, self.args.conv_stride_w) if self.is_3d else 
                     (self.args.conv_stride_h, self.args.conv_stride_w),
            'padding': (self.args.pad_d, self.args.pad_h, self.args.pad_w) if self.is_3d else 
                      (self.args.pad_h, self.args.pad_w),
            'dilation': (self.args.dilation_d, self.args.dilation_h, self.args.dilation_w) if self.is_3d else 
                       (self.args.dilation_h, self.args.dilation_w),
            'groups': self.args.group_count
        }
    
    def run_convolution(self, operation):
        with record_function(f"convolution_{operation}"):
            conv_args = self._get_conv_args()

            if operation == 1:  # Forward convolution
                result = self.conv_fn(self.input_tensor, self.weight, self.bias, **conv_args)
                return result

            elif operation == 2:  # Backward data
                output_mask = [True, False, False]
                grad_input, _, _ = torch.ops.aten.convolution_backward(
                    grad_output=self.grad_output,
                    input=self.input_tensor,
                    weight=self.weight,
                    bias_sizes=[0],
                    stride=conv_args['stride'],
                    padding=conv_args['padding'],
                    dilation=conv_args['dilation'],
                    transposed=False,
                    output_padding=(0, 0, 0) if self.is_3d else (0, 0),
                    groups=conv_args['groups'],
                    output_mask=output_mask
                )
                return grad_input

            elif operation == 4:  # Backward weight
                output_mask = [False, True, False]
                _, grad_weight, _ = torch.ops.aten.convolution_backward(
                    grad_output=self.grad_output,
                    input=self.input_tensor,
                    weight=self.weight,
                    bias_sizes=[0],
                    stride=conv_args['stride'],
                    padding=conv_args['padding'],
                    dilation=conv_args['dilation'],
                    transposed=False,
                    output_padding=(0, 0, 0) if self.is_3d else (0, 0),
                    groups=conv_args['groups'],
                    output_mask=output_mask
                )
                return grad_weight
    
    def run_convolution_ref(self, operation):
        with record_function(f"convolution_ref_{operation}"):
            conv_args = self._get_conv_args()
            
            reference = MIOpenDriver_Ref.gpu_convolution_reference(
                grad_output=self.grad_output,
                weight=self.weight,
                input=self.input_tensor,
                padding=conv_args['padding'][0],
                stride=conv_args['stride'][0],
                dilation=conv_args['dilation'][0],
                group=conv_args['groups'],
                solution_id=86,
                operation=operation,
                type=self.data_type_str
            )
            
            if reference is None:
                print("Error: Reference result is None")
            
            return reference
    
    def warmup(self):
        if self.args.warmup > 0:
            if self.device.type == 'cuda':
                with torch.cuda.stream(self.stream):
                    for _ in range(self.args.warmup):
                        self.run_convolution(self.args.forw)
    
    def run_with_profiling(self):
        trace_name = f"conv{self.data_type_str}-F{self.args.forw}-n{self.args.batchsize}-c{self.args.in_channels}"
        
        if self.args.trace or self.args.event:
            return self._run_with_trace(trace_name)
        else:
            return self._run_with_timing()
    
    def _run_with_trace(self, trace_name):
        prefix = self.args.trace.split(".")[0] if self.args.trace else self.args.event.split(".")[0]
        trace_path = f"{prefix}_{trace_name}.json"
        trace_path = os.path.abspath(trace_path)
        
        trace_dir = os.path.dirname(trace_path)
        if trace_dir and not os.path.exists(trace_dir):
            os.makedirs(trace_dir)
        
        print(f"\nStarting PyTorch trace capture (saving to {trace_path})")
        
        with profile(
            activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=False,
            with_flops=True
        ) as prof:
            for _ in range(self.args.iter):
                result = self.run_convolution(self.args.forw)
            
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
        
        if self.args.trace:
            prof.export_chrome_trace(trace_path)
            print(f"PyTorch trace saved to {trace_path}")
        
        if self.args.event:
            event_path = f"{prefix}_{trace_name}_event.json"
            events = prof.key_averages(group_by_input_shape=False).table(
                sort_by="self_cuda_time_total", row_limit=120
            )
            with open(event_path, "w") as file:
                file.write(events)
            print(events)
            print(f"Events have been written to {event_path}")
        
        return result
    
    def _run_with_timing(self):
        start_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.args.iter)]
        end_event = [torch.cuda.Event(enable_timing=True) for _ in range(self.args.iter)]
        
        result = None
        
        if self.device.type == 'cuda':
            with torch.cuda.stream(self.stream):
                start_event[0].record()
                for _ in range(self.args.iter):
                    result = self.run_convolution(self.args.forw)
                end_event[0].record()
                self.stream.synchronize()
            self.execution_time = start_event[0].elapsed_time(end_event[0]) / self.args.iter
        else:
            start_time = time.time()
            self.args.iter = 1
            for _ in range(self.args.iter):
                result = self.run_convolution(self.args.forw)
            end_time = time.time()
            self.execution_time = ((end_time - start_time) * 1000) / self.args.iter
        
        return result
    
    def verify_result(self, result):
        if not self.args.verify:
            return
        
        shape_key = DataHash.generate_shape_key(self.args)
        
        self.args.save_db = 1
        exist, golden_stats = DataHash.load_golden_stats_from_memory(
            shape_key, self.golden_database, need_lock=self.args.save_db
        )
        
        if not exist:
            golden_result = None
            if self.device.type == 'cuda':
                with torch.cuda.stream(self.stream):
                    golden_result = self.run_convolution_ref(self.args.forw)
                    self.stream.synchronize()
            else:
                golden_result = self.run_convolution(self.args.forw)
            
            golden_stats = DataHash.summarize_conv_output(golden_result, include_histogram=False, bins=6)
            
            if self.args.save_db:
                DataHash.save_golden_stats_to_memory(golden_stats, shape_key, self.golden_database)
        
        stats = DataHash.summarize_conv_output(result, include_histogram=False, bins=6)
        
        tolerance = 0.05
        if golden_stats:
            return DataHash.compare_stats(golden_stats, stats, tolerance=tolerance)
    
    def run(self):
        if self.args.dbshape:
            self._run_dbshape_test()
        
        self.warmup()
        
        if self.args.search:
            torch.backends.cudnn.benchmark = True
        
        result = self.run_with_profiling()
        
        return self.verify_result(result)
    
    def _run_dbshape_test(self):
        import miopUtil.shapeConvert as shapeConvert
        
        problem = shapeConvert.ProblemDescription(
            in_channels=self.args.in_channels if self.args.forw == 1 else self.args.out_channels,
            spatial_dims=self.args.spatial_dim,
            in_depth=self.args.in_d,
            in_height=self.args.in_h,
            in_width=self.args.in_w,
            weights_depth=self.args.fil_d,
            weights_height=self.args.fil_h,
            weights_width=self.args.fil_w,
            out_channels=self.args.out_channels if self.args.forw == 1 else self.args.in_channels,
            out_depth=0,
            out_height=0,
            out_width=0,
            in_batch_size=self.args.batchsize,
            pad_d=self.args.pad_d,
            pad_h=self.args.pad_h,
            pad_w=self.args.pad_w,
            kernel_stride_d=self.args.conv_stride_d,
            kernel_stride_h=self.args.conv_stride_h,
            kernel_stride_w=self.args.conv_stride_w,
            dilation_d=self.args.dilation_d,
            dilation_h=self.args.dilation_h,
            dilation_w=self.args.dilation_w,
            bias=int(self.args.bias),
            in_layout='NCHW' if self.args.spatial_dim == 2 else 'NCDHW',
            weights_layout='NCHW' if self.args.spatial_dim == 2 else 'NCDHW',
            out_layout='NCHW' if self.args.spatial_dim == 2 else 'NCDHW',
            in_data_type=self.in_data_type,
            weights_data_type=self.in_data_type,
            out_data_type=self.in_data_type,
            direction_str=shapeConvert.get_direction_str(self.args.forw),
            group_count=self.args.group_count
        )
        problem.InitDef()
        problem.test()
    
    def cleanup(self):
        if self.stream:
            self.stream.synchronize()
        
        del self.input_tensor
        del self.weight
        del self.grad_output
        if self.bias is not None:
            del self.bias
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

@dataclass
class ExecutionConfig:
    num_gpus: int
    max_workers: int
    golden_database_file: str = "conv_golden_stats.json"
    enable_profiling: bool = False
    batch_size: int = 10
    timeout: Optional[float] = None

class ConvolutionManager:
    # private var
    is_validate_pass = False
    max_error = 0.0
    tolerance = 0.05
    
    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.devices = self._setup_devices()
        self.golden_database = self._load_golden_database()
        self.runners = []
        self.execution_stats = {
            'total_tests': 0,
            'successful_tests': 0,
            'failed_tests': 0,
            'total_time': 0,
            'gpu_time': 0
        }
    
    def _setup_devices(self) -> list[torch.device]:
        if torch.cuda.is_available():
            num_gpus = min(torch.cuda.device_count(), self.config.num_gpus)
            devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
            print(f"Using {num_gpus} CUDA devices")
        else:
            devices = [torch.device('cpu') for _ in range(self.config.num_gpus)]
            print(f"Using {self.config.num_gpus} CPU devices (CUDA not available)")
        
        return devices
    
    def _load_golden_database(self) -> dict:
        try:
            if not os.path.exists(self.config.golden_database_file):
                print(f"Creating new golden database: {self.config.golden_database_file}")
                # DataHash.create_golden_stats(self.config.golden_database_file)
                return {}
            
            database = DataHash.load_golden_stats_from_file(self.config.golden_database_file)
            print(f"Loaded golden database with {len(database)} entries")
            return database
            
        except Exception as e:
            print(f"Error loading golden database: {e}")
            return {}
    
    def _save_golden_database(self):
        try:
            DataHash.save_golden_stats_to_file(self.golden_database, self.config.golden_database_file)
            print(f"Saved golden database with {len(self.golden_database)} entries")
        except Exception as e:
            print(f"Error saving golden database: {e}")
    
    def run_single_test(self, device: torch.device, args, in_data_type, gpu_idx: int, test_idx: int) -> Optional[ConvolutionRunner]:
        runner = None
        try:
            runner = ConvolutionRunner(device, args, in_data_type, self.golden_database, gpu_idx, test_idx)
            is_validate_pass, max_error, channel_error = runner.run()
            
            self.is_validate_pass = is_validate_pass
            self.max_error = max_error
            
            with database_lock:
                self.execution_stats['successful_tests'] += 1
                self.execution_stats['gpu_time'] += (runner.execution_time)
            
            return runner
            
        except Exception as e:
            print(f"Error in test {test_idx} on GPU {gpu_idx}: {e}")
            with database_lock:
                self.execution_stats['failed_tests'] += 1
            if runner:
                try:
                    runner.cleanup()
                except:
                    pass
            return None
    
    def run_batch_tests(self, test_list: list[tuple]) -> list[Optional[ConvolutionRunner]]:
        results = []
        gpu_cycle = itertools.cycle(range(len(self.devices)))
        
        print(f"Running {len(test_list)} tests in batch mode")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for test_idx, (args, in_data_type) in enumerate(test_list, 1):
                gpu_idx = next(gpu_cycle)
                device = self.devices[gpu_idx]
                
                future = executor.submit(
                    self.run_single_test, 
                    device, args, in_data_type, gpu_idx, test_idx
                )
                futures.append((future, test_idx, gpu_idx))
            
            for future, test_idx, gpu_idx in futures:
                try:
                    if self.config.timeout:
                        runner = future.result(timeout=self.config.timeout)
                    else:
                        runner = future.result()
                    
                    results.append(runner)
                    
                    if runner:
                        print(f"✓ Test {test_idx} successfully on GPU {gpu_idx}, Verify {self.is_validate_pass}: ({self.max_error} < {self.tolerance}), execution time: {runner.execution_time:.4f} ms")
                    else:
                        print(f"✗ Test {test_idx} failed on GPU {gpu_idx}")
                        
                except concurrent.futures.TimeoutError:
                    print(f"✗ Test {test_idx} timed out on GPU {gpu_idx}")
                    results.append(None)
                except Exception as e:
                    print(f"✗ Test {test_idx} failed with exception: {e}")
                    results.append(None)
        
        return results
    
    def run_sequential_tests(self, test_list: list[tuple]) -> list[Optional[ConvolutionRunner]]:
        results = []
        print(f"Running {len(test_list)} tests in sequential mode")
        
        for test_idx, (args, in_data_type) in enumerate(test_list, 1):
            gpu_idx = args.gpu if hasattr(args, 'gpu') else 0
            device = self.devices[min(gpu_idx, len(self.devices) - 1)]
            
            runner = self.run_single_test(device, args, in_data_type, gpu_idx, test_idx)
            results.append(runner)
            
            if runner:
                print(f"✓ Test {test_idx} completed successfully, execution time: {runner.execution_time/self.args.iter:.4f} ms")
            else:
                print(f"✗ Test {test_idx} failed")
        
        return results
    
    def cleanup_runners(self, runners: list[Optional[ConvolutionRunner]]):
        for runner in runners:
            if runner:
                try:
                    runner.cleanup()
                except Exception as e:
                    print(f"Warning: Error during cleanup: {e}")
    
    def print_summary(self):
        stats = self.execution_stats
        total_tests = stats['successful_tests'] + stats['failed_tests']
        
        print("\n" + "="*50)
        print("EXECUTION SUMMARY")
        print("="*50)
        print(f"Total tests: {total_tests}")
        print(f"Successful: {stats['successful_tests']}")
        print(f"Failed: {stats['failed_tests']}")
        print(f"Success rate: {stats['successful_tests']/max(total_tests, 1)*100:.1f}%")
        print(f"Total CPU time: {stats['total_time']:.2f} ms")
        print(f"Total GPU time: {stats['gpu_time']:.2f} ms")
        
        if stats['successful_tests'] > 0:
            avg_gpu_time = stats['gpu_time'] / stats['successful_tests']
            print(f"Average GPU time per test: {avg_gpu_time:.2f} ms")
        
        print("="*50)

def Solve():
    start_time = time.time()
    
    file_input = "--input" in sys.argv[1]
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
    else:
        num_gpus = 20
    
    config = ExecutionConfig(
        num_gpus=num_gpus,
        max_workers=min(num_gpus, 8),
        enable_profiling=any('--trace' in arg or '--event' in arg for arg in sys.argv),
        timeout=300.0
    )
    
    manager = ConvolutionManager(config)
    runners = []
    
    try:
        if file_input:
            conv_run_list = ParseRunList(sys.argv[2])
            print(f"Parsed {len(conv_run_list)} commands from input file")
            
            manager.execution_stats['total_tests'] = len(conv_run_list)
            
            runners = manager.run_batch_tests(conv_run_list)
            
        else:
            args = MIArgs.ParseParam(sys.argv[1:])
            
            if hasattr(args, 'cpu') and args.cpu == 1:
                device = torch.device('cpu')
                gpu_idx = -1
            else:
                gpu_idx = getattr(args, 'gpu', 0)
                device = manager.devices[min(gpu_idx, len(manager.devices) - 1)]
            
            manager.execution_stats['total_tests'] = 1
            
            runner = manager.run_single_test(
                device, args, args.in_data_type, gpu_idx, 1
            )
            runners = [runner] if runner else []
    
    finally:
        end_time = time.time()
        manager.execution_stats['total_time'] = (end_time - start_time) * 1000
        
        manager._save_golden_database()
        manager.cleanup_runners(runners)
        manager.print_summary()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    try:
        Solve()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()