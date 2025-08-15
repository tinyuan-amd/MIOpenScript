import threading
print_lock = threading.Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs)

def get_tensor_storage_bytes(tensor):
    return tensor.untyped_storage().nbytes()

def print_forward_stats(args, input_tensor, weight_tensor, output_tensor, test_idx, elapsed_time_ms):
    # Determine spatial dimension
   
    num_dim = args.spatial_dim
    
    kernel_average_time = elapsed_time_ms / args.iter
    
    input_shape = input_tensor.size()
    weight_shape = weight_tensor.size()
    output_shape = output_tensor.size()
    
    group_count = args.group_count
    if num_dim == 2:
        in_n = input_shape[0]
        in_c = input_shape[1]
        in_h = input_shape[2]
        in_w = input_shape[3]
        
        wei_n = weight_shape[0]
        wei_c = weight_shape[1]
        wei_h = weight_shape[2]
        wei_w = weight_shape[3]
        
        out_n = output_shape[0]
        out_c = output_shape[1]
        out_h = output_shape[2]
        out_w = output_shape[3]
        
        flop_cnt = (2 * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w) / group_count
        input_bytes = in_n * in_c * in_h * in_w * input_tensor.element_size()
        weight_bytes = wei_n * wei_c * wei_h * wei_w * weight_tensor.element_size()
        read_bytes = input_bytes + weight_bytes
        
        output_bytes = 1.0 * out_n * out_c * out_h * out_w * output_tensor.element_size()
        
        with print_lock:
            print("stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs")
            print("stats: %s%dx%du%d, %d, %d, %d, %d, %d, %d, %d, %u, %u, %u, %.0f, %.0f, %f"
                %("fwd-conv", 
                    wei_h, 
                    wei_w, 
                    args.conv_stride_h,
                    in_n,
                    in_c,
                    out_h,
                    out_w,
                    wei_h,
                    wei_w,
                    out_c,
                    flop_cnt,
                    read_bytes,
                    output_bytes,
                    flop_cnt / kernel_average_time / 1e6,
                    (read_bytes + output_bytes) / kernel_average_time / 1e6,
                    kernel_average_time)
            )
    elif num_dim == 3:
        in_n = input_shape[0]
        in_c = input_shape[1]
        in_d = input_shape[2]
        in_h = input_shape[3]
        in_w = input_shape[4]
        
        wei_n = weight_shape[0]
        wei_c = weight_shape[1]
        wei_d = weight_shape[2]
        wei_h = weight_shape[3]
        wei_w = weight_shape[4]
        
        out_n = output_shape[0]
        out_c = output_shape[1]
        out_d = output_shape[2]
        out_h = output_shape[3]
        out_w = output_shape[4]
        
        flop_cnt = (2 * in_n * in_c * in_d * wei_h * wei_w * out_c * out_h * out_w) / group_count
        input_bytes = in_n * in_c * in_d * in_h * in_w * input_tensor.element_size()
        weight_bytes = wei_n * wei_c * wei_d * wei_h * wei_w * weight_tensor.element_size()
        read_bytes = input_bytes + weight_bytes
        
        output_bytes = 1.0 * out_n * out_c * out_d * out_h * out_w * output_tensor.element_size()
        with print_lock:
            print("stats: name, n, c, do, ho, wo, z, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs")
            print("stats: %s%dx%dx%du%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %u, %u, %u, "
                "%.0f, %.0f, %f"
                %("fwd-conv",
                wei_d,
                wei_h,
                wei_w,
                args.conv_stride_h,
                in_n,
                in_c,
                out_d,
                out_h,
                out_w,
                wei_d,
                wei_h,
                wei_w,
                out_c,
                flop_cnt,
                read_bytes,
                output_bytes,
                flop_cnt / kernel_average_time / 1e6,
                (read_bytes + output_bytes) / kernel_average_time / 1e6,
                kernel_average_time)
            )
        
def print_backward_data_stats(args, input_tensor, weight_tensor, output_tensor, test_idx, elapsed_time_ms):
    # Determine spatial dimension
    num_dim = args.spatial_dim
    
    kernel_average_time = elapsed_time_ms / args.iter
    
    input_shape = input_tensor.size()
    weight_shape = weight_tensor.size()
    output_shape = output_tensor.size()
    
    group_count = args.group_count
    if num_dim == 2:
        in_n = input_shape[0]
        in_c = input_shape[1]
        in_h = input_shape[2]
        in_w = input_shape[3]
        
        wei_n = weight_shape[0]
        wei_c = weight_shape[1]
        wei_h = weight_shape[2]
        wei_w = weight_shape[3]
        
        out_n = output_shape[0]
        out_c = output_shape[1]
        out_h = output_shape[2]
        out_w = output_shape[3]
        
        flop_cnt = (2 * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w) / group_count
        input_bytes = in_n * in_c * out_c * input_tensor.element_size()
        weight_bytes = wei_n * wei_c * wei_h * wei_w * weight_tensor.element_size()
        read_bytes = input_bytes + weight_bytes
        
        output_bytes = 1.0 * out_n * out_c * out_h * out_w * output_tensor.element_size()
        with print_lock:
            print("stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs")
            print("stats: %s%dx%du%d, %d, %d, %d, %d, %d, %d, %d, %u, %u, %u, %.0f, %.0f, %f"
                %("bwdd-conv", 
                    wei_h, 
                    wei_w, 
                    args.conv_stride_h,
                    in_n,
                    in_c,
                    out_h,
                    out_w,
                    wei_h,
                    wei_w,
                    out_c,
                    flop_cnt,
                    read_bytes,
                    output_bytes,
                    flop_cnt / kernel_average_time / 1e6,
                    (read_bytes + output_bytes) / kernel_average_time / 1e6,
                    kernel_average_time)
            )
    elif num_dim == 3:
        in_n = input_shape[0]
        in_c = input_shape[1]
        in_d = input_shape[2]
        in_h = input_shape[3]
        in_w = input_shape[4]
        
        wei_n = weight_shape[0]
        wei_c = weight_shape[1]
        wei_d = weight_shape[2]
        wei_h = weight_shape[3]
        wei_w = weight_shape[4]
        
        out_n = output_shape[0]
        out_c = output_shape[1]
        out_d = output_shape[2]
        out_h = output_shape[3]
        out_w = output_shape[4]
        
        flop_cnt = (2 * in_n * in_c * wei_d * wei_h * wei_w * out_c * out_h * out_w) / group_count
        input_bytes = in_n * in_c * out_c * input_tensor.element_size()
        weight_bytes = wei_n * wei_c * wei_d * wei_h * wei_w * weight_tensor.element_size()
        read_bytes = input_bytes + weight_bytes
        
        output_bytes = 1.0 * out_n * out_c * out_d * out_h * out_w * output_tensor.element_size()
        
        with print_lock:
            print("stats: name, n, c, do, ho, wo, z, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs")
            print("stats: %s%dx%dx%du%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %u, %u, %u, "
                "%.0f, %.0f, %f"
                %("bwdd-conv",
                wei_d,
                wei_h,
                wei_w,
                args.conv_stride_h,
                in_n,
                in_c,
                out_d,
                out_h,
                out_w,
                wei_d,
                wei_h,
                wei_w,
                out_c,
                flop_cnt,
                read_bytes,
                output_bytes,
                flop_cnt / kernel_average_time / 1e6,
                (read_bytes + output_bytes) / kernel_average_time / 1e6,
                kernel_average_time)
            )
            
def print_backward_weight_stats(args, input_tensor, weight_tensor, output_tensor, test_idx, elapsed_time_ms):
    # Determine spatial dimension
    num_dim = args.spatial_dim
    
    kernel_average_time = elapsed_time_ms / args.iter
    
    input_shape = input_tensor.size()
    weight_shape = weight_tensor.size()
    output_shape = output_tensor.size()
    
    group_count = args.group_count
    if num_dim == 2:
        in_n = input_shape[0]
        in_c = input_shape[1]
        in_h = input_shape[2]
        in_w = input_shape[3]
        
        wei_n = weight_shape[0]
        wei_c = weight_shape[1]
        wei_h = weight_shape[2]
        wei_w = weight_shape[3]
        
        out_n = output_shape[0]
        out_c = output_shape[1]
        out_h = output_shape[2]
        out_w = output_shape[3]
        
        flop_cnt = (2 * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w) / group_count
        read_bytes = 0
        
        output_bytes = 0
        with print_lock:
            print("stats: name, n, c, ho, wo, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs")
            print("stats: %s%dx%du%d, %d, %d, %d, %d, %d, %d, %d, %u, %u, %u, %.0f, %.0f, %f"
                %("bwdw-conv", 
                    wei_h, 
                    wei_w, 
                    args.conv_stride_h,
                    in_n,
                    in_c,
                    out_h,
                    out_w,
                    wei_h,
                    wei_w,
                    out_c,
                    flop_cnt,
                    read_bytes,
                    output_bytes,
                    flop_cnt / kernel_average_time / 1e6,
                    (read_bytes + output_bytes) / kernel_average_time / 1e6,
                    kernel_average_time)
            )
    elif num_dim == 3:
        in_n = input_shape[0]
        in_c = input_shape[1]
        in_d = input_shape[2]
        in_h = input_shape[3]
        in_w = input_shape[4]
        
        wei_n = weight_shape[0]
        wei_c = weight_shape[1]
        wei_d = weight_shape[2]
        wei_h = weight_shape[3]
        wei_w = weight_shape[4]
        
        out_n = output_shape[0]
        out_c = output_shape[1]
        out_d = output_shape[2]
        out_h = output_shape[3]
        out_w = output_shape[4]
        
        flop_cnt = (2 * in_n * in_c * wei_d * wei_h * wei_w * out_c * out_h * out_w) / group_count
        read_bytes = 0
        
        output_bytes = 0
        
        with print_lock:
            print("stats: name, n, c, do, ho, wo, z, y, x, k, flopCnt, bytesRead, bytesWritten, GFLOPs, GB/s, timeMs")
            print("stats: %s%dx%dx%du%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %u, %u, %u, "
                "%.0f, %.0f, %f"
                %("bwdw-conv",
                wei_d,
                wei_h,
                wei_w,
                args.conv_stride_h,
                in_n,
                in_c,
                out_d,
                out_h,
                out_w,
                wei_d,
                wei_h,
                wei_w,
                out_c,
                flop_cnt,
                read_bytes,
                output_bytes,
                flop_cnt / kernel_average_time / 1e6,
                (read_bytes + output_bytes) / kernel_average_time / 1e6,
                kernel_average_time)
            )
        
def print_stats(args, input_tensor, weight_tensor, grad_output_tensor, test_idx, elapsed_time_ms):
    """ Print the statistical information """
    if args.forw == 1:
        print_forward_stats(args, input_tensor, weight_tensor, grad_output_tensor, test_idx, elapsed_time_ms)
    elif args.forw == 2:
        print_backward_data_stats(args, input_tensor, weight_tensor, grad_output_tensor, test_idx, elapsed_time_ms)
    elif args.forw == 4:
        print_backward_weight_stats(args, input_tensor, weight_tensor, grad_output_tensor, test_idx, elapsed_time_ms)
    else:
        print("Error: the conv type is error !")
