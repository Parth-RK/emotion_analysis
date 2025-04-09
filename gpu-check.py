import tensorflow as tf
import cupy as cp
import torch
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Matrix size for the comparison
MATRIX_SIZE = 3000

# Function to print execution time summary
def print_summary(times):
    print("\n" + "="*50)
    print("MATRIX MULTIPLICATION PERFORMANCE COMPARISON")
    print("="*50)
    print(f"Matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"CuPy time:      {times['cupy']:.4f} seconds")
    print(f"PyTorch time:   {times['pytorch']:.4f} seconds")
    print(f"TensorFlow time: {times['tensorflow']:.4f} seconds")
    print("="*50)
    fastest = min(times.items(), key=lambda x: x[1])[0]
    print(f"Fastest framework: {fastest.upper()}")
    print("="*50)

print("\n" + "="*50)
print("PYTORCH GPU DETAILS")
print("="*50)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print("cuDNN version:", torch.backends.cudnn.version())
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")
    props = torch.cuda.get_device_properties(0)
    print(f"Memory: {props.total_memory / 1024**2:.2f} MB")
print("="*50)

print("\n" + "="*50)
print("CUPY GPU DETAILS")
print("="*50)
# Print GPU device name
device = cp.cuda.Device(0)
print(f"Using GPU: Device {device.id} - {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")
print("="*50)

print("\n" + "="*50)
print("TENSORFLOW GPU DETAILS")
print("="*50)
# Check if any GPUs are available
gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow version: {tf.__version__}")
if not gpus:
    print("No GPU found.")
else:
    print(f"Number of GPUs available: {len(gpus)}")
    for idx, gpu in enumerate(gpus):
        details = tf.config.experimental.get_device_details(gpu)
        print(f"\nGPU #{idx}:")
        print(f"  Name: {details.get('device_name', 'Unknown')}")
        print(f"  Compute Capability: {details.get('compute_capability', 'Unknown')}")
        print(f"  Memory Limit: {details.get('memory_limit', 'Unknown') / 1024**3:.2f} GB")
print("="*50)

# Dictionary to store execution times
execution_times = {}

print("\n" + "="*50)
print("CUPY MATRIX MULTIPLICATION")
print("="*50)
# Start measuring time
start_time = time.time()

# Create two random arrays on the GPU
a_cp = cp.random.rand(MATRIX_SIZE, MATRIX_SIZE)
b_cp = cp.random.rand(MATRIX_SIZE, MATRIX_SIZE)

# Perform a matrix multiplication on the GPU
c_cp = cp.matmul(a_cp, b_cp)

# Synchronize to make sure computation is done before stopping the timer
cp.cuda.Device(0).synchronize()

# Stop measuring time
end_time = time.time()
cupy_time = end_time - start_time
execution_times['cupy'] = cupy_time

# Print time taken
print(f"CuPy matrix multiplication took {cupy_time:.4f} seconds.")

# Show memory usage
free, total = cp.cuda.runtime.memGetInfo()
free_mb = free / (1024**2)
total_mb = total / (1024**2)
used_mb = total_mb - free_mb
print(f"GPU Memory Usage: {used_mb:.2f} MB used / {total_mb:.2f} MB total")
print("="*50)

print("\n" + "="*50)
print("PYTORCH MATRIX MULTIPLICATION")
print("="*50)
if torch.cuda.is_available():
    # Create two random arrays on the GPU
    a_torch = torch.rand(MATRIX_SIZE, MATRIX_SIZE, device='cuda')
    b_torch = torch.rand(MATRIX_SIZE, MATRIX_SIZE, device='cuda')
    
    # Start measuring time
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Perform a matrix multiplication on the GPU
    c_torch = torch.matmul(a_torch, b_torch)
    
    # Synchronize to make sure computation is done before stopping the timer
    torch.cuda.synchronize()
    
    # Stop measuring time
    end_time = time.time()
    pytorch_time = end_time - start_time
    execution_times['pytorch'] = pytorch_time
    
    # Print time taken
    print(f"PyTorch matrix multiplication took {pytorch_time:.4f} seconds.")
    
    # Show memory usage
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
else:
    print("PyTorch CUDA is not available.")
    execution_times['pytorch'] = float('inf')
print("="*50)

print("\n" + "="*50)
print("TENSORFLOW MATRIX MULTIPLICATION")
print("="*50)
if gpus:
    # Create two random arrays on the GPU
    a_tf = tf.random.uniform((MATRIX_SIZE, MATRIX_SIZE), dtype=tf.float32)
    b_tf = tf.random.uniform((MATRIX_SIZE, MATRIX_SIZE), dtype=tf.float32)
    
    # Ensure tensors are created before timing
    tf.keras.backend.clear_session()
    _ = a_tf.numpy()  # Force execution
    _ = b_tf.numpy()  # Force execution
    
    # Start measuring time
    start_time = time.time()
    
    # Perform a matrix multiplication on the GPU
    c_tf = tf.matmul(a_tf, b_tf)
    
    # Force execution of the operation
    _ = c_tf.numpy()
    
    # Stop measuring time
    end_time = time.time()
    tensorflow_time = end_time - start_time
    execution_times['tensorflow'] = tensorflow_time
    
    # Print time taken
    print(f"TensorFlow matrix multiplication took {tensorflow_time:.4f} seconds.")
    
    # Show memory usage (TensorFlow doesn't provide direct memory usage API like PyTorch)
    print("Note: TensorFlow doesn't provide direct GPU memory usage API")
else:
    print("TensorFlow GPU is not available.")
    execution_times['tensorflow'] = float('inf')
print("="*50)

# Print summary comparison
print_summary(execution_times)