import tensorflow as tf
import cupy as cp
# import torch
import time

# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# print(f"Device: {torch.cuda.get_device_name}")
# print(f"Memory: {torch.cuda.get_device_properties.total_memory / 1024**2:.2f} MB")

# Print GPU device name
device = cp.cuda.Device(0)
print(f"Using GPU: Device {device.id} - {cp.cuda.runtime.getDeviceProperties(device.id)['name'].decode()}")

# Start measuring time
start_time = time.time()

# Create two random arrays on the GPU
a = cp.random.rand(3000, 3000)
b = cp.random.rand(3000, 3000)

# Perform a matrix multiplication on the GPU
c = cp.matmul(a, b)

# Synchronize to make sure computation is done before stopping the timer
cp.cuda.Device(0).synchronize()

# Stop measuring time
end_time = time.time()

# Print time taken
print(f"Matrix multiplication took {end_time - start_time:.4f} seconds.")

# Show memory usage
free, total = cp.cuda.runtime.memGetInfo()
free_mb = free / (1024**2)
total_mb = total / (1024**2)
used_mb = total_mb - free_mb

print(f"GPU Memory Usage: {used_mb:.2f} MB used / {total_mb:.2f} MB total")



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