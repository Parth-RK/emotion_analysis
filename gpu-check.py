import cupy as cp
import time

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
