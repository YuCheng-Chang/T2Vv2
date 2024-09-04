import torch

def check_gpu():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9  # Convert to GB
            print(f"GPU {i}: {gpu_name}, Total Memory: {gpu_memory:.2f} GB")
    else:
        print("No GPU available. Using CPU.")

check_gpu()