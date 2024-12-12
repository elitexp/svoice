# Clear memory cached by MPS
import gc
import torch
torch.mps.empty_cache()

torch.mps.synchronize()  # Ensure all MPS operations are complete


gc.collect()  # Run garbage collector
