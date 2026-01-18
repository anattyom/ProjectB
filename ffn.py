import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np


# ... (SimpleFFN and generate_dummy_data classes remain the same) ...
class SimpleFFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleFFN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.output(x)


def generate_dummy_data(num_samples, input_dim, output_dim):
    print(f"Generating {num_samples} samples...")
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    return X, y


# ==========================================
# Latency Measurement Function
# ==========================================
def measure_performance(batch_size_b, chunk_size_c, input_dim=4096):
    if batch_size_b % chunk_size_c != 0:
        print("Error: Batch not divisible by chunks.")
        return

    micro_batch_size = int(batch_size_b / chunk_size_c)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Setup Model
    model = SimpleFFN(input_dim, 4096, 10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Data Setup (Warmup + Actual Runs)
    # We ignore the first few steps (Warmup) to let the GPU stabilize
    warmup_steps = 3
    active_steps = 10
    total_data = batch_size_b * (warmup_steps + active_steps)
    X_train, y_train = generate_dummy_data(total_data, input_dim, 10)

    model.train()

    # Store timing results
    step_times = []  # Total time per full optimization step

    print(f"\n--- Starting Measurement (B={batch_size_b}, C={chunk_size_c}) ---")

    for step, i in enumerate(range(0, len(X_train), batch_size_b)):
        if i + batch_size_b > len(X_train): break

        # Prepare Batch
        X_batch_total = X_train[i: i + batch_size_b]
        y_batch_total = y_train[i: i + batch_size_b]

        optimizer.zero_grad()

        # -------------------------------------------------
        # START TIMER (Synchronize CPU and GPU)
        # -------------------------------------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start_time = time.time()

        # --- Gradient Accumulation Loop ---
        for c in range(chunk_size_c):
            start = c * micro_batch_size
            end = start + micro_batch_size

            # Move Micro-batch to GPU (Include Data Transfer time in measurement)
            X_micro = X_batch_total[start:end].to(device)
            y_micro = y_batch_total[start:end].to(device)

            # Forward + Backward
            outputs = model(X_micro)
            loss = criterion(outputs, y_micro)
            (loss / chunk_size_c).backward()

        # Optimizer Step
        optimizer.step()

        # -------------------------------------------------
        # STOP TIMER
        # -------------------------------------------------
        if torch.cuda.is_available(): torch.cuda.synchronize()
        end_time = time.time()

        elapsed = end_time - start_time

        # Only record if after warmup
        if step >= warmup_steps:
            step_times.append(elapsed)
            print(f"Step {step + 1}: {elapsed * 1000:.2f} ms")
        else:
            print(f"Step {step + 1}: Warmup...")

    # ==========================================
    # Results Analysis
    # ==========================================
    avg_step_time = np.mean(step_times)
    throughput = batch_size_b / avg_step_time  # Samples / sec
    throughput_kb = throughput * input_dim * 4 / 1024  # KB/s (assuming float32)

    print(f"\n--- Final Results ---")
    print(f"Average Step Latency: {avg_step_time:.4f} sec")
    print(f"Throughput:           {throughput:.2f} samples/sec")
    print(f"Data Throughput:      {throughput_kb:.2f} KB/s")

    return avg_step_time, throughput


# Run
if __name__ == "__main__":
    # Test with the Optimal B/C from your simulator
    measure_performance(batch_size_b=1024, chunk_size_c=4)