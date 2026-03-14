# 6. Training Infrastructure & Optimization

Training self-play Reinforcement Learning agents requires generating massive amounts of game data. In the early stages of the project, training was strictly sequential—the agent played one game at a time, collecting experiences into a buffer before running a PPO update. This single-process approach was agonizingly slow; simulating 5,000 games per epoch took entirely too long to allow for meaningful iteration and debugging.

To solve this, I evolved the training pipeline from a naive single-process script into a highly optimized, GPU-accelerated parallel architecture.

### 6.1 Profiling the Bottleneck

My first attempt at parallelization involved using Python's `multiprocessing.Pool` to spawn 16 CPU workers. Each worker loaded its own copy of the model onto the CPU and played games independently. While this was a major improvement, it left my primary compute unit (an NVIDIA RTX 5090) completely idle during the data-collection phase.

Before blindly throwing the model onto the GPU, I wrote a benchmark script to measure where the execution time was actually going. The profiling revealed two critical insights:

1. **Model inference dominated execution time**: Over 93% of the per-step time was spent running the neural network. The pure Python game logic (`env.step()`) took only ~0.12 ms per call, whereas a single-batch CPU inference took 1.68 ms.
2. **GPU scaling is non-linear**: A single-batch GPU inference (1.01 ms) was barely faster than CPU due to kernel launch overheads. However, when batched, the GPU time remained essentially flat. Processing a batch of 16 states took 0.96 ms on the GPU, compared to 7.30 ms on the CPU—a massive 29x speedup per sample.

This proved that a decentralized approach (each worker having its own GPU model) would be terribly inefficient because batch sizes would be 1. The solution was a centralized architecture that batched requests from many workers.

### 6.2 The GPU Inference Server Architecture

I engineered a custom parallel infrastructure featuring a **Centralized GPU Inference Server** and a persistent pool of **CPU Workers**.

- **Inference Server (Thread)**: A dedicated `BatchedGPUServer` runs as a `threading.Thread` within the Main Process. It has direct access to the GPU and loops continuously, draining an incoming request queue. It uses adaptive batching—pulling pending requests with a tiny timeout (~3ms) to naturally scale the batch size to the concurrency of the workers. Once a batch is processed, it routes the resulting actions and values back to the specific workers via response queues.
- **CPU Workers (Processes)**: Dozens of workers run as standard `multiprocessing.Process` instances. They handle the lightweight `CheckersEnv` game logic on the CPU. When it is the agent's turn to act, the worker submits the board state to the server and blocks until the server returns the action. 
- **Opponent Inference**: Crucially, opponent pool models are cached locally on each worker's CPU. Sending opponent states to the GPU server would add significant routing complexity, and since pool opponents are only active in a fraction of games, the CPU handles them efficiently.

### 6.3 Shared Memory IPC (Zero-Copy)

Initially, the CPU workers sent the 4-channel 8×8 board states and action masks to the server via a `multiprocessing.Queue`. However, Python's pickling overhead for serializing hundreds of thousands of NumPy arrays per epoch created a massive IPC bottleneck (transferring ~300MB of pickled data per epoch).

To eliminate this, I implemented a **zero-copy shared memory protocol** using Python's `multiprocessing.shared_memory`. 
When the worker pool is initialized, a massive flat memory buffer is pre-allocated. Both the Main Process (the server) and the child processes (the workers) map this buffer directly into NumPy arrays (`_state_buf` and `_mask_buf`). 

Instead of pickling the state tensor over the queue, a worker simply writes its state into its designated slot in the shared array and sends only its integer `worker_id` over the queue. The GPU server reads directly from the shared memory block to build the PyTorch batch.

### 6.4 Worker Tail Mitigation

During league training, games vary wildly in length (from rapid 40-move Aggressive blowouts to 150-move Terminal stalemates). In an early design, I used static round-robin task assignment. This created a "straggler" problem—workers assigned randomly longer games would still be running while the rest of the CPUs sat completely idle, waiting to start the PPO update.

I fixed this by implementing a **dynamic task queue**. The workers pull game configurations from a shared queue one at a time. Fast workers automatically grab more tasks, keeping all CPUs continuously busy right up until the final game of the epoch completes.

### 6.5 PPO Update Optimizations

Once the workers finish collecting the 5,000 games, the data is aggregated in the Main Process for the PPO gradient update. Processing ~400,000 transitions (which doubles to 800,000 with DrAC data augmentation) requires significant throughput. I applied two key optimizations to the update loop:

1. **TorchScript GAE**: The Generalized Advantage Estimation (GAE) calculation is an inherently sequential backward scan over the trajectory. Running this loop in standard Python was slow (~3-5 seconds per epoch). By wrapping the GAE function in `@torch.jit.script`, PyTorch compiles the loop down to native C++, eliminating the Python interpreter overhead entirely.
2. **Mixed Precision (BF16)**: All forward passes, backward passes, and DrAC augmentations during the update phase utilize `torch.amp.autocast` targeting `bfloat16`. BF16 was specifically chosen over standard FP16 because it shares the same exponent range as FP32, preventing gradient overflow issues when dealing with the extreme scale of the terminal rewards (±100), while still achieving a ~2x throughput increase via the RTX 5090's tensor cores.

The final architecture reduced epoch wall-clock times from ~12 minutes down to just 4.5 minutes (a 2.65x speedup over standard CPU multiprocessing), saving roughly 5 days of absolute compute time over the course of a 1,000-epoch training run.