---
tags: [basic, nlp, gpu, serverless]
dataset: [Tiny Shakespeare]
framework: [torch]
---

# Serverless Federated Learning with NanoGPT and Flower

This example demonstrates **serverless** federated learning for character-level
language modeling using [NanoGPT](https://github.com/karpathy/nanoGPT) and
Flower. Unlike the standard Flower client/server pattern, this approach uses
`AsyncFederatedNode` from
[flwr_serverless](https://github.com/kungfuai/flwr_serverless) — there is no
central server orchestrating training. Nodes coordinate through a shared folder
(in-memory for simulation, or a filesystem path for multi-process/multi-machine
setups).

The model is a "baby GPT" (~10.8M parameters) — small enough to train on CPU.

## How It Works

In serverless federated learning, each node independently:

1. Trains on its local data partition
2. Writes its model weights to a shared folder
3. Reads other nodes' weights from the shared folder
4. Aggregates using FedAvg (via Flower's `FedAvg` strategy)

No server process is needed. Nodes can join and leave at any time.

## Project Layout

```shell
nanogpt-serverless
├── nanogpt_serverless
│   ├── __init__.py
│   ├── main.py              # Serverless federated training loop
│   ├── task.py              # NanoGPT model, data loading, train/test
│   └── serverless/          # Vendored flwr_serverless modules
│       ├── __init__.py
│       ├── async_federated_node.py
│       ├── aggregatable.py
│       ├── base_folder.py
│       ├── local_folder.py
│       └── in_memory_folder.py
├── pyproject.toml
└── README.md
```

## Install Dependencies

```bash
cd examples/nanogpt-serverless
pip install -e .
```

## Run the Example

### Single-Process Simulation (InMemoryFolder)

```bash
python -m nanogpt_serverless.main
```

This simulates 2 federated nodes in a single process using `InMemoryFolder`.

### Multi-Process / Multi-Machine (LocalFolder)

Use `--shared-folder` to point at a directory accessible by all participants
(e.g., NFS mount, shared disk):

```bash
# Terminal 1
python -m nanogpt_serverless.main --shared-folder /tmp/nanogpt_shared

# Terminal 2 (on same or different machine)
python -m nanogpt_serverless.main --shared-folder /tmp/nanogpt_shared
```

## Expected Results

Over 5 rounds (2 simulated nodes, 50 steps each, CPU):

| Round | Train Loss | Val Loss | Perplexity |
| ----- | ---------- | -------- | ---------- |
| 0     | —          | 4.25     | 70.0       |
| 1     | 2.90       | 2.59     | 13.3       |
| 2     | 2.56       | 2.72     | 15.1       |
| 3     | 2.53       | 2.54     | 12.7       |
| 4     | 2.48       | 2.52     | 12.4       |
| 5     | 2.45       | 2.47     | 11.8       |

After training, the model generates a Shakespeare-style text sample.

## Configuration

All parameters are configurable via CLI flags:

| Flag              | Default | Description                                              |
| ----------------- | ------- | -------------------------------------------------------- |
| `--shared-folder` | `None`  | Path for `LocalFolder`. If unset, uses `InMemoryFolder`. |
| `--num-nodes`     | `2`     | Number of simulated federated nodes                      |
| `--num-rounds`    | `5`     | Number of federated rounds                               |
| `--max-steps`     | `50`    | Max training steps per round per node                    |
| `--batch-size`    | `64`    | Batch size                                               |
| `--block-size`    | `256`   | Context window size                                      |
| `--lr`            | `5e-4`  | Learning rate                                            |

## Comparison with Standard Flower

| Feature              | Standard (`nanogpt-shakespeare`)     | Serverless (`nanogpt-serverless`)      |
| -------------------- | ------------------------------------ | -------------------------------------- |
| Server               | Flower `ServerApp`                   | None — shared folder only              |
| Run command           | `flwr run .`                         | `python -m nanogpt_serverless.main`    |
| Multi-machine        | SuperLink + SuperNodes               | Shared filesystem (NFS, S3, etc.)      |
| Node coordination    | Server-orchestrated                  | Async, peer-to-peer via shared storage |
| Aggregation          | Flower FedAvg strategy               | Flower FedAvg strategy (same)          |
