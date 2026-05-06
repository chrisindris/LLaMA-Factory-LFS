import ray
import os
import time
import sys

# Wait for Ray cluster to be ready
print("Waiting for Ray cluster to be ready...", flush=True)
max_retries = 30
retry_count = 0
connected = False

while retry_count < max_retries and not connected:
    try:
        # Try to initialize connection
        ray.init(
            address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",
            _node_ip_address=os.environ['HEAD_NODE'],
            ignore_reinit_error=True
        )
        # Verify cluster is actually ready by checking nodes
        nodes = ray.nodes()
        alive_nodes = [node for node in nodes if node.get('Alive', False)]
        if len(alive_nodes) >= 2:
            connected = True
            print(f"Successfully connected to Ray cluster after {retry_count} retries", flush=True)
            print(f"Found {len(alive_nodes)} alive nodes", flush=True)
        else:
            print(f"Cluster connected but only {len(alive_nodes)}/{len(nodes)} nodes are alive, waiting...", flush=True)
            if hasattr(ray, 'disconnect'):
                ray.disconnect()
            retry_count += 1
            time.sleep(1)
    except Exception as e:
        if retry_count < max_retries - 1:
            print(f"Connection attempt {retry_count + 1} failed: {e}, retrying...", flush=True)
            retry_count += 1
            time.sleep(1)
        else:
            print(f"Failed to connect to Ray cluster after {max_retries} attempts: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

if not connected:
    print("Failed to connect to Ray cluster", file=sys.stderr, flush=True)
    sys.exit(1)

# Check that Ray sees two nodes and their status is 'Alive'
print("Nodes in the Ray cluster:", flush=True)
nodes = ray.nodes()
print(nodes, flush=True)

# Check that Ray sees 2 CPUs and 2 GPUs over 2 Nodes
print("Available resources:", flush=True)
resources = ray.available_resources()
print(resources, flush=True)

# Properly disconnect from Ray before exiting
if hasattr(ray, 'disconnect'):
    ray.disconnect()
elif hasattr(ray, 'shutdown'):
    # Don't shutdown the cluster, just disconnect
    pass

print("Script completed successfully", flush=True)