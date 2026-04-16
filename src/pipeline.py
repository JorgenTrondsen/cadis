def calculate_pipeline(workers_latency_data: list, master_latency_data: dict, master_ip: str) -> list:
    """
    Determines the optimal pipeline sequence by using a greedy nearest neighbor approach.
    Starting from the master node (Rank 0), it iteratively adds the node with the lowest
    latency from the previous node's perspective, ensuring each node is added only once.
    Args:
        workers_latency_data (list): Latency maps and IPs from each worker.
        master_latency_data (dict): Latency map from the master node.
        master_ip (str): The IP address of the master node.
    Returns:
        list: Ordered list of IP addresses representing the pipeline (Rank 0, 1, ...).
    """
    graph = {}
    graph[master_ip] = master_latency_data

    for data in workers_latency_data:
        ip = data["ip"]
        graph[ip] = data["latency"]

    all_nodes = list(graph.keys())

    if len(all_nodes) <= 1:
        return all_nodes

    best_path = [master_ip]
    remaining_nodes = [node for node in all_nodes if node != master_ip]

    while remaining_nodes:
        last_node = best_path[-1]
        best_next_node = None
        min_lat = float('inf')

        for node in remaining_nodes:
            if last_node in graph and node in graph[last_node]:
                lat = graph[last_node][node]
                if lat < min_lat:
                    min_lat = lat
                    best_next_node = node

        if best_next_node is None:
            best_next_node = remaining_nodes[0]

        best_path.append(best_next_node)
        remaining_nodes.remove(best_next_node)

    return best_path
