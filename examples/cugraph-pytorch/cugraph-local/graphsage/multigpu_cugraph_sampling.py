import dgl
import torch
import time


# Utils to create cora graph based on DGL
def create_cora_cugraph_from_dgl(g):
    import cudf
    import cupy
    from cugraph.experimental import PropertyGraph

    pg = PropertyGraph()

    edges = g.edges()
    edge_data = cudf.DataFrame(
        {"source": cupy.asarray(edges[0]), "destination": cupy.asarray(edges[1])}
    )
    del edges
    pg.add_edge_data(edge_data, vertex_col_names=["source", "destination"])
    del edge_data
    return pg


def ceate_cora_cugraph_from_dgl():
    from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage

    data = dgl.data.CoraGraphDataset()
    g = data[0]
    gs = CuGraphStorage(create_cora_cugraph_from_dgl(g))

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = data.graph.number_of_edges()
    print(
        """----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d"""
        % (n_edges, n_classes, train_mask.sum(), val_mask.sum(), test_mask.sum())
    )

    return gs, features, labels, train_mask, val_mask, test_mask


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def run_workflow(proc_id, devices):

    # Below sets gpu_num
    dev_id = devices[proc_id]

    import numba.cuda as cuda  # Order is very important, do this first before cuda work

    cuda.select_device(
        dev_id
    )  # Create cuda context on the right gpu, defaults to gpu-0
    import cudf #TODO: Maybe dont need to import
    import cugraph #TODO: Maybe dont need to import
    import cupy

    # Start the init_process_group
    torch.cuda.set_device(dev_id)
    device = torch.device(f"cuda:{dev_id}")

    dist_init_method = "tcp://{master_ip}:{master_port}".format(
        master_ip="127.0.0.1", master_port="12345"
    )
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=dist_init_method,
        world_size=len(devices),
        rank=proc_id,
    )

    (
        gs,
        features,
        labels,
        train_mask,
        val_mask,
        test_mask,
    ) = ceate_cora_cugraph_from_dgl()
    print("Created Dataset", flush=True)

    train_nid = cupy.arange(len(train_mask))[train_mask]
    # Create PyTorch DataLoader for constructing blocks

    # prefetch_node_feats=['feat'], prefetch_labels=['label']
    sampler = dgl.dataloading.MultiLayerNeighborSampler([10, 20])

    # Their new dataloader will automatically call our graphsage.
    # no need to change this part
    dataloader = dgl.dataloading.DataLoader(
        gs,
        train_nid,
        sampler,
        device=device,  # Put the sampled MFGs on CPU or GPU
        use_ddp=True,  # Make it work with distributed data parallel
        batch_size=1000,
        shuffle=True,  # Whether to shuffle the nodes for every epoch
        drop_last=False,
        num_workers=0,
    )

    del gs  # Clean up gs reference

    for _ in range(0, 100):
        st = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            batch_inputs, batch_labels = load_subtensor(
                features, labels, seeds, input_nodes, device
            )
            print("Input batch length =  {}".format(len(batch_inputs)))

            del batch_inputs, batch_labels
        et = time.time()
        print(f"Data Loading took = {et-st} s")

    # TODO: Fix Cclean up a problem that can occur sometimes
    time.sleep(200)


if __name__ == "__main__":
    num_gpus = 16
    import torch.multiprocessing as mp

    mp.spawn(run_workflow, args=(list(range(num_gpus)),), nprocs=num_gpus)
