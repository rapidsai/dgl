import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse

# Example based on
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py
## cugraph imports
import cugraph
import cudf
import cupy
import time


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = "cpu"
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(args, device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size)
        pred = pred[nid]
        label = graph.ndata["label"][nid]
        return MF.accuracy(pred, label)


def train(device, g, dataset, model):

    # batch size
    b_size = 128

    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for layer-0, layer-1 and layer-2
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = False
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=b_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=False,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=b_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        last_it_t = time.time()
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            print(f"Time taken for {it} = {time.time()-last_it_t} s")
            last_it_t = time.time()

        acc = evaluate(model, g, val_dataloader)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )


# util to create property graph from ns
def add_ndata(gs, graph):
    if len(graph.ntypes) != 1:
        raise "graph.ntypes!=1 not currently supported"

    for key in graph.ndata.keys():
        # TODO: Change to_dlpack
        # tensor = tensor.to('cpu')
        # ar = cupy.from_dlpack(F.zerocopy_to_dlpack(tensor))
        ar = graph.ndata[key].cpu().detach().numpy()
        cudf_data_obj = cudf.DataFrame(ar)
        # handle 1d tensors
        if isinstance(cudf_data_obj, cudf.Series):
            df = cudf_data_obj.to_frame()
        else:
            df = cudf_data_obj

        df.columns = [f"{key}_{i}" for i in range(len(df.columns))]
        node_ids = dgl.backend.zerocopy_to_dlpack(graph.nodes())
        node_ser = cudf.from_dlpack(node_ids)
        df["node_id"] = node_ser

        gs.add_node_data(df, "node_id", key, graph.ntypes[0])
    return gs


def add_edata(gs, graph):
    src_t, dst_t = graph.edges()
    edge_ids = graph.edge_ids(src_t, dst_t)
    df = cudf.DataFrame(
        {
            "src": cudf.from_dlpack(dgl.backend.zerocopy_to_dlpack(src_t)),
            "dst": cudf.from_dlpack(dgl.backend.zerocopy_to_dlpack(dst_t)),
            "edge_id": cudf.from_dlpack(dgl.backend.zerocopy_to_dlpack(edge_ids)),
        }
    )
    gs.add_edge_data(df, ["src", "dst"], "edge_id")
    return gs


def create_cugraph_store(graph):
    # create pg from cugraph graph
    pg = cugraph.experimental.PropertyGraph()
    # create gs from pg
    gs = dgl.contrib.cugraph.CuGraphStorage(pg)
    add_edata(gs, graph)
    add_ndata(gs, graph)
    return gs


### Create a cugraph store dataset from
### DGL.graph
dataset = AsNodePredDataset(
    dgl.data.CoraGraphDataset(raw_dir="/datasets/vjawa/CoraGraphDataset/")
)
g = dataset[0]
g = g.to("cuda")
g = create_cugraph_store(g)

# Create GraphSAGE model
in_size = g.ndata["feat"].shape[1]
out_size = dataset.num_classes
model = SAGE(in_size, 256, out_size).to("cuda")

# # model training
# print("Training")
train(torch.device("cuda"), g, dataset, model)
