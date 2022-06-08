import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from model import SAGE
import sys

sys.path.append("../data_loader")


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(
            g, nfeat, device, args.batch_size, args.num_workers
        )
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid].to(pred.device))


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


# Entry point
def run(args, device, data):
    # Unpack data
    # the reading data part need to be changed
    (
        n_classes,
        train_g,
        val_g,
        test_g,
        train_nfeat,
        train_labels,
        val_nfeat,
        val_labels,
        test_nfeat,
        test_labels,
        idx_train,
        idx_val,
        idx_test,
    ) = data
    in_feats = train_nfeat.shape[1]
    dataloader_device = device

    train_nid = th.tensor(idx_train, device=dataloader_device)
    test_nid = th.tensor(idx_test, device=dataloader_device)
    val_nid = th.tensor(idx_val, device=dataloader_device)

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )

    # their new dataloader will automatically call our graphsage.
    # no need to change this part
    dataloader = dgl.dataloading.DataLoader(
        train_g,
        train_nid,
        sampler,
        device=dataloader_device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Define model and optimizer
    model = SAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation
        # dependency graph as a list of blocks.
        tic_step = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(
                train_nfeat, train_labels, seeds, input_nodes, device
            )
            blocks = [block.int().to(device) for block in blocks]

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                # gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 \
                #                if th.cuda.is_available() else 0
                print(
                    f"Epoch {epoch: 05d} | Step {step: 05d}" +
                    f"| Loss {loss.item(),: .4f}" +
                    f"| Train Acc {acc.item(): .4f} |" +
                    f"Speed (samples/sec) {np.mean(iter_tput[3:]),: .4f}"
                    )
            tic_step = time.time()

        toc = time.time()
        print("Epoch Time(s): {:.4f}".format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(
                model, val_g, val_nfeat, val_labels, val_nid, device
            )
            print("Eval Acc {:.4f}".format(eval_acc))
            test_acc = evaluate(
                model, test_g, test_nfeat, test_labels, test_nid, device
            )
            print("Test Acc: {:.4f}".format(test_acc))

    print("Avg epoch time: {}".format(avg / (epoch - 4)))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--dataset", type=str, default="reddit")
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=16)
    argparser.add_argument("--num-layers", type=int, default=2)
    argparser.add_argument("--fan-out", type=str, default="10,25")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes." "Use 0 for no extra process.",
    )
    argparser.add_argument(
        "--sample-gpu",
        action="store_true",
        help="Perform the sampling process on the GPU"
        " Must have 0 workers.",
    )
    argparser.add_argument(
        "--inductive",
        action="store_true",
        help="Inductive learning setting",
    )
    argparser.add_argument(
        "--data-cpu",
        action="store_true",
        help="By default the script puts all node features"
        "and labels on GPU when using it to save time"
        "for data copy. This may be undesired if they"
        "cannot fit in GPU memory at once. "
        "This flag disables that.",
    )
    args = argparser.parse_args()

    if args.gpu >= 0:
        import numba.cuda as cuda
        # Order is very important, do this first before cuda work
        cuda.select_device(
            args.gpu
        )  # Create cuda context on the right gpu, defaults to gpu-0
        device = th.device("cuda:%d" % args.gpu)
    else:
        raise Exception("Unsupported device")

    if args.dataset == "cora":
        from read_cora import read_cora

        raw_path = "../datasets/cora"
        gstore, labels, idx_train, idx_val, idx_test = read_cora(raw_path)
        n_classes = 7

        # we only consider transductive cases for now
        train_g = val_g = test_g = gstore
        train_nfeat = val_nfeat = test_nfeat = gstore.ndata
        train_labels = val_labels = test_labels = th.tensor(
            labels, dtype=th.long
        )

    elif args.dataset == "reddit":
        raw_path = "../datasets/reddit"
        from read_reddit import read_reddit

        gstore, labels, train_mask, val_mask, test_mask = read_reddit(
            raw_path
        )
        n_classes = 41
        train_g = val_g = test_g = gstore
        train_nfeat = val_nfeat = test_nfeat = gstore.ndata
        train_labels = val_labels = test_labels = th.tensor(
            labels, dtype=th.long
        )
        # need to add more code from changing mask to id
        idx_train = np.nonzero(train_mask)[0]
        idx_test = np.nonzero(test_mask)[0]
        idx_val = np.nonzero(val_mask)[0]

    else:
        raise Exception("unknown dataset")

    # Pack data
    data = (
        n_classes,
        train_g,
        val_g,
        test_g,
        train_nfeat,
        train_labels,
        val_nfeat,
        val_labels,
        test_nfeat,
        test_labels,
        idx_train,
        idx_val,
        idx_test,
    )

    run(args, device, data)
