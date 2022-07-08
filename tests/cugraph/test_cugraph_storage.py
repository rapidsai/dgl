# Copyright (c) 2020-2021, NVIDIA CORPORATION.:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import pytest

from cugraph.experimental import PropertyGraph
from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage
import cudf
import cugraph
import dgl
import dgl.backend as F

dataset1 = {
    "merchants": [
        [
            "merchant_id",
            "merchant_locaton",
            "merchant_size",
            "merchant_sales",
            "merchant_num_employees",
        ],
        [
            (11, 78750, 44, 123.2, 12),
            (4, 78757, 112, 234.99, 18),
            (21, 44145, 83, 992.1, 27),
            (16, 47906, 92, 32.43, 5),
            (86, 47906, 192, 2.43, 51),
        ],
    ],
    "users": [
        ["user_id", "user_location", "vertical"],
        [
            (89021, 78757, 0),
            (32431, 78750, 1),
            (89216, 78757, 1),
            (78634, 47906, 0),
        ],
    ],
    "taxpayers": [
        ["payer_id", "amount"],
        [
            (11, 1123.98),
            (4, 3243.7),
            (21, 8932.3),
            (16, 3241.77),
            (86, 789.2),
            (89021, 23.98),
            (78634, 41.77),
        ],
    ],
    "transactions": [
        ["user_id", "merchant_id", "volume", "time", "card_num"],
        [
            (89021, 11, 33.2, 1639084966.5513437, 123456),
            (89216, 4, None, 1639085163.481217, 8832),
            (78634, 16, 72.0, 1639084912.567394, 4321),
            (32431, 4, 103.2, 1639084721.354346, 98124),
        ],
    ],
    "relationships": [
        ["user_id_1", "user_id_2", "relationship_type"],
        [
            (89216, 89021, 9),
            (89216, 32431, 9),
            (32431, 78634, 8),
            (78634, 89216, 8),
        ],
    ],
    "referrals": [
        ["user_id_1", "user_id_2", "merchant_id", "stars"],
        [
            (89216, 78634, 11, 5),
            (89021, 89216, 4, 4),
            (89021, 89216, 21, 3),
            (89021, 89216, 11, 3),
            (89021, 78634, 21, 4),
            (78634, 32431, 11, 4),
        ],
    ],
}


def create_CuGraphStore():
    """
    Fixture which returns an instance of a CuGraphStore with vertex and edge
    data added from dataset1, parameterized for different DataFrame types.
    """

    # util to create test
    def create_df_from_dataset(col_n, rows):
        data_d = defaultdict(list)
        for row in rows:
            for col_id, col_v in enumerate(row):
                data_d[col_n[col_id]].append(col_v)
        return cudf.DataFrame(data_d)

    merchant_df = create_df_from_dataset(
        dataset1["merchants"][0], dataset1["merchants"][1]
    )
    user_df = create_df_from_dataset(dataset1["users"][0], dataset1["users"][1])
    taxpayers_df = create_df_from_dataset(
        dataset1["taxpayers"][0], dataset1["taxpayers"][1]
    )
    transactions_df = create_df_from_dataset(
        dataset1["transactions"][0], dataset1["transactions"][1]
    )
    relationships_df = create_df_from_dataset(
        dataset1["relationships"][0], dataset1["relationships"][1]
    )
    referrals_df = create_df_from_dataset(
        dataset1["referrals"][0], dataset1["referrals"][1]
    )

    pG = PropertyGraph()
    graph = CuGraphStorage(pG)
    # Vertex and edge data is added as one or more DataFrames; either a Pandas
    # DataFrame to keep data on the CPU, a cuDF DataFrame to keep data on GPU,
    # or a dask_cudf DataFrame to keep data on distributed GPUs.

    # For dataset1: vertices are merchants and users, edges are transactions,
    # relationships, and referrals.

    # property_columns=None (the default) means all columns except
    # vertex_col_name will be used as properties for the vertices/edges.

    graph.add_node_data(merchant_df, "merchant_id", "merchant_k", "merchant")
    graph.add_node_data(user_df, "user_id", "user_k", "user")
    graph.add_node_data(taxpayers_df, "payer_id", "taxpayers_k", "taxpayers")

    graph.add_edge_data(
        referrals_df, ("user_id_1", "user_id_2"), "referrals_k", "referrals"
    )
    graph.add_edge_data(
        relationships_df, ("user_id_1", "user_id_2"), "relationships_k", "relationships"
    )
    graph.add_edge_data(
        transactions_df, ("user_id", "merchant_id"), "transactions_k", "transactions"
    )

    return graph, merchant_df, relationships_df


def test_dummy_ds_CuGraphStore():
    # Test 1 - loading a Property Graph
    gs, merchant_df, relationships_df = create_CuGraphStore()
    assert gs.num_nodes() == 9
    assert gs.num_edges() == 14

    # Test 2 - num_nodes
    assert gs.num_nodes("merchant") == 5

    # Test 3 - get_node_storage
    fs = gs.get_node_storage(key="merchant_k", ntype="merchant")
    merchent_gs = fs.fetch([11, 4, 21, 316, 11], device="cuda")
    cudf_t = F.tensor(
        merchant_df.set_index("merchant_id").loc[[11, 4, 21, 316, 11]].values
    ).to(device="cuda")
    assert F.allclose(cudf_t, merchent_gs)

    # Test 4- get_edge_storage

    fs = gs.get_edge_storage("relationships_k", "relationships")
    relationship_t = fs.fetch(F.tensor([6, 7, 8]), device="cuda")
    cudf_t = F.tensor(relationships_df["relationship_type"].iloc[[0, 1, 2]].values).to(
        device="cuda"
    )
    assert F.allclose(cudf_t, relationship_t)

    # Test 3 - Sampling
    nodes = F.tensor([4])
    x = gs.sample_neighbors(seed_nodes=nodes, fanout=1)
    assert x is not None


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

        gs.add_node_data(df, "node_id", key)
    return gs


def add_edata(gs, graph):
    src_t, dst_t = graph.edges()
    edge_ids = graph.edge_ids(src_t, dst_t)
    df = cudf.DataFrame(
        {
            "src": cudf.from_dlpack(F.zerocopy_to_dlpack(src_t)),
            "dst": cudf.from_dlpack(F.zerocopy_to_dlpack(dst_t)),
            "edge_id": cudf.from_dlpack(F.zerocopy_to_dlpack(edge_ids)),
        }
    )
    gs.add_edge_data(df, ["src", "dst"], "edge_id")
    return gs


@pytest.fixture
def dgl_Cora():
    dataset = dgl.data.CoraGraphDataset()
    graph = dataset[0].to("cuda")
    return graph


@pytest.fixture
def cugraphstore_Cora(dgl_Cora):
    # create pg from cugraph graph
    pg = cugraph.experimental.PropertyGraph()
    # create gs from pg
    gs = dgl.contrib.cugraph.CuGraphStorage(pg)
    add_edata(gs, dgl_Cora)
    add_ndata(gs, dgl_Cora)
    return gs


def test_get_node_storage(cugraphstore_Cora, dgl_Cora):
    # get_node_storage tests
    index_t = F.tensor([1, 2, 3, 4, 1, 5, 2704, 2707, 11]).to("cuda")
    device = "cuda"
    key = "label"
    dgl_g = dgl_Cora.get_node_storage(key).fetch(indices=index_t, device=device)
    cugraph_gs = cugraphstore_Cora.get_node_storage(key).fetch(
        indices=index_t, device=device
    )

    assert F.allclose(dgl_g, cugraph_gs)

    device = "cuda"
    key = "feat"
    dgl_o = dgl_Cora.get_node_storage(key).fetch(indices=index_t, device=device)
    cugraph_o = cugraphstore_Cora.get_node_storage(key).fetch(
        indices=index_t, device=device
    )

    assert F.allclose(dgl_o, cugraph_o)


def test_num_nodes(cugraphstore_Cora, dgl_Cora):
    assert cugraphstore_Cora.num_nodes() == dgl_Cora.num_nodes()


def test_num_edges(cugraphstore_Cora, dgl_Cora):
    assert cugraphstore_Cora.num_edges() == dgl_Cora.num_edges()


def test_sample_neighbors(cugraphstore_Cora, dgl_Cora):
    index_t = F.tensor([1, 2, 3, 4, 1, 5, 2704, 2707, 11, 19]).to("cuda")
    gs_output = cugraphstore_Cora.sample_neighbors(index_t, fanout=-1)
    gs_u, gs_v = gs_output.edges(order="srcdst")
    graph_output = dgl_Cora.sample_neighbors(index_t, fanout=-1)
    g_u, g_v = graph_output.edges(order="srcdst")

    assert F.allclose(gs_u, g_u)
    assert F.allclose(gs_v, g_v)

    gs_output = cugraphstore_Cora.sample_neighbors(index_t, fanout=10)
    gs_u, gs_v = gs_output.edges(order="srcdst")
    graph_output = dgl_Cora.sample_neighbors(index_t, fanout=10)
    g_u, g_v = graph_output.edges(order="srcdst")

    assert F.allclose(gs_u, g_u)
    assert F.allclose(gs_v, g_v)
