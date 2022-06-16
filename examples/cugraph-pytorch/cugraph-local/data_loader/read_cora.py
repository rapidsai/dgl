# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

import dgl
import os
import random
import numpy as np

# Util to create cora graph based on DGL
def create_cora_cugraph_pg_from_dgl(g):
    import cudf
    import cupy as cp
    from cugraph.experimental import PropertyGraph

    pg = PropertyGraph()

    edges = g.edges()
    edge_data = cudf.DataFrame(
        {"src": cp.asarray(edges[0]), "dst": cp.asarray(edges[1])}
    )
    del edges
    pg.add_edge_data(edge_data, vertex_col_names=["src", "dst"])
    del edge_data

    feat_ar = cp.asarray(g.ndata["feat"])
    feat_df = cudf.DataFrame(feat_ar)
    feat_df["vertex_id"] = cp.arange(0, len(feat_df))
    pg.add_vertex_data(feat_df, vertex_col_name="vertex_id")

    pg._vertex_prop_dataframe.drop(columns=["_TYPE_"], inplace=True)
    pg._vertex_prop_dataframe.drop(columns=["_VERTEX_"], inplace=True)
    pg._vertex_prop_dataframe.drop(columns=["vertex_id"], inplace=True)

    return pg


def read_cora():
    from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage
    import cupy

    data = dgl.data.CoraGraphDataset()
    g = data[0]
    pg = create_cora_cugraph_pg_from_dgl(g)

    gstore = CuGraphStorage(pg)
    del pg

    labels = g.ndata["label"]

    indices = np.arange(len(labels))
    random.shuffle(indices)
    idx_train, idx_val, idx_test = np.split(indices, [1000, 1500])
    return gstore, labels, idx_train, idx_val, idx_test


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)
