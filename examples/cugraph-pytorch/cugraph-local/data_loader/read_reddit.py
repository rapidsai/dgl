# Copyright (c) 2019-2022, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at   #
# http://www.apache.org/licenses/LICENSE-2.0   #
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cudf
from cugraph.experimental import PropertyGraph
from dgl.contrib.cugraph import CuGraphStorage
import numpy as np
import scipy.sparse as sp
import os


def read_reddit(raw_path, self_loop=False):
    coo_adj = sp.load_npz(os.path.join(raw_path, "reddit_graph.npz"))
    edgelist = cudf.DataFrame()
    edgelist["src"] = cudf.Series(coo_adj.row)
    edgelist["dst"] = cudf.Series(coo_adj.col)
    edgelist["wt"] = cudf.Series(coo_adj.data)

    # features and labels
    reddit_data = np.load(os.path.join(raw_path, "reddit_data.npz"))
    features = reddit_data["feature"]
    cu_features = cudf.DataFrame(features)
    cu_features["name"] = np.arange(cu_features.shape[0])
    labels = reddit_data["label"]
    # tarin/val/test indices
    node_types = reddit_data["node_types"]
    train_mask = node_types == 1
    val_mask = node_types == 2
    test_mask = node_types == 3
    # add features to nodes and edges
    pg = PropertyGraph()

    pg.add_edge_data(edgelist, vertex_col_names=("src", "dst"))

    pg.add_vertex_data(cu_features, vertex_col_name="name")
    pg._vertex_prop_dataframe.drop(columns=["name"], inplace=True)
    pg._vertex_prop_dataframe.drop(columns=["_TYPE_"], inplace=True)
    pg._vertex_prop_dataframe.drop(columns=["_VERTEX_"], inplace=True)

    gstore = CuGraphStorage(pg)

    return gstore, labels, train_mask, val_mask, test_mask
