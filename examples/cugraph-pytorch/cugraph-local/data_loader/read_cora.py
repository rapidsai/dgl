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


import cudf
from cugraph.experimental import PropertyGraph
from dgl.contrib.cugraph import CuGraphStorage
import numpy as np
import random
import os


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def read_cora(raw_path, self_loop=False):
    graph_path = os.path.join(raw_path, "cora.cites")
    feat_path = os.path.join(raw_path, "cora.content")
    cora_M = cudf.read_csv(graph_path, sep="\t", header=None)
    cora_content = cudf.read_csv(feat_path, sep="\t", header=None)
    # the last column is true label
    labels = cora_content["1434"]
    cora_content.drop(columns="1434", inplace=True)
    # add weight into graph
    cora_M["weight"] = 1.0

    # add features to nodes and edges
    pg = PropertyGraph()

    pg.add_edge_data(cora_M, vertex_col_names=("0", "1"))
    pg.add_vertex_data(cora_content, vertex_col_name="0")

    pg._vertex_prop_dataframe.drop(columns=["0"], inplace=True)
    pg._edge_prop_dataframe.drop(columns=["0", "1"], inplace=True)
    pg._vertex_prop_dataframe.drop(columns=["_TYPE_"], inplace=True)
    pg._vertex_prop_dataframe.drop(columns=["_VERTEX_"], inplace=True)

    gstore = CuGraphStorage(pg)

    # define train, test and val splits
    indices = np.arange(len(labels))
    random.shuffle(indices)
    idx_train, idx_val, idx_test = np.split(indices, [1000, 1500])

    return gstore, labels, idx_train, idx_val, idx_test
