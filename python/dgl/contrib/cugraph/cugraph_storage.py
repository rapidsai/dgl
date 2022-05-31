# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

# NOTE: Requires cuGraph nightly cugraph-22.06.00a220417 or later

import cugraph
from cugraph.experimental import PropertyGraph
from .cugraph_utils import cugraphToDGL
import dgl
import cupy
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack

import cudf


class CuGraphStorage:
    """
    Duck-typed version of the DGL GraphStorage class made for cuGraph
    """

    def __init__(self, g, ndata=None, edata=None):
        # g must be a cugraph property graph
        if not isinstance(g, PropertyGraph):
            raise TypeError(f"g must be a PropertyGraphh, got {type(g)}")

        self.graphstore = cugraph.gnn.CuGraphStore(graph=g)
        self._edata = self.graphstore.edata
        self._ndata = self.graphstore.ndata
        self._edge_prop_df = self.graphstore.gdata._edge_prop_dataframe

    @property
    def ndata(self):
        ndata_capsule = self._ndata.to_dlpack()
        nfeat = from_dlpack(ndata_capsule)
        return nfeat.to(torch.float32)

    @property
    def edata(self):
        return self._edata

    def get_node_storage(self, key, ntype=None):
        node_col = self.graphstore.get_node_storage(key, ntype)
        return from_dlpack(node_col.to_dlpack())

    def get_edge_storage(self, key, etype=None):
        edge_col = self.graphstore.get_edge_storage(key, etype)
        return from_dlpack(edge_col.to_dlpack())

    # Required for checking whether single dict is allowed for ndata and edata
    @property
    def ntypes(self):
        data_ntypes = self._ndata[PropertyGraph.type_col_name]
        # TODO: double check the return type
        return data_ntypes

    @property
    def canonical_etypes(self):
        raise NotImplementedError("canonical not implemented")

    def etypes(self):
        data_etypes = self._edata[PropertyGraph.type_col_name]
        return data_etypes

    def sample_neighbors(
        self,
        seed_nodes,
        fanout,
        edge_dir="in",
        prob=None,
        exclude_edges=None,
        replace=False,
        output_device=None,
    ):
        """
        Return a DGLGraph which is a subgraph induced by sampling neighboring
        edges ofthe given nodes.
        See ``dgl.sampling.sample_neighbors`` for detailed semantics.
        Parameters
        ----------
        seed_nodes : Tensor or dict[str, Tensor]
            Node IDs to sample neighbors from.
            This argument can take a single ID tensor or a dictionary of node
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        fanout : int or dict[etype, int]
            The number of edges to be sampled for each node on each edge type.
            This argument can take a single int or a dictionary of edge types
            and ints. If a single int is given, DGL will sample this number of
            edges for each node for every edge type.
            If -1 is given for a single edge type, all the neighboring edges
            with that edge type will be selected.
        prob : str, optional
            Feature name used as the (unnormalized) probabilities associated
            with each neighboring edge of a node.  The feature must have only
            one element for each edge.
            The features must be non-negative floats, and the sum of the
            features of inbound/outbound edges for every node must be positive
            (though they don't have to sum up to one).  Otherwise, the result
            will be undefined. If :attr:`prob` is not None, GPU sampling is
            not supported.
        exclude_edges: tensor or dict
            Edge IDs to exclude during sampling neighbors for the seed nodes.
            This argument can take a single ID tensor or a dictionary of edge
            types and ID tensors. If a single tensor is given, the graph must
            only have one type of nodes.
        replace : bool, optional
            If True, sample with replacement.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            A sampled subgraph with the same nodes as the original graph, but
            only the sampled neighboring edges.  The induced edge IDs will be
            in ``edata[dgl.EID]``.
        """
        # change the seed_nodes from pytorch tensor to cudf series
        if torch.is_tensor(seed_nodes):
            seed_nodes = to_dlpack(seed_nodes)
            seed_nodes = cudf.from_dlpack(seed_nodes)

        parents_nodes, children_nodes = self.graphstore.sample_neighbors(
            seed_nodes, fanout, edge_dir="in", prob=None, replace=False
        )

        sample_df = cudf.DataFrame({"_SRC_": parents_nodes, "_DST_": children_nodes})
        del parents_nodes, children_nodes
        edge_df = sample_df.merge(
            self._edge_prop_df[["_SRC_", "_DST_", "_EDGE_ID_"]], on=["_SRC_", "_DST_"]
        )
        del sample_df

        edge_df = edge_df.astype(seed_nodes.dtype)

        # construct dgl graph, want to double check if children and parents
        # are in the correct order

        sampled_graph = dgl.graph(
            (
                from_dlpack(edge_df["_DST_"].to_dlpack()),
                from_dlpack(edge_df["_SRC_"].to_dlpack()),
            )
        )

        sampled_graph.edata["_ID"] = from_dlpack(edge_df["_EDGE_ID_"].to_dlpack())

        # to device function move the dgl graph to desired devices
        if output_device is not None:
            sampled_graph.to_device(output_device)

        return sampled_graph

    # Required in Cluster-GCN
    def subgraph(self, nodes, relabel_nodes=False, output_device=None):
        """Return a subgraph induced on given nodes.
        This has the same semantics as ``dgl.node_subgraph``.
        Parameters
        ----------
        nodes : nodes or dict[str, nodes]
            The nodes to form the subgraph. The allowed nodes formats are:
            * Int Tensor: Each element is a node ID. The tensor must have the
             same device type and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether node :math:`i` is in the subgraph.
             If the graph is homogeneous, directly pass the above formats.
             Otherwise, the argument must be a dictionary with keys being
             node types and values being the node IDs in the above formats.
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        sampled_cugraph = self.graphstore.node_subgraph(nodes)
        # the return type is cugraph subgraph
        sample_graph = cugraphToDGL(sampled_cugraph)
        sample_graph.to_device(output_device)
        return sample_graph

    # Required in Link Prediction
    # relabel = F we use dgl functions,
    # relabel = T, we need delete codes and relabel
    def edge_subgraph(self, edges, relabel_nodes=False, output_device=None):
        """
        Return a subgraph induced on given edges.
        This has the same semantics as ``dgl.edge_subgraph``.
        Parameters
        ----------
        edges : edges or dict[(str, str, str), edges]
            The edges to form the subgraph. The allowed edges formats are:
            * Int Tensor: Each element is an edge ID. The tensor must have the
              same device type and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * Bool Tensor: Each :math:`i^{th}` element is a bool flag
             indicating whether edge :math:`i` is in the subgraph.
            If the graph is homogeneous, one can directly pass the above
            formats. Otherwise, the argument must be a dictionary with keys
            being edge types and values being the edge IDs in the above formats
        relabel_nodes : bool, optional
            If True, the extracted subgraph will only have the nodes in the
            specified node set and it will relabel the nodes in order.
        output_device : Framework-specific device context object, optional
            The output device.  Default is the same as the input graph.
        Returns
        -------
        DGLGraph
            The subgraph.
        """
        pass

    # Required in Link Prediction negative sampler
    def find_edges(self, edges, etype=None, output_device=None):
        """Return the source and destination node IDs given the edge IDs within
        the given edge type.
        return type is tensor need to change.
        """
        # edges are a range of edge IDs, for example 0-100
        selected_edges = self._edata[self._edata["_TYPE_"] == etype].iloc[edges]
        src_nodes = selected_edges["_SRC_"]
        dst_nodes = selected_edges["_DST_"]
        src_nodes_tensor = from_dlpack(src_nodes.to_dlpack()).to(output_device)
        dst_nodes_tensor = from_dlpack(dst_nodes.to_dlpack()).to(output_device)
        return src_nodes_tensor, dst_nodes_tensor

    # Required in Link Prediction negative sampler
    def num_nodes(self, ntype):
        """Return the number of nodes for the given node type."""
        # use graphstore function
        return self._ndata[self._ndata["_TYPE_"] == ntype].shape[0]

    def global_uniform_negative_sampling(
        self, num_samples, exclude_self_loops=True, replace=False, etype=None
    ):
        """
        Per source negative sampling as in ``dgl.dataloading.GlobalUniform``
        """
        raise NotImplementedError("canonical not implemented")
