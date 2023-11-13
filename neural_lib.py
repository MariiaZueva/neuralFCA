from dataclasses import dataclass
from typing import List, Tuple, FrozenSet, Set, Dict
import pandas as pd

from fcapy.lattice import ConceptLattice
from fcapy.lattice.formal_concept import FormalConcept
from fcapy.poset import POSet
from fcapy.visualizer.line_layouts import calc_levels

import torch
from sparselinear import SparseLinear


@dataclass(eq=False)
class DisjunctiveNeuron:
    intent: FrozenSet[str]
    level: int

    def __eq__(self, other: 'DisjunctiveNeuron'):
        return self.intent == other.intent and self.level == other.level

    def __lt__(self, other: 'DisjunctiveNeuron'):
        return self.intent & other.intent == other.intent and self.level > other.level

    def __le__(self, other: 'DisjunctiveNeuron'):
        return self < other or self == other

    def __hash__(self):
        return hash((self.intent, self.level))


class ConceptNetwork:
    def __init__(self, poset: POSet, network=None, attributes: Tuple[str] = None, targets: Tuple[str] = None):
        self._poset = poset
        self._network = network
        self._attributes = attributes
        self._targets = targets

    @property
    def poset(self) -> POSet:
        return self._poset

    @property
    def network(self) -> torch.nn.Sequential:
        return self._network

    @property
    def attributes(self) -> Tuple[str]:
        return self._attributes

    @property
    def targets(self):
        return self._targets

    def trace_description(self, description: FrozenSet[str], include_targets: bool = False) -> Set[int]:
        P = self.poset

        tops_activated = [node for node in P.tops if P[node].intent & description == P[node].intent]
        activated_nodes = set(tops_activated)
        for node in tops_activated:
            activated_nodes |= P.descendants(node)
        if not include_targets:
            activated_nodes -= set(P.bottoms)

        return activated_nodes

    @classmethod
    def from_lattice(
            cls,
            lattice: ConceptLattice, best_concepts_indices: List[int],
            targets: Tuple[str]
    ):
        assert lattice.is_monotone, 'The lattice should be monotone'

        targets = tuple(targets)

        attrs_tpl = tuple(lattice[lattice.bottom].intent)
        P = cls._poset_from_best_concepts(lattice[best_concepts_indices], targets, attrs_tpl)
        P = cls._fill_levels(P)
        return cls(P, None, attributes=attrs_tpl, targets=targets)

    def fit(
            self,
            X_df: 'pd.DataFrame[bool]', y: 'pd.Series[bool]',
            loss_fn=torch.nn.CrossEntropyLoss(), nonlinearity=torch.nn.ReLU,
            n_epochs: int = 2000
    ):
        X = torch.tensor(X_df[list(self.attributes)].values).float()
        y = torch.tensor(y.values).long()

        self._network = self._poset_to_network(self.poset, nonlinearity)

        optimizer = torch.optim.Adam(self.network.parameters())

        for t in range(n_epochs):
            optimizer.zero_grad()
            y_pred = self.network(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

    def predict_proba(self, X_df: 'pd.DataFrame[bool]') -> torch.Tensor:
        X = torch.tensor(X_df[list(self.attributes)].values).float()
        return self.network(X)

    def predict(self, X_df: 'pd.DataFrame[bool]') -> torch.Tensor:
        return self.predict_proba(X_df).argmax(1)

    def edge_weights_from_network(self) -> Dict[Tuple[int, int], float]:
        max_level = self.poset[self.poset.bottoms[0]].level
        nodes_per_levels = {lvl: [] for lvl in range(max_level + 1)}
        for node_i, node in enumerate(self.poset):
            nodes_per_levels[node.level].append(node_i)
        nodes_per_levels = [nodes_per_levels[lvl] for lvl in range(max_level + 1)]

        edge_weights = {}
        for layer_i, nodes in enumerate(nodes_per_levels[:-1]):
            next_nodes = nodes_per_levels[layer_i+1]

            nn_layer = self.network[layer_i*2]
            idxs = nn_layer.weight.indices().numpy().T.tolist()
            vals = nn_layer.weight.values().numpy()

            for (child_i, parent_i), v in zip(idxs, vals):
                edge_weights[(nodes[parent_i], next_nodes[child_i])] = v
        return edge_weights

    @staticmethod
    def _poset_from_best_concepts(
            best_concepts: List[FormalConcept], targets: Tuple[str], attrs_tpl: Tuple[str]
    ) -> POSet:
        P_best = POSet(best_concepts)
        lvls = calc_levels(P_best)[0]
        lvls = [lvl + 1 for lvl in lvls]
        target_lvl = max(lvls) + 1

        attrs_set = set(attrs_tpl)

        best_neurons = [DisjunctiveNeuron(frozenset(c.intent), lvl) for c, lvl in zip(P_best, lvls)]
        first_level_neurons = [DisjunctiveNeuron(frozenset({m}), 0) for m in attrs_tpl]
        last_level_neurons = [DisjunctiveNeuron(frozenset({f"y={y}"} | attrs_set), target_lvl) for y in targets]
        return POSet(first_level_neurons + best_neurons + last_level_neurons)

    @staticmethod
    def _fill_levels(poset: POSet) -> POSet:
        nodes_i = sorted(range(len(poset)), key=lambda node_i: poset[node_i].level)
        for node_i in nodes_i:
            children_i = poset.children(node_i)
            if len(children_i) == 0:
                continue

            max_children_level = max([poset[child_i].level for child_i in children_i])
            for lvl in range(poset[node_i].level+1, max_children_level):
                poset.add(DisjunctiveNeuron(poset[node_i].intent, lvl))
        return poset

    @staticmethod
    def _poset_to_network(poset: POSet, nonlinearity: type = torch.nn.ReLU) -> 'torch.nn.Sequential':
        max_level = poset[poset.bottoms[0]].level
        nodes_per_levels = {lvl: [] for lvl in range(max_level + 1)}
        for node_i, node in enumerate(poset):
            nodes_per_levels[node.level].append(node_i)
        nodes_per_levels = [nodes_per_levels[lvl] for lvl in range(max_level + 1)]

        connectivities = []
        for layer_i, layer in enumerate(nodes_per_levels[1:]):
            layer_i += 1
            prev_layer = nodes_per_levels[layer_i - 1]
            layer_con = [(layer.index(node), prev_layer.index(parent))
                         for node in layer for parent in poset.parents(node)]
            connectivities.append(layer_con)

        linear_layers = []
        for layer_i in range(max_level):
            con = torch.tensor(connectivities[layer_i]).T
            layer = SparseLinear(len(nodes_per_levels[layer_i]), len(nodes_per_levels[layer_i + 1]), connectivity=con)
            linear_layers.append(layer)

        layers = [layer for ll in linear_layers for layer in [ll, nonlinearity()]][:-1] + [torch.nn.Softmax(dim=1)]
        model_sparse = torch.nn.Sequential(*layers)
        return model_sparse


def neuron_label_func(el_i: int, P: POSet, M: set, only_new_attrs: bool = True):
    el = P[el_i]

    if len(el.intent - M) > 0:  # if target node
        attrs_to_show = list(el.intent - M)
    else:
        attrs_to_show = set(el.intent)
        if only_new_attrs:
            for parent_i in P.parents(el_i):
                attrs_to_show = attrs_to_show - P[parent_i].intent

        attrs_to_show = list(attrs_to_show)
    return ','.join(attrs_to_show)
