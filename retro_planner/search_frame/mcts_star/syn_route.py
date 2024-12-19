from copy import deepcopy
import numpy as np
from queue import Queue
from graphviz import Digraph


class SynRoute:
    def __init__(self, target_mol, succ_value, search_status):
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.values = [None]
        self.templates = [None]
        self.parents = [-1]
        self.children = [None]
        self.optimal = False
        self.costs = {}
        self.mol_end = []
        self.succ = False
        self._dict_route = None
        self._rxn_attributes_dicts = None

        self.succ_value = succ_value
        self.total_cost = 0
        self.length = 0
        self.search_status = search_status
        if self.succ_value <= self.search_status:
            self.optimal = True
    @property
    def dict_route(self):
        if self._dict_route is None:
            self._dict_route = self.route_to_dict()
        return self._dict_route

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.values.append(None)
        self.templates.append(None)
        self.parents.append(parent_id)
        self.children.append(None)
        # self.mol_end.append(None)

        self.children[parent_id].append(len(self.mols)-1)

    def set_value(self, mol, value):
        assert mol in self.mols

        mol_id = self.mols.index(mol)
        self.values[mol_id] = value

    def add_reaction(self, mol, value, template, reactants, cost, mol_node=None):
        assert mol in self.mols

        self.total_cost += cost
        self.length += 1

        parent_id = self.mols.index(mol)
        self.values[parent_id] = value
        self.templates[parent_id] = template
        self.children[parent_id] = []
        self.costs[parent_id] = cost
        if mol_node and len(mol_node.children) == 0:
            self.mol_end.append(mol_node)

        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def add_route(self, mol_node, all_routes):
        if self.end_check():
            all_routes.append(self)
            return
        if mol_node.parent is not None:
            for sbling_mol_node in mol_node.parent.children:
                if sbling_mol_node.id != mol_node.id:
                    self.add_route(sbling_mol_node, all_routes)
        if mol_node.is_known:
            self.set_value(mol_node.mol, mol_node.succ_value)

        if not mol_node.children:
            self.mol_end.append(mol_node)

        else:
            for reaction_node in mol_node.children:
                new_route = deepcopy(self)
                reactants = []
                for reactant in reaction_node.children:
                    reactants.append(reactant.mol)
                new_route.add_reaction(
                    mol=mol_node.mol,
                    value=mol_node.succ_value,
                    template=reaction_node.template,
                    reactants=reactants,
                    cost=reaction_node.cost,
                    mol_node=mol_node
                )
                for reactant in reaction_node.children:

                    new_route.add_route(reactant, all_routes)

    def route_to_dict(self):
        all_dict = []
        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            # if self.templates[i] is not None:
            #     name += ' | %s' % self.templates[i]
            mol_dict = {
                'smiles': name,
                'type': 'mol',
                'in_stock': None,
                'values':self.values[i],
                'children': [],
            }
            names.append(name)
            all_dict.append(mol_dict)
        node_queue = Queue()
        node_queue.put((0, -1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()
            mol_dict = all_dict[idx]

            if self.children[idx] is not None:
                reaction_dict = {
                    'type': 'reaction',
                    'smiles': '',
                    'template': self.templates[idx],
                    'children': [],

                }
                all_dict[idx]['in_stock'] = False
                for c in self.children[idx]:
                    reaction_dict['children'].append(all_dict[c])
                    node_queue.put((c, idx))
                all_dict[idx]['children'].append(reaction_dict)
            else:
                all_dict[idx]['in_stock'] = True
                all_dict[idx].pop('children')
        return all_dict[0]
            
    def viz_route(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr('node', shape='box')
        G.format = 'pdf'

        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            # if self.templates[i] is not None:
            #     name += ' | %s' % self.templates[i]
            names.append(name)

        node_queue = Queue()
        node_queue.put((0, -1))   # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                G.edge(names[parent_idx], names[idx], label='cost')

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))

        G.render()

    def serialize_reaction(self, idx):
        s = self.mols[idx]
        if self.children[idx] is None:
            return s
        s += '>%.4f>' % np.exp(-self.costs[idx])
        s += self.mols[self.children[idx][0]]
        for i in range(1, len(self.children[idx])):
            s += '.'
            s += self.mols[self.children[idx][i]]

        return s

    def serialize(self):
        s = self.serialize_reaction(0)
        for i in range(1, len(self.mols)):
            if self.children[i] is not None:
                s += '|'
                s += self.serialize_reaction(i)

        return s
