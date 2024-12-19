import pandas as pd
import numpy as np
from queue import Queue
from graphviz import Digraph
from retro_planner.viz_utils import image
from retro_planner.viz_utils.chem import TreeMolecule, Reaction


class SynRoute:
    def __init__(self, target_mol):
        self.target_mol = target_mol
        self.mols = [target_mol]
        self.templates = [None]
        self.parents = [-1]
        self.children = [None]
        self.leaf = []

    def _add_mol(self, mol, parent_id):
        self.mols.append(mol)
        self.templates.append(None)
        self.parents.append(parent_id)

        self.children.append(None)
        self.children[parent_id].append(len(self.mols) - 1)

    def add_reaction(self, mol, reactants, template):
        assert mol in self.mols
        parent_id = self.mols.index(mol)
        self.templates[parent_id] = template
        self.children[parent_id] = []
        for reactant in reactants:
            self._add_mol(reactant, parent_id)

    def viz_route(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr('node', shape='box')
        G.format = 'pdf'

        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            names.append(name)

        node_queue = Queue()
        node_queue.put((0, -1))  # target mol idx, and parent idx
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            if parent_idx >= 0:
                G.edge(names[parent_idx], names[idx], label='cost')

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))

        G.render()

    def viz_graph_route(self):
        graph = image.GraphvizReactionGraph()
        names = []
        for i in range(len(self.mols)):
            name = self.mols[i]
            names.append(name)
        node_queue = Queue()
        node_queue.put((0, -1))  # target mol idx, and parent idx
        edge_list = []
        TreeMolDic = {}
        frame_type = []
        for is_leaf in self.leaf:
            if is_leaf:
                frame_type.append('green')
            else:
                frame_type.append('orange')
        # TreeReaDic = {}
        while not node_queue.empty():
            idx, parent_idx = node_queue.get()

            def saveToDic(idx):
                if idx in TreeMolDic:
                    pass
                else:
                    TreeMolDic[idx] = TreeMolecule(smiles=names[idx], parent=None)

            if parent_idx >= 0:
                edge_list.append((parent_idx, idx))
                saveToDic(parent_idx)
                saveToDic(idx)

            if self.children[idx] is not None:
                for c in self.children[idx]:
                    node_queue.put((c, idx))

        for idx, treemol in list(sorted(TreeMolDic.items(), key=lambda x: x[0], reverse=True)):
            graph.add_molecule(treemol, frame_type[idx])
        for out_idx, in_idx in edge_list:
            # graph.add_edge(TreeMolDic[in_idx], TreeMolDic[out_idx])
            graph.add_edge(TreeMolDic[out_idx], TreeMolDic[in_idx])
        return graph


def change_empty(x):
    if x:
        return x
    else:
        return None


def get_route_tree(tree_dic):
    assert tree_dic['type'] == 'mol'
    syn_route = SynRoute(
        target_mol=tree_dic['smiles']
    )
    mol_quene = Queue()
    mol_quene.put(tree_dic)

    while not mol_quene.empty():
        mol = mol_quene.get()
        if 'children' in mol.keys():
            reaction = mol['children'][0]
            assert reaction['type'] == 'reaction'
            reactants = []
            for reactant in reaction['children']:
                assert reactant['type'] == 'mol'
                mol_quene.put(reactant)
                reactants.append(reactant['smiles'])
            syn_route.add_reaction(
                mol=mol['smiles'],
                reactants=reactants,
                template=reaction['smiles']
            )
        else:
            syn_route.add_reaction(
                mol=mol['smiles'],
                reactants=[],
                template=None
            )

    syn_route.children = [change_empty(i) for i in syn_route.children]
    return syn_route


def copy_route_tree(syn_class):
    new_syn_class = SynRoute(
        syn_class.target_mol
    )
    new_syn_class.mols = syn_class.mols
    new_syn_class.templates = syn_class.templates
    new_syn_class.parents = syn_class.parents
    new_syn_class.children = syn_class.children
    assert len(new_syn_class.parents) == len(new_syn_class.children) == len(new_syn_class.mols)
    for mol, chd in zip(new_syn_class.mols, new_syn_class.children):
        if chd is None:
            new_syn_class.leaf.append(True)
        else:
            new_syn_class.leaf.append(False)
    return new_syn_class


if __name__ == '__main__':
    import pickle

    with open('../retro_planner/results/plan.pkl', 'rb') as f:
        results = pickle.load(f)
    results_df = pd.DataFrame.from_dict(results)
    results_succ_df = results.loc[results['succ'] == True]
    route_class = results['routes'].tolist()
    new_route_class = []
    for syn_class in route_class:
        if syn_class is not None:
            new_syn_class = copy_route_tree(syn_class)
            new_route_class.append(new_syn_class)

        else:
            new_route_class.append(None)
    results['new_route_for_viz'] = new_route_class
    results.to_hdf('./train_all_dataset/dfs/test_190mol/test_190mol.hdf', 'table')
