import numpy as np
from queue import Queue
import logging
from copy import deepcopy
import networkx as nx
from graphviz import Digraph
from retro_planner.search_frame.mcts_star.mol_node import MolNode
from retro_planner.search_frame.mcts_star.reaction_node import ReactionNode
from retro_planner.search_frame.mcts_star.syn_route import SynRoute


def split_route_fn(node, tree):
    new_tree = deepcopy(tree)
    new_node = deepcopy(node)
    org_parent_id = node.parent.serialize()
    new_tree.id2node[org_parent_id].children = []
    new_tree.id2node[org_parent_id].children.append(new_node)
    new_node.parent = new_tree.id2node[org_parent_id]
    new_tree.init_and_or()
    return new_node, new_tree


def remove_not_succ_node(tree):
    known_mols = tree.known_mols
    tree.known_mols = set()
    new_tree = deepcopy(tree)
    tree.known_mols = known_mols
    node_queue = Queue()
    node_queue.put(new_tree.root)
    while not node_queue.empty():
        node = node_queue.get()
        if hasattr(node, 'mol'):
            children = node.children
            node.children = []
            for child in children:
                if child.succ:
                    node.children.append(child)
                    node_queue.put(child)
        else:
            for child in node.children:
                node_queue.put(child)
    new_tree.init_and_or()
    return new_tree


class MolTree:
    def __init__(self, target_mol, known_mols, value_fn, zero_known_value=True, max_depth=20):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_fn = value_fn
        self.zero_known_value = zero_known_value
        self.max_depth = max_depth
        self.mol_nodes = []
        self.reaction_nodes = []

        self.root = self._add_mol_node(target_mol, None)
        self.succ = target_mol in known_mols
        self.search_status = 0

        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')

    def _add_mol_node(self, mol, parent):
        is_known = mol in self.known_mols

        init_value = self.value_fn(mol)

        mol_node = MolNode(
            mol=mol,
            init_value=init_value,
            parent=parent,
            is_known=is_known,
            zero_known_value=self.zero_known_value,
            max_depth=self.max_depth
        )
        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)

        return mol_node

    def _add_reaction_and_mol_nodes(self, cost, mols, parent, template, ancestors):
        assert cost >= 0

        for mol in mols:
            if mol in ancestors:
                return

        reaction_node = ReactionNode(parent, cost, template)
        current_mol_nodes = []
        for mol in mols:
            new_mol_node = self._add_mol_node(mol, reaction_node)
            current_mol_nodes.append(new_mol_node)
        for new_mol_node in current_mol_nodes:
            for sibling_mol_node in current_mol_nodes:
                if new_mol_node.id != sibling_mol_node.id:
                    new_mol_node.sibling.append(sibling_mol_node)
        reaction_node.init_values()
        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def _select_promising_grandchild(self, mol_node):
        assert mol_node

        if len(mol_node.children) == 0 and mol_node.open:  # 应该不会发生这个，先放着
            logging.info('Rollout fails on %s!' % mol_node.mol)
            return
        if mol_node.depth == self.max_depth - 1:  # 快要撞南墙了~~
            assert not mol_node.open
            mol_node.go_back = True
            return mol_node

        grandchildren_values = []
        grandchildren = []
        succ_mol_sbiling = []
        for child in mol_node.children:
            if child is not None:
                for grandchild in child.children:
                    if grandchild.is_terminal():
                        grandchildren_values.append(np.inf)
                    else:
                        grandchildren_values.append(grandchild.v_target())
                    if grandchild.succ and grandchild.sibling:
                        succ_mol_sbiling.extend(grandchild.sibling)
                    else:
                        pass
                    grandchildren.append(grandchild)
        succ_mol_sbiling = list(set(succ_mol_sbiling))
        if succ_mol_sbiling:
            succ_mol_sbiling_values = []
            for grandchild in succ_mol_sbiling:
                if grandchild.is_terminal():
                    succ_mol_sbiling_values.append(np.inf)
                else:
                    succ_mol_sbiling_values.append(grandchild.v_target())
            succ_mol_sbiling_values = np.array(succ_mol_sbiling_values)
            rollout_next = succ_mol_sbiling[np.argmin(succ_mol_sbiling_values)]
            return rollout_next
        grandchildren_values = np.array(grandchildren_values)
        rollout_next = grandchildren[np.argmin(grandchildren_values)]
        return rollout_next

    def expand(self, mol_node, reactant_lists, costs, templates, max_depth=10):  # TODO 添加最大深度
        assert not mol_node.is_known and not mol_node.children

        if (costs is None) or (mol_node.depth > max_depth):  # 已经添加最大深度    # No expansion results
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ, mol_node.succ

        assert mol_node.open
        ancestors = mol_node.get_ancestors()
        for i in range(len(costs)):
            self._add_reaction_and_mol_nodes(costs[i], reactant_lists[i],
                                             mol_node, templates[i], ancestors)

        if len(mol_node.children) == 0:      # No valid expansion results
            assert mol_node.init_values(no_child=True) == np.inf
            if mol_node.parent:
                mol_node.parent.backup(np.inf, from_mol=mol_node.mol)
            return self.succ, mol_node.succ

        v_delta = mol_node.init_values()
        if mol_node.parent:
            mol_node.parent.backup(v_delta, from_mol=mol_node.mol)

        if not self.succ and self.root.succ:
            # logging.info('Synthesis route found!')
            self.succ = True

        return self.succ, mol_node.succ

    def rollout(self, mol_node, expand_fn, max_depth=10):
        try:
            result = expand_fn(mol_node.mol)
        except:
            result = None
        if result is not None and (len(result['scores']) > 0):
            reactants = result['reactants']
            scores = result['scores']
            costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
            # costs = 1.0 - np.array(scores)
            if 'templates' in result.keys():
                templates = result['templates']
            else:
                templates = result['template']

            reactant_lists = []
            for j in range(len(scores)):
                reactant_list = list(set(reactants[j].split('.')))
                reactant_lists.append(reactant_list)

            # assert mol_node.rolloutable
            assert mol_node.open
            succ, mol_node_succ = self.expand(
                mol_node, reactant_lists, costs, templates, max_depth=max_depth)
            return succ, mol_node_succ

        else:
            self.expand(mol_node, None, None, None)
            logging.info('Rollout expansion fails on %s!' % mol_node.mol)
            return False, False

    def get_best_route(self):
        if not self.succ:
            return None

        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )

        mol_queue = Queue()
        mol_queue.put(self.root)
        while not mol_queue.empty():
            mol = mol_queue.get()
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue

            best_reaction = None
            for reaction in mol.children:
                if reaction.succ:
                    if best_reaction is None or \
                            reaction.succ_value < best_reaction.succ_value:
                        best_reaction = reaction
            assert best_reaction.succ_value == mol.succ_value

            reactants = []
            for reactant in best_reaction.children:
                mol_queue.put(reactant)
                reactants.append(reactant.mol)

            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=best_reaction.template,
                reactants=reactants,
                cost=best_reaction.cost
            )
        syn_route.succ = True

        return syn_route

    def convert_succ_route(self):
        if not self.succ:
            return None

        syn_route = SynRoute(
            target_mol=self.root.mol,
            succ_value=self.root.succ_value,
            search_status=self.search_status
        )

        mol_queue = Queue()
        mol_queue.put(self.root)
        while not mol_queue.empty():
            mol = mol_queue.get()
            if mol.is_known:
                syn_route.set_value(mol.mol, mol.succ_value)
                continue
            assert len(mol.children) == 1
            reaction = mol.children[0]
            assert reaction.succ
            reactants = []
            for reactant in reaction.children:
                mol_queue.put(reactant)
                reactants.append(reactant.mol)

            syn_route.add_reaction(
                mol=mol.mol,
                value=mol.succ_value,
                template=reaction.template,
                reactants=reactants,
                cost=reaction.cost
            )
        syn_route.succ = True
        return syn_route

    def init_and_or(self):
        tree_id = []
        self.mol_nodes = []  # 清空储存的节点
        self.reaction_nodes = []  # 清空储存的节点
        self.id2node = {}
        self.node2id = {}
        node_quene = Queue()
        node_quene.put(self.root)
        while not node_quene.empty():
            node = node_quene.get()
            tree_id.append(node.serialize())
            self.id2node.update({node.serialize(): node})
            self.node2id.update({node: node.serialize()})
            if hasattr(node, 'mol'):
                self.mol_nodes.append(node)
            else:
                self.reaction_nodes.append(node)
            for c in node.children:
                node_quene.put(c)
        # tree_id.sort()
        self.tree_id = ',  '.join(tree_id)

    def check_split_end(self):
        flag = True
        for rxn_node in self.reaction_nodes:
            if not rxn_node.parent.children == [rxn_node]:
                return False
        return flag

    def check_split_route_succ(self):
        flag = True
        leaf_mol_nodes = []
        for mol_node in self.mol_nodes:
            if len(mol_node.children) == 0:
                leaf_mol_nodes.append(mol_node)
                if not mol_node.succ:
                    flag = False
        return flag

    def extract_all_succ_routes(self):
        copy_route = remove_not_succ_node(self)
        all_routes = []
        all_succ_routes = []
        tree_id_list = []
        node_queue = Queue()
        node_queue.put((copy_route.root, None))
        while not node_queue.empty():
            node, split_route = node_queue.get()
            if node.parent is None:
                new_root = deepcopy(node)
                new_root.children = []
                split_route = MolTree(target_mol=self.target_mol,
                                      known_mols=set(),
                                      value_fn=lambda x: 0.)
                split_route.succ = self.succ
                split_route.max_depth = self.max_depth
                split_route.zero_known_value = self.zero_known_value
                split_route.search_status = self.search_status
                split_route.root = new_root

                split_route.mol_nodes = []
                split_route.reaction_nodes = []
                split_route.known_mols = set()

                split_route.init_and_or()
            if hasattr(node, 'mol'):
                this_new_tree = []
                for rxn_c in node.children:
                    if not rxn_c.succ:
                        continue
                    new_rxn_c, new_route = split_route_fn(rxn_c, split_route)
                    node_queue.put((new_rxn_c, new_route))
                    this_new_tree.append(new_route)
                    if new_route.check_split_end() and new_route.tree_id not in tree_id_list:
                        all_routes.append(new_route)
                        tree_id_list.append(new_route.tree_id)
                if node.parent:
                    for sbling in node.parent.children:
                        if sbling != node:
                            for sbling_rxn_c in sbling.children:
                                if not sbling_rxn_c.succ:
                                    continue
                                for new_route in this_new_tree:
                                    new_sbling_rxn_c, new_route = split_route_fn(
                                        sbling_rxn_c, new_route)
                                    node_queue.put(
                                        (new_sbling_rxn_c, new_route))
                                    if new_route.check_split_end() and new_route.tree_id not in tree_id_list:
                                        all_routes.append(new_route)
                                        tree_id_list.append(new_route.tree_id)
            else:
                for mol_c in node.children:
                    split_route.init_and_or()
                    node_queue.put((mol_c, split_route))
                    if split_route.check_split_end() and split_route.tree_id not in tree_id_list:
                        all_routes.append(split_route)
                        tree_id_list.append(split_route.tree_id)

        for route in all_routes:
            if route.check_split_route_succ():
                all_succ_routes.append(route.convert_succ_route())
        return all_succ_routes

    def viz_search_tree(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'pdf'

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                shape = 'rarrow'

            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = '%.3f' % node.cost
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()
