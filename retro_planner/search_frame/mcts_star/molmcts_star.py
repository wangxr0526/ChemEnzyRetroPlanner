import os
import numpy as np
import logging
from tqdm import tqdm
import time
from retro_planner.search_frame.mcts_star.mol_tree import MolTree


def mol_planner(target_mol, target_mol_id, starting_mols, expand_fn,
                iterations, max_depth=10, viz=False, exclude_target=True, viz_dir=None, value_fn=lambda x: 0, keep_search=False):
    t0 = time.time()
    exclude_flag = False
    if exclude_target:
        if target_mol in starting_mols:
            exclude_flag = True
            print(f'Exclude {target_mol} from buyable set.')
            starting_mols.discard(target_mol)
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn,
        max_depth=max_depth
    )

    i = -1
    first_succ_time = np.inf
    if not mol_tree.succ:
        for i in tqdm(range(iterations)):
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open
            try:
                result = expand_fn(m_next.mol)
            except:
                result = None

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                if 'costs' in result:
                    costs = result['costs']
                else:
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

                assert m_next.open
                succ, _ = mol_tree.expand(
                    m_next, reactant_lists, costs, templates, max_depth=max_depth)

                if succ:
                    if not keep_search:
                        logging.info('Return first!')
                        logging.info('Synthesis route found!')
                        break
                    else:
                        if first_succ_time == np.inf:
                            first_succ_time = time.time() - t0
                        pass
                current_depth = m_next.depth
                rollout_depth = current_depth
                rollout_succ = False
                while not m_next.is_terminal() and not rollout_succ and (rollout_depth - current_depth < 6):
                    assert not m_next.open or not m_next.go_back
                    grandchild = mol_tree._select_promising_grandchild(
                        m_next)
                    assert grandchild
                    if grandchild.go_back:
                        m_next = grandchild
                        continue
                    succ, grandchild_succ = mol_tree.rollout(
                        grandchild, expand_fn=expand_fn, max_depth=max_depth)
                    rollout_depth = grandchild.depth
                    m_next = grandchild
                    if not keep_search:
                        rollout_succ = succ
                    else:
                        rollout_succ = grandchild_succ


                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    if not keep_search:
                        break
                    else:
                        pass

            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)
        if not keep_search:
            logging.info('Final search status | success value | iter: %s | %s | %d'
                         % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))
        else:
            logging.info('Final search status | iter: %s | %d'
                         % (str(mol_tree.search_status), i+1))

    best_route = None
    all_routes = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        all_routes = mol_tree.extract_all_succ_routes()
        assert best_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    if exclude_flag:
        starting_mols.add(target_mol)
    return mol_tree.succ, (best_route, i+1, all_routes, first_succ_time)
