import numpy as np
import logging


class MolNode:
    def __init__(self, mol, init_value, parent=None, is_known=False,
                 zero_known_value=True, max_depth=10):
        self.max_depth = max_depth
        self.mol = mol
        self.pred_value = init_value
        self.value = init_value
        self.succ_value = np.inf    # total cost for existing solution
        self.sibling = []
        self.parent = parent

        # ############ MCTS score###########
        # self.mcts_value = None
        # self.mcts_visitation = None
        # self.children_mcts_value = []
        # self.children_mcts_visitation = []
        # ##################################

        self.id = -1
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth

        self.is_known = is_known
        self.children = []
        self.succ = is_known
        self.open = True    # before expansion: True, after expansion: False
        # self.rolloutable = True
        self.go_back = False
        if is_known:
            self.open = False
            # self.rolloutable = False
            self.go_back = True
            if zero_known_value:
                self.value = 0
            self.succ_value = self.value

        if parent is not None:
            parent.children.append(self)

    def v_self(self):
        """
        :return: V_self(self | subtree)
        """
        return self.value

    def v_target(self):
        """
        :return: V_target(self | whole tree)
        """
        if self.parent is None:
            return self.value
        else:
            return self.parent.v_target()

    # def children_q(self):
    #     return np.array(self.children_mcts_value) / np.array(self.children_mcts_visitation)

    # def children_u(self):
    #     total_visits = np.log(np.sum(self.children_mcts_visitation))
    #     child_visits = np.array(self.children_mcts_visitation)
    #     return 1.4 * np.sqrt(2 * total_visits / child_visits)  ####### C = 1.4

    def init_values(self, no_child=False):
        assert self.open and (no_child or self.children)

        new_value = np.inf
        self.succ = False
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
            self.succ |= reaction.succ

        v_delta = new_value - self.value
        self.value = new_value

        if self.succ:
            for reaction in self.children:
                self.succ_value = np.min((self.succ_value,
                                          reaction.succ_value))

        # if not rollout:
        #     self.open = False
        # else:
        #     self.rolloutable = False
        self.open = False

        return v_delta

    def backup(self, succ):
        assert not self.is_known

        new_value = np.inf
        for reaction in self.children:
            new_value = np.min((new_value, reaction.v_self()))
        new_succ = self.succ | succ
        updated = (self.value != new_value) or (self.succ != new_succ)

        new_succ_value = np.inf
        if new_succ:
            for reaction in self.children:
                new_succ_value = np.min((new_succ_value, reaction.succ_value))
            updated = updated or (self.succ_value != new_succ_value)

        v_delta = new_value - self.value
        self.value = new_value
        self.succ = new_succ
        self.succ_value = new_succ_value

        if updated and self.parent:
            return self.parent.backup(v_delta, from_mol=self.mol)

    def serialize(self):
        text = '%d | %s' % (self.id, self.mol)
        # text = '%d | %s | pred %.2f | value %.2f | target %.2f' % \
        #        (self._id, self.mol, self.pred_value, self.v_self(),
        #         self.v_target())
        return text

    def get_ancestors(self):
        if self.parent is None:
            return {self.mol}

        ancestors = self.parent.parent.get_ancestors()
        ancestors.add(self.mol)
        return ancestors

    def is_terminal(self):
        # if not self.rolloutable:  # 不可扩展
        #     return True
        if self.depth > self.max_depth:  # 达到最大探索深度
            return True
        elif self.succ:  # 该分子节点已经在可买分子库中了，成功了
            return True
        elif self.go_back : # 
            return True
        elif not self.open and len(self.children)==0:
            return True
        else:
            return False

    def random_select_child(self):
        pass
