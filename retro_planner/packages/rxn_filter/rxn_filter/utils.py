from rdkit import Chem
from rdkit.Chem import AllChem


def canonicalize_smiles(smi, clear_map=False):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        if clear_map:
            [atom.ClearProp('molAtomMapNumber') for atom in mol.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        return Chem.MolToSmiles(mol)
    else:
        return ''


class RunRdRxn:
    def __init__(self, org_rxn_smiles, retro_tpl_smarts):
        self.smiles, self.ref_prod_smiles = org_rxn_smiles.split('>>')
        self.retro_tpl_smarts = retro_tpl_smarts
        self.forward_tpl = self.get_forward_tpl()
        # self.outcomes = self.apply_reaction()
        self.output = None

    def apply_reaction(self):
        mols = [Chem.MolFromSmiles(x) for x in self.smiles.split('.')]
        rd_reaction = AllChem.ReactionFromSmarts(self.forward_tpl)
        num_rectantant_templates = rd_reaction.GetNumReactantTemplates()
        if len(mols) < num_rectantant_templates:
            return []
        self.reactants = mols[:num_rectantant_templates]
        outcomes = []
        try:
            products_list = rd_reaction.RunReactants(self.reactants)
            for products in products_list:
                outcomes.append(products)
        except Exception as e:
            print(e)
        return outcomes

    def get_results(self):
        if not self.output:
            self.output = set()
            self.outcomes = self.apply_reaction()
            if self.outcomes:
                for products in self.outcomes:
                    prod_smiles = '.'.join(Chem.MolToSmiles(mol) for mol in products)
                    can_prod_smiles = canonicalize_smiles(prod_smiles)
                    if can_prod_smiles == '': continue
                    if can_prod_smiles == self.ref_prod_smiles: continue
                    react_smiles = '.'.join(Chem.MolToSmiles(mol) for mol in self.reactants)
                    rxn_smiles = f'{react_smiles}>>{prod_smiles}'
                    self.output.add(rxn_smiles)
                return self.output
                # self.output = set(self.output)
            else:
                self.output = None
                return self.output
        else:
            return self.output

    def get_forward_tpl(self):
        prod_smarts, react_smarts = self.retro_tpl_smarts.split('>>')
        return f'{react_smarts}>>{prod_smarts}'


if __name__ == '__main__':
    rxn_smarts = '([NH2;D1;+0:7]-[c;H0;D3;+0:1]1:[c:2]:[c:3]:[#7;a:4]:[c:5]:[c:6]:1)>>Cl-[c;H0;D3;+0:1]1:[c:2]:[c:3]:[#7;a:4]:[c:5]:[c:6]:1.[NH3;D0;+0:7]'
    rxn_smiles = 'FC(F)(F)c1cc(Cl)cc(C(F)(F)F)n1.N>>Nc1cc(C(F)(F)F)nc(C(F)(F)F)c1'
    run_rxn = RunRdRxn(rxn_smiles, rxn_smarts)
    results = run_rxn.get_results()
    print(results)
