import pandas as pd
from rdchiral.main import rdchiralRunText
from tqdm import tqdm

if __name__ == '__main__':
    prod_templates = pd.read_csv('../../../single_step_datasets/train_all_dataset/templates.dat', sep='\t', header=None)
    templates = prod_templates[0].tolist()
    prods = prod_templates[1].tolist()
    react_list = []
    for template, prod in tqdm(list(zip(templates, prods))[:5000]):
        try:
            react = rdchiralRunText(template, prod)
            react_list.append(react)
        except:
            react_list.append('')