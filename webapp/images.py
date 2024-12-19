from urllib.parse import quote
from rdkit import Chem
from rdkit.Chem import rdDepictor, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import py3Dmol

LABEL2ACTIVE_TYPE = {
    0: None,
    1: 'Binding Site',
    2: 'Catalytic Site',    # Active Site in UniProt
    3: 'Other Site'
}

def smitosvg_url(smi, molSize=(150,150), kekulize=True):
    try:
        mol = Chem.MolFromSmiles(smi)
        url = moltosvg_url(mol, molSize=molSize, kekulize=kekulize)
        return url
    except:
        return ''

def moltosvg_url(mol, molSize=(150,150), kekulize=True):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
    opts = drawer.drawOptions()
    
    # 设置背景颜色为透明
    opts.backgroundColour = (255, 255, 255, 0)  # RGBA格式，A为0代表透明

    opts.addStereoAnnotation = True
    
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    # 清除SVG的XML标头中可能存在的白色背景设置
    svg = svg.replace('white', 'none')
    
    url = "data:image/svg+xml;charset=utf-8," + quote(svg)
    return url


def reactiontosvg_url(reaction_smiles, molSize=(300, 150)):
    try:
        # 将反应 SMILES 转换为反应对象
        rxn = AllChem.ReactionFromSmarts(reaction_smiles, useSmiles=True)
        
        # 生成所有反应物和生成物的2D坐标
        for mol in rxn.GetReactants():
            if mol:
                rdDepictor.Compute2DCoords(mol)
        for mol in rxn.GetProducts():
            if mol:
                rdDepictor.Compute2DCoords(mol)
        
        # 使用 RDKit 的绘图工具绘制反应
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
        opts = drawer.drawOptions()
        opts.backgroundColour = (255, 255, 255, 0)  # 设置背景为透明
        
        drawer.DrawReaction(rxn, highlightByReactant=True)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        
        # 清除 SVG 的 XML 标头中可能存在的白色背景设置
        svg = svg.replace('white', 'none')
        
        url = "data:image/svg+xml;charset=utf-8," + quote(svg)
        return url
    except Exception as e:
        print("Failed to generate SVG for reaction:", str(e))
        return ''
    
def get_structure_html_and_active_data(
                   enzyme_structure_path,
                   site_labels=None, 
                   view_size=(900, 900), 
                   res_colors={
                    0: '#73B1FF',   # 非活性位点
                    1: '#FF0000',     # Binding Site
                    2: '#00B050',     # Active Site
                    3: '#FFFF00',     # Other Site
                    },
                   show_active=True,
                   debug=False,
):
    with open(enzyme_structure_path) as ifile:
        system = ''.join([x for x in ifile])
    
    view = py3Dmol.view(width=view_size[0], height=view_size[1])
    view.addModelsAsFrames(system)
    
    active_data = []
    
    if show_active and (site_labels is not None) and not debug:
        i = 0
        res_idx = None
        for line in system.split("\n"):
            split = line.split()
            if len(split) == 0 or split[0] != "ATOM":
                continue
            if res_idx is None:
                first_res_idx = int(line[22:26].strip())
            res_idx = int(line[22:26].strip()) - first_res_idx
            color = res_colors[site_labels[res_idx]]
            view.setStyle({'model': -1, 'serial': i+1}, {"cartoon": {'color': color}})
            atom_name = line[12:16].strip()
            if (atom_name == 'CA') and (site_labels[res_idx] !=0) :
                residue_name = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                view.addLabel(f'{residue_name} {res_idx + 1}', {"fontSize": 8, "position": {"x": x, "y": y, "z": z}, "fontColor": color, "fontOpacity":1.0 ,"backgroundColor": 'white', "bold":True,"backgroundOpacity": 0.1})
                active_data.append((res_idx + 1, residue_name, color, LABEL2ACTIVE_TYPE[site_labels[res_idx]])) # 设置label从1开始#
            
            i += 1
    else:
        view.setStyle({'model': -1}, {"cartoon": {'color': res_colors[0]}})
        # view.addSurface(py3Dmol.SAS, {'opacity': 0.5})
    view.zoomTo()
    # view.show()
    view.zoom(1.6, 600)
    return view.write_html(), active_data
