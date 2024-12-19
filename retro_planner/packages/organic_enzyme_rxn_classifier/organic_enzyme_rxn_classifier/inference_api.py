from collections import defaultdict
from itertools import islice
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .train_organic_enzyme_rxn_classifier import Classifier, load_model
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, FingerprintGenerator
)

def generate_fingerprints(rxns: List[str], fingerprint_generator:FingerprintGenerator, batch_size=1, show_progress_bar=True) -> np.array:
    fps = []

    n_batches = len(rxns) // batch_size
    if len(rxns) % batch_size != 0:
        n_batches += 1
    emb_iter = iter(rxns)
          
    
    for i in tqdm(range(n_batches), disable=not show_progress_bar):
        batch = list(islice(emb_iter, batch_size))

        fps_batch = fingerprint_generator.convert_batch(batch)

        fps += fps_batch
    return np.array(fps)

class RXNClassificationInferenceDataset(Dataset):
    def __init__(self, rxnfps) -> None:
        super(RXNClassificationInferenceDataset, self).__init__()
        self.rxnfps = rxnfps
        self.fp_dim = rxnfps.size(1)


    def __len__(self):
        return len(self.rxnfps)

    def __getitem__(self, index):
        return self.rxnfps[index]



class OrganicEnzymeRXNClassifier:
    def __init__(self, checkpoint_path, device) -> None:
        model_state, config = load_model(checkpoint_path)
        classifier = Classifier(fp_dim=config['fp_dim'], h_dim=config['h_dim'], out_dim=config['out_dim'])
        classifier.load_state_dict(model_state)
        classifier = classifier.to(device)
        classifier.eval()
        self.classifier = classifier
        self.config = config
        self.device = device
        
        model, tokenizer = get_default_model_and_tokenizer('bert_ft_10k_25s')
        self.rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
        
        self.labels2name = {
            0:'Organic Reaction',
            1:'Enzymatic Reaction',
            # 1:'Enzyme catalyzed Reaction'
            }
    
    
    def _generate_rxnfps(self, rxns:List, batch_size:int):
        rxnfps = generate_fingerprints(rxns, batch_size=batch_size, fingerprint_generator=self.rxnfp_generator, show_progress_bar=False)
        return rxnfps

    def predict(self, rxns, batch_size:int=32):
        rxnfps = torch.tensor(self._generate_rxnfps(rxns=rxns, batch_size=batch_size), device=self.device).float()
        
        inference_dataset = RXNClassificationInferenceDataset(rxnfps=rxnfps)
        inference_dataloader = DataLoader(inference_dataset, shuffle=False, batch_size=batch_size)
        
        labels_collect = []
        confidence_collect = []
        names_collect = []
        with torch.no_grad():
            for batch in inference_dataloader:
                logic = self.classifier(batch)
                probs = torch.softmax(logic, dim=1)
                confidence, labels = torch.topk(probs, k=1)
                confidence = confidence.view(-1)
                labels = labels.view(-1)
                
                names = [self.labels2name[x] for x in labels.tolist()]
                labels_collect.extend(labels.tolist())
                confidence_collect.extend(confidence.tolist())
                names_collect.extend(names)
        return labels_collect, confidence_collect, names_collect
    
    
class EnzymeRXNClassifier(OrganicEnzymeRXNClassifier):
    def __init__(self, checkpoint_path, device) -> None:
        super().__init__(checkpoint_path, device)
        ecs2labels = self.config['ecs2labels']
        self.labels2name = {v:k for k,v in ecs2labels.items()}
    
    def predict(self, rxns, batch_size:int=32, topk:int=5):
        rxnfps = torch.tensor(self._generate_rxnfps(rxns=rxns, batch_size=batch_size), device=self.device).float()
        
        inference_dataset = RXNClassificationInferenceDataset(rxnfps=rxnfps)
        inference_dataloader = DataLoader(inference_dataset, shuffle=False, batch_size=batch_size)
        
        labels_collect = []
        confidence_collect = []
        names_collect = []
        with torch.no_grad():
            for batch in inference_dataloader:
                logic = self.classifier(batch)
                probs = torch.softmax(logic, dim=1)
                confidence, labels = torch.topk(probs, k=topk)

                
                names = [[self.labels2name[x] for x in label] for label in labels.tolist()]
                labels_collect.extend(labels.tolist())
                confidence_collect.extend(confidence.tolist())
                names_collect.extend(names)
        return labels_collect, confidence_collect, names_collect
    
class RXNClassifier(OrganicEnzymeRXNClassifier):
    def __init__(self, organic_enzyme_checkpoint_path, enzyme_checkpoint_path, device) -> None:
        self.device = device
        
        self.organic_enzyme_rxn_classifier, self.organic_enzyme_config = self._load_classifier(checkpoint_path=organic_enzyme_checkpoint_path, device=self.device)
        self.enzyme_rxn_classifier, self.enzyme_config = self._load_classifier(checkpoint_path=enzyme_checkpoint_path, device=self.device)

        
        model, tokenizer = get_default_model_and_tokenizer('bert_ft_10k_25s')
        self.rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
        
        self.labels2rxntype = {0:'Organic Reaction', 1:'Enzymatic Reaction'}
        ecs2labels = self.enzyme_config['ecs2labels']
        self.labels2ecs = {v:k for k,v in ecs2labels.items()}
        
    def _load_classifier(self, checkpoint_path, device):
        model_state, config = load_model(checkpoint_path)
        classifier = Classifier(fp_dim=config['fp_dim'], h_dim=config['h_dim'], out_dim=config['out_dim'])
        classifier.load_state_dict(model_state)
        classifier = classifier.to(device)
        classifier.eval()
        return classifier, config
    
    def predict(self, rxns, batch_size:int=32, enzyme_assign_topk:int=5, tasks=['organic_enzyme_rxn_classification', 'enzyme_assign']):
        rxnfps = torch.tensor(self._generate_rxnfps(rxns=rxns, batch_size=batch_size), device=self.device).float()
        
        inference_dataset = RXNClassificationInferenceDataset(rxnfps=rxnfps)
        inference_dataloader = DataLoader(inference_dataset, shuffle=False, batch_size=batch_size)
        
        organic_enzyme_rxn_classification_results = defaultdict(list)
        enzyme_rxn_classification_results = defaultdict(list)
        with torch.no_grad():
            for batch in inference_dataloader:
                
                for task in tasks:
                    if task == 'organic_enzyme_rxn_classification':
                        organic_enzyme_rxn_logic = self.organic_enzyme_rxn_classifier(batch)
                        organic_enzyme_rxn_probs = torch.softmax(organic_enzyme_rxn_logic, dim=1)
                        organic_enzyme_rxn_confidence, organic_enzyme_rxn_labels = torch.topk(organic_enzyme_rxn_probs, k=1)
                        organic_enzyme_rxn_confidence = enzyme_rxn_confidence.view(-1)
                        organic_enzyme_rxn_labels = enzyme_rxn_labels.view(-1)
                        
                        organic_enzyme_rxn_names = [self.labels2rxntype[x] for x in enzyme_rxn_labels.tolist()]
                        organic_enzyme_rxn_classification_results['labels'].extend(organic_enzyme_rxn_labels.tolist())
                        organic_enzyme_rxn_classification_results['names'].extend(organic_enzyme_rxn_names.tolist())
                        organic_enzyme_rxn_classification_results['confidence'].extend(organic_enzyme_rxn_confidence.tolist())
                    elif task == 'enzyme_assign':
                        enzyme_rxn_logic = self.classifier(batch)
                        enzyme_rxn_probs = torch.softmax(enzyme_rxn_logic, dim=1)
                        enzyme_rxn_confidence, enzyme_rxn_labels = torch.topk(enzyme_rxn_probs, k=enzyme_assign_topk)

                        enzyme_rxn_names = [[self.labels2name[x] for x in label] for label in enzyme_rxn_labels.tolist()]
                        enzyme_rxn_classification_results['lables'].extend(enzyme_rxn_labels.tolist())
                        enzyme_rxn_classification_results['names'].extend(enzyme_rxn_names.tolist())
                        enzyme_rxn_classification_results['confidence'].extend(enzyme_rxn_confidence.tolist())

        return organic_enzyme_rxn_classification_results, enzyme_rxn_classification_results


if __name__ == '__main__':
    
    # checkpoint_path = './checkpoints/unbalance_organic_enzyme_classification'
    # device = torch.device('cuda')
    # organic_enzyme_reaction_classifier = OrganicEnzymeRXNClassifier(checkpoint_path=checkpoint_path, device=device)
    
    # rxns = [
    #     'CC/C=C\CCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@@H](O)CO)OC(=O)CCCCCCCCC/C=C\CCCCCC.CCCCCC/C=C\CCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2ccc(N)nc2=O)C(O)[C@H]1O)OC(=O)CCCCCCCCC/C=C\CCCCCC>>CC/C=C\CCCCCCCCCC(=O)OC[C@H](COP(=O)(O)OC[C@H](O)COP(=O)(O)OC[C@@H](COC(=O)CCCCCCCCC/C=C\CCCCCC)OC(=O)CCCCCCCCC/C=C\CCCCCC)OC(=O)CCCCCCCCC/C=C\CCCCCC',
    #     'C1COCCO1.CO.COc1ccc(Cn2cc3c(n2)c(Cl)nc2ccc(OC)cc23)cc1.COc1cccc(N)c1.Cl>>COc1cccc(Nc2nc3ccc(OC)cc3c3c[nH]nc23)c1',
    # ]
    
    # labels, confidence, names = organic_enzyme_reaction_classifier.predict(rxns, batch_size=32)
    
    # for rxn, label, name, conf in zip(rxns, labels, names, confidence):
    #     print(f'rxn: {rxn}')
    #     print(f'label: {label}')
    #     print(f'name: {name}')
    #     print(f'confidence: {conf}')
    #     print()
    #     print('#'*20)
        
        
        
    checkpoint_path = './checkpoints/ecreact_classification'
    device = torch.device('cuda')
    enzyme_reaction_classifier = EnzymeRXNClassifier(checkpoint_path=checkpoint_path, device=device)
    
    rxns = [
        'CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)C(O)[C@H]1OP(=O)(O)O.CCCCCCCCCCCCCCCCCCCCCCCC(=O)O[C@@H](CO)COC(=O)CCCCCCCCCCCCCC>>CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCCCCCCCCCC',   # 2.3.1.76
        'CCCCC/C=C\C/C=C\C/C=C\C/C=C\C/C=C\CCC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O.CCCCCCCCCCCCCCCCCCCCCCCC(=O)OC[C@H](CO)OC(=O)CCCCCCCCCCCCCCC>>CCCCC/C=C\C/C=C\C/C=C\C/C=C\C/C=C\CCC(=O)OC[C@H](COC(=O)CCCCCCCCCCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC',   # 2.3.1.76
        'CC1OC(=O)C(C)C1C.O>>CC(O)CCC(=O)[O-]', # 3.1.1.25
    ]
    
    labels, confidence, names = enzyme_reaction_classifier.predict(rxns, batch_size=32)
    
    for rxn, label, name, conf in zip(rxns, labels, names, confidence):
        print(f'rxn: {rxn}')
        print(f'label: {label}')
        print(f'name: {name}')
        print(f'confidence: {conf}')
        print()
        print('#'*20)
    
    
