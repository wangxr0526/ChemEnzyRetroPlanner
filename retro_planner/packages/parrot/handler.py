# handler.py
import os
import pandas as pd
import torch
from ts.torch_handler.base_handler import BaseHandler
import yaml

from model import ParrotConditionPredictionModel, caonicalize_rxn_smiles, get_output_results, inference_load

class ParrotInferenceHandler(BaseHandler):
    """
    Custom handler for ParrotConditionPredictionModel
    """

    def initialize(self, ctx):
        """
        Initialize the model and any other resources.
        """
        self.manifest = ctx.manifest
        model_dir = ctx.system_properties.get("model_dir")
        self.model_work_path = os.path.abspath(model_dir)
        # parrot'model_work_path', self.model_work_path)
        # parrot'list_path', os.listdir(self.model_work_path))
        # parrot'list_model_dir', os.listdir(model_dir))
        # Load configuration
        config_path = os.path.join(self.model_work_path, 'config_inference_use_uspto.yaml')
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        model_args = config['model_args']
        model_args['use_multiprocessing'] = False
        model_args['best_model_dir'] = os.path.abspath(model_dir)
        model_args['output_dir'] = os.path.abspath(model_dir)
        model_args['pretrained_path'] = os.path.abspath(model_dir)

        dataset_args = config['dataset_args']
        dataset_args['dataset_root'] = os.path.abspath(self.model_work_path)

        inference_args = config['inference_args']
        try:
            model_args['use_temperature'] = dataset_args['use_temperature']
            print('Using Temperature:', model_args['use_temperature'])
        except KeyError:
            print('No temperature information is specified!')

        condition_label_mapping = inference_load(**dataset_args)
        model_args['decoder_args'].update({
            'tgt_vocab_size': len(condition_label_mapping[0]),
            'condition_label_mapping': condition_label_mapping
        })

        trained_path = model_args['best_model_dir']
        self.model = ParrotConditionPredictionModel(
            "bert",
            trained_path,
            args=model_args,
            use_cuda=True if torch.cuda.is_available() else False,
            cuda_device=0
        )
        self.config = config
        self.model_args = model_args
        self.dataset_args = dataset_args
        self.inference_args = inference_args

        # Initialize any other resources if needed

    def preprocess(self, data):
        """
        Preprocess the input data.
        Expects a list of dictionaries with 'body' containing the input SMILES strings.
        """
        input_rxn_smiles = []
        # parrotdata)
        for row in data:
            if 'body' in row:
                input_data = row['body']
                if isinstance(input_data, bytes):
                    input_data = input_data.decode('utf-8')
                input_rxn_smiles.extend(input_data)
        # Remove duplicates and empty strings
        input_rxn_smiles = list(set([x.strip() for x in input_rxn_smiles if x.strip()]))
        return input_rxn_smiles

    def inference(self, input_rxn_smiles):
        """
        Perform inference using the loaded model.
        """
        test_df = pd.DataFrame({
            'text': input_rxn_smiles,
            'labels': [[0] * 7] * len(input_rxn_smiles) if not self.model_args['use_temperature'] else [[0] * 8] * len(input_rxn_smiles)
        })
        print('Canonicalize reaction smiles and remove invalid reactions...')
        test_df['text'] = test_df.text.apply(caonicalize_rxn_smiles)
        test_df = test_df.loc[test_df['text'] != ''].reset_index(drop=True)
        beam = self.inference_args['beam']
        pred_conditions, pred_temperatures = self.model.condition_beam_search(
            test_df,
            output_dir=self.model_args['best_model_dir'],
            beam=beam,
            test_batch_size=8,
            calculate_topk_accuracy=False
        )
        output_results = get_output_results(
            test_df.text.tolist(),
            pred_conditions,
            pred_temperatures,
            output_dataframe=False
        )
        print('Inference done!')
        return output_results

    def postprocess(self, inference_output):
        """
        Postprocess the inference results to JSON format.
        """
        # Convert output_results to a list of dictionaries
        # parrot'inference_output type:', type(inference_output))
        # parrot'inference_output:', inference_output)
        response = []
        for result in inference_output:
            # parrot'result type:', type(result))
            response.append(result.to_json())
        return [response]


# if __name__ == '__main__':
    
#     class MockContext:
#         def __init__(self):
#             self.system_properties = {
#                 "model_dir": "mars/",
#                 "gpu_id": "0"  # 如果没有使用GPU，可以设置为 None
#             }
#             self.manifest = {
#                 "model": {
#                     "modelName": "USPTO_condition",
#                     "serializedFile": "outputs/Parrot_train_in_USPTO_Condition_enhance/pytorch_model.bin",
#                     "handler": "handler.py",
#                     "modelFile": "model.py",
#                     "modelVersion": "1.0"
#                 }
#             }

#         def get_model_dir(self):
#             return self.system_properties["model_dir"]

#         def get_system_properties(self):
#             return self.system_properties

#         def get_manifest(self):
#             return self.manifest

#     data = [{'body': 'CCCCC>>CCCCCC'}]
#     handler = ParrotInferenceHandler()
#     ctx = MockContext()  # 需要创建一个适合你环境的模拟Context对象
#     handler.initialize(ctx)
#     preprocessed_data = handler.preprocess(data)
#     inference_results = handler.inference(preprocessed_data)
#     response = handler.postprocess(inference_results)
#     print(response)