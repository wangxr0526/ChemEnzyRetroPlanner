from value_function.value_data_loader import ValueDataset, ValueDataLoader
val_data_loader = ValueDataLoader('../value_data_dic_sample_convert.pkl', batch_size=128)
val_data = val_data_loader.dataset
print(val_data.reaction_costs.shape)
