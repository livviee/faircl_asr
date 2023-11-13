## Hparams
### Training Settings

1. multi-lingual model : train with `1_train_en+de.yaml`
2. fine-tuned model : first train en model with `2_train_en_first.yaml` and then fine-tune with de using `3_train_de_second.yaml`
3. naive memory replay model : first train en model with `2_train_en_first.yaml` and then train it with de + 1_percent_of_en using `4_train_de_second_with_naive_mem_replay.yaml`

### What you should care / modify
- You should change `!PLACEHOLDER` in regard to your own setting.
  - Set `output_folder` and `data_folder` to of your own corresponding directory.
- Preprocessed `train.csv`, `test.csv`, `dev.csv` should be in your `save_folder` before start training in order to avoid data_preprocessing
