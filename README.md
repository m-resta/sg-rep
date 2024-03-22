# Code for "Self-generated Replay Memories for Continual Neural Machine Translation"

## Requirements

- enchant spell-checker (https://abiword.github.io/enchant/)
- `pip install -r requirements.txt`

Optional: wandb
A note about dictionaries: pyenchant works with different providers: be sure to check the requirements to install additional dictionaries.
For example, for enchant, if the USER_CONFIG_DIR/PROVIDER_NAME/ is set, you can add Hunspell dictionaries (.dic and .aff files) to the directory and they will be automatically loaded.
Refer to this thread for more info: https://github.com/pyenchant/pyenchant/issues/167



## Setup
1. Create a data directory and a model directory:
    - `data directory`: Where the datasets will be saved.
    - `model directory`: Folder of the various models.

2. Make a copy of `env.example` and save it as `env`. In `env`, set the value of DATA_DIR as `data directory` and set the value of  MODEL_BASE_DIR as `model directory`.



## Experiments:

The first argument is the model subdirectory name, the second are the CUDA_VISIBLE_DEVICES

- Sequential finetuning: 
    - `./cill_noreplay.sh seq_finetune 0`
- Joint Training 
    - `./cill_joint_training.sh joint_training 0`
- Replay: 
    - `./cill_rp10.sh rp10_training 0`
- EWC: 
    - `./ewc_cill.sh ewc_training 0`
- AGEM: 
    - `./agem_cill.sh agem_training 0` 
- Self-Replay 
    - `./selfrep_cill_rp20.sh selfreplay_rp20 0`

Note: The scripts are set to use the iwslt2017 dataset. To use the unpc dataset, change the `--dataset_name` flag to `unpc` in the scripts and the language pairs accordingly.

## Changing default parameters

Below are some of the most common cmd line flags.

To use them run generic_entrypoint.sh specifying as third argument the strategy you want to use. Strategies are under src/strategies.

Example:

`./generic_entrypoint.sh model_directory 0 src.strategies.ewc_cill_unified --train_epochs 10 --dataset_name unpc`

will run the EWC training with specified arguments.



| Options               | Description   |
| -------------         | ------------- |
| model_save_path       | Where to save the models |
| dataset_save_path     | Folder of the datasets. If no data is present it will be downloaded |
| dataset_name          | Name of the dataset: `iwslt2017` or `unpc` |
| train_epochs          | Epochs for training for all tasks. |
| lang_pairs            | List of translation pairs for the various experiences e.g `en-fr en-ro` |
| replay_memory         | Number of samples for the memory |
| pairs_in_experience   | Number of translation pairs in each experience |
| metric_for_best_model | Evaluation metric to select the best model|
| batch_size            | Batch size for all tasks. |
| save_steps            | Saving interval steps|
| eval_steps            | Evaluate every steps|
| fp16                  | Use float16 precision |
| early_stopping        | Patience parameter |
| ewc_lambda            | Lambda parameter for EWC strategy |
| agem_sample_size      | Sise of the AGEM samples during optimization |
| agem_pattern_per_exp  | Number of examples to sample from to populate agem memory buffer |
| logging_dir           | Directory to store logs. |
| bidirectional | Defaults to `True`. If `True` it will include also the reverse direction e.g. with `--lang_pairs en-fr `it will train also on `fr-en` |

Check individual strategies for more options.

## Evaluation

Models are evaluated on validation set during training. At the end of the training phase the best model is evaluated on the test set.
During training BLEU score is used as evaluation metric as it is the most common metric for machine translation and cheaper than COMET.

To evaluate a model on the test set, use the `eval_models.py` script under `src/utils` folder and pass as cmdline arguments the same used to train the model. The script will evaluate the best model using BLEU and COMET.

## Docker
We provide a template Dockerfile to build an image with all the dependencies needed to run the experiments. Check the Dockerfile for more info.

## Citation
If find this code useful, please consider citing our work:

```
@misc{resta2024selfgenerated,
      title={Self-generated Replay Memories for Continual Neural Machine Translation}, 
      author={Michele Resta and Davide Bacciu},
      year={2024},
      eprint={2403.13130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
