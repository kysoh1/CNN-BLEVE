# CNN-Bleve

Environment to train and evaluate the effectiveness of artificial neural networks to accurately predict peak pressure of BLEVE based off tabular data.
The idea is to first convert tabular data into images (pre-existing IGTD and REFINED methods) then determine the limitations of the model based off feature quantity (https://github.com/kysoh1/Feature-Generator) and network depth.

# How to use
Enter current training set in directory "../dataset_run" as a zip file.
Do not alter "output.txt", it contains corresponding truth values for final model evaluation.
Configure "../scripts/config.py" to set hyperparameters for the training.
You can enable wandb (https://wandb.ai/site) to track visualised training progress if preferred, logs are also stored in "../running_logs" and current models in "../saved_models".

# Run
python train.py 

### Notes
Training can take extremely long depending on the size of your dataset, training configurations, model type and your hardware. To give some perspective, a dataset of 46000 images trained on ResNet-50 architecture took about 12 hours using the best GPUs provided on Google Colab...
