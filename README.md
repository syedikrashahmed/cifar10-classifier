# CIFAR-10 Image Classifier

A series of image classification models on CIFAR-10 dataset, moving from a simple Multi Layer Perceptron baseline through Convolutional Neural Networks to transfer learning with ResNet18, comparing
feature extraction against fine-tuning.

## Results

| Model                        | Test Accuracy | F1     | Training Time |
|-------------------------------|:-------------:|:------:|:--------------:|
| MLP Baseline (historical)     | 49.57%        | –      | –              |
| CNN Baseline                  | 72.06%        | 0.7240 | 19.75 min      |
| CNN Best (Augmentation + BN)  | 76.70%        | 0.7685 | 52.62 min      |
| ResNet18 – Feature Extraction | 80.65%        | 0.8071 | 7.68 h         |
| ResNet18 – Fine-Tuning        | 90.72%    | 0.9074 | 9.37 h     |

## Key Findings

- A basic CNN beat the MLP baseline by +22.5 points before any tuning.
- Augmentation and BatchNorm improved baseline CNN by +4.64 points, at 2.7x the training time.
- A pretrained ResNet18 with a frozen backbone outperformed both CNNs (80.65%), showing ImageNet features transfer well to CIFAR-10 with no adaptation.
- Fine-tuning gave the single largest jump (+10.07 points), by letting the final residual block adapt to CIFAR-10 specific features, but training time rose to 9.4 hours and the train/test gap widened (98.37% train vs 90.72% test).

## Notebook

[`CIFAR10_Complete_Experiments.ipynb`](./CIFAR10_Complete_Experiments.ipynb), the complete writeup: dataset/preprocessing, all four experiments with architecture, configuration, results, master comparison, and analysis.

Earlier learning notebooks (tensors, dataloaders, MLP, CNN) are in `notebooks/`.

## Reproduce

```
pip install -r requirements.txt

python train.py --experiment cnn_baseline
python train.py --experiment cnn_best
python train.py --experiment resnet_feature
python train.py --experiment resnet_finetune

python evaluate.py --experiment cnn_baseline
python visualize.py --experiment cnn_baseline
```

Trained checkpoints are not included in the repo due to it's size, run `train.py` to reproduce them. Metrics and figures from the original runs are already saved in `results/`.

## Limitations

- All experiments were trained on CPU, which constrained epoch counts for the ResNet configurations.
- Single random seed per experiment, so no statistical significance testing.
- No comparison against published CIFAR-10 SOTA methods, the goal was a controlled internal comparison of training strategies, not a new method.