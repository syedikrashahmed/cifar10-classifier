# Experiment Summary

| # | Experiment | Test Accuracy | Notes |
|---|------------|--------------:|------|
| 1 | MLP Baseline | 49.57% | Baseline model |
| 2 | CNN Baseline | 69.61% | Better features, some overfitting |
| 3 | CNN + Data Augmentation | 75.30% | Best so far, improved generalization |
| 4 | CNN + Data Augmentation + Dropout | 71.97% | Introduced excessive regularization, making learning more difficult |
| 5 | CNN + Data Augmentation + Weight Decay | 72.45% | Slight improvement over Dropout, but below the best model |
| 6 | CNN + Data Augmentation + Batch Normalization | 74.08% | Improved stability and generalization, second-best |

# Experiment 1 — MLP Baseline

## Model
MLP

## Structure
MLP ( <br>
  &emsp; (fc1): Linear(in_features=3072, out_features=512, bias=True) <br>
  &emsp; (fc2): Linear(in_features=512, out_features=256, bias=True) <br>
  &emsp; (fc3): Linear(in_features=256, out_features=10, bias=True) <br>
  &emsp; (relu): ReLU() <br>
)

## Changes
Initial baseline model.

## Results
- Loss: 1.3324
- Train Accuracy: 52.31%
- Test Accuracy: 49.57%

## Observations
- Model learns slowly.
- Confusion matrix shows many random mistakes.
- Struggles with animal classes.

## Conclusion
Baseline established for future comparison.

---

# Experiment 2 — CNN Baseline

## Model
CNN

## Structure
CNN ( <br>
  &emsp; (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) <br>
  &emsp; (relu): ReLU() <br>
  &emsp; (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) <br>
  &emsp; (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) <br>
  &emsp; (flatten): Flatten(start_dim=1, end_dim=-1) <br>
  &emsp; (fc1): Linear(in_features=4096, out_features=512, bias=True) <br>
  &emsp; (fc2): Linear(in_features=512, out_features=10, bias=True) <br>
)

## Changes
Replaced MLP with a 2-layer CNN.

## Results
- Loss: 0.1267
- Train Accuracy: 95.87%
- Test Accuracy: 69.61%

## Observations
- Much lower training loss.
- Better feature learning.
- Confusion matrix is cleaner.
- Some overfitting.
- Main confusion: cat/dog and deer/horse.

## Conclusion
CNN significantly outperforms MLP but overfits.

---

# Experiment 3 — CNN + Data Augmentation

## Model
CNN + Data Augmentation

## Structure
same as CNN Baseline (Experiment 2)

## Changes
Added training data augmentation:
- RandomCrop(32, padding=4)
- RandomHorizontalFlip()

No changes were made to the CNN architecture, optimizer, learning rate, batch size, or number of epochs.

## Results
- Loss: 0.7760
- Train Accuracy: 72.82%
- Test Accuracy: 75.30%

## Observations
- Training loss decreases smoothly and converges.
- Test accuracy improves by approximately 5.7 percentage points compared to the baseline CNN.
- Training accuracy is lower because data augmentation makes training images more challenging.
- Test accuracy is slightly higher than training accuracy, indicating strong generalization.
- Confusion matrix is cleaner, with most errors limited to visually similar animal classes.
- Vehicle classes are classified consistently with very few mistakes.

## Conclusion
Data augmentation significantly improves generalization without changing the model architecture. Although training becomes more difficult, the model learns more robust visual features and achieves the best test accuracy so far while greatly reducing overfitting.

---

# Experiment 4 — CNN + Dropout

## Model
CNN + Data Augmentation + Dropout

## Structure
CNN ( <br>
  &emsp; (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) <br>
  &emsp; (relu): ReLU() <br>
  &emsp; (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False) <br>
  &emsp; (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) <br>
  &emsp; (flatten): Flatten(start_dim=1, end_dim=-1) <br>
  &emsp; (fc1): Linear(in_features=4096, out_features=512, bias=True) <br>
  &emsp; (dropout): Dropout(p=0.5, inplace=False) <br>
  &emsp; (fc2): Linear(in_features=512, out_features=10, bias=True) <br>
)

## Changes
Added a Dropout Layer (p=0.5) after the first fully connected layer.

No other changes were made.

## Results
- Loss: 0.9583
- Train Accuracy: 66.53%
- Test Accuracy: 71.97%

## Observations
- Training loss decreased steadily but converged to a higher final loss than the previous experiment.
- Training and test accuracy increased consistently throughout training, indicating stable learning.
- No signs of overfitting were observed, as the training and test accuracy remained close.
- Test accuracy decreased compared to the previous experiment (75.30% → 71.97%).
- The confusion matrix remained largely diagonal, with most errors occurring between visually similar animal classes such as birds, cats, dogs, and deer.
- Vehicle classes continued to be classified accurately with relatively few mistakes.

## Conclusion
Adding Dropout(0.5) increased regularization but reduced overall performance. Although the model continued to generalize well and avoided overfitting, both training and test accuracy decreased compared to the previous experiment. For this relatively small CNN, the combination of data augmentation and Dropout(0.5) introduced excessive regularization, making learning more difficult without improving generalization. Data augmentation alone remained the better-performing approach.

---

# Experiment 5 — CNN + Weight Decay

## Model
CNN + Data Augmentation + Weight Decay

## Structure
same as CNN Baseline (Experiment 2)

## Changes
Added Weight Decay (L2 Regularization) to the Adam optimizer using: weight_decay = 1e-4

The CNN architecture and data augmentation pipeline remained unchanged.

## Results
- Loss: 0.8198
- Train Accuracy: 71.17%
- Test Accuracy: 72.45%

## Observations
- Training loss decreased smoothly and converged to a lower value than the previous Dropout experiment.
- Training and test accuracy improved compared to the Dropout model.
- The train–test accuracy gap remained small, indicating good generalization.
- Confusion matrix continued to show strong performance on vehicle classes, while most remaining errors occurred between visually similar animal classes.
- Despite improved optimization, test accuracy remained below the best-performing CNN with data augmentation alone.

## Conclusion
Adding Weight Decay improved optimization and slightly increased both training and test accuracy compared with the Dropout experiment. However, it did not outperform the CNN trained with data augmentation alone. This suggests that, for this architecture and dataset, data augmentation provided the most effective regularization, while Weight Decay offered only a modest additional benefit.

---

# Experiment 6 — CNN + Batch Normalization

## Model
CNN + Data Augmentation + Batch Normalization

## Structure
CNN ( <br>
  &emsp; (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) <br>
  &emsp; (bn1): BatchNorm2d(32) <br>
  &emsp; (relu): ReLU() <br>
  &emsp; (pool): MaxPool2d(kernel_size=2, stride=2) <br>
  &emsp; (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) <br>
  &emsp; (bn2): BatchNorm2d(64) <br>
  &emsp; (flatten): Flatten(start_dim=1, end_dim=-1) <br>
  &emsp; (fc1): Linear(in_features=4096, out_features=512, bias=True) <br>
  &emsp; (fc2): Linear(in_features=512, out_features=10, bias=True) <br>
)

## Changes
Added Batch Normalization after each convolution layer.

Data augmentation and all training hyperparameters remained unchanged.

## Results
- Loss: 0.8119
- Train Accuracy: 71.44%
- Test Accuracy: 74.08%

## Observations
- Training loss decreased smoothly, indicating stable optimization.
- Training and test accuracy improved consistently throughout training.
- Test accuracy increased compared with the Weight Decay experiment.
- The confusion matrix showed strong performance on vehicle classes, while most remaining errors occurred between visually similar animal classes.
- Although Batch Normalization improved convergence and generalization, it did not outperform the CNN trained with data augmentation alone.

## Conclusion
Adding Batch Normalization improved training stability and produced better generalization than the Weight Decay experiment. However, the test accuracy remained below the best-performing model using data augmentation alone. Batch Normalization proved to be an effective architectural improvement, but for this CNN, data augmentation continued to provide the largest overall performance gain.