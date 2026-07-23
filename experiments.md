# Experiment Summary

| # | Experiment | Test Accuracy | Notes |
|---|------------|--------------:|------|
| 1 | MLP Baseline | 49.57% | Baseline model |
| 2 | CNN Baseline | 69.61% | Better features, some overfitting |
| 3 | CNN + Data Augmentation | 75.30% | Best so far, improved generalization |

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