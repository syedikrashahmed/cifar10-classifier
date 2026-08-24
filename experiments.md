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

---

# Experiment 7 — Hyperparameter Tuning (Learning Rate)

| Learning Rate | Train Loss | Train Acc | Test Acc | Verdict |
| ------------: | ---------: | ---------: | ------: | ------- |
| 0.0001| 0.8650 | 69.58% | 71.06% | Stable but slow |
| 0.001 | 0.8846 | 68.83% | 71.89% | Best balance |
| 0.01 | 1.2644 | 52.53% | 55.82% | Too high|

---

# Experiment 7A — Learning Rate (0.0001)

## Model
CNN + Data Augmentation + Batch Normalization

## Structure
same as CNN + Batch Normalization (Experiment 6) 

## Changes
Reduced the learning rate to 0.0001.

All other training settings remained unchanged.

## Results
- Loss: 0.8650
- Train Accuracy: 69.58%
- Test Accuracy: 71.06%
- Best Test Accuracy: 72.16% (Epoch 9)

## Observations
- Training loss decreased smoothly, indicating stable optimization.
- Training accuracy improved slightly compared to the baseline.
- The highest test accuracy occurred at Epoch 9 before decreasing slightly.
- The lower learning rate produced smoother convergence but slower learning.

## Conclusion
Reducing the learning rate to 0.0001 improved optimization stability and achieved a slightly higher peak test accuracy. However, because learning was slower, the final epoch accuracy remained below the baseline. This learning rate may perform better if combined with additional training epochs or early stopping.

---

# Experiment 7B — Learning Rate (0.01)

## Model
CNN + Data Augmentation + Batch Normalization

## Structure
same as CNN + Batch Normalization (Experiment 6) 

## Changes
Increased the learning rate to 0.01.

All other training settings remained unchanged.

## Results
- Loss: 1.2644
- Train Accuracy: 52.53%
- Test Accuracy: 55.82%

## Observations
- Training loss remained much higher than the baseline.
- Both training and test accuracy were significantly lower.
- The model learned throughout training but converged to a poor solution.
- The confusion matrix showed increased misclassification, particularly among visually similar classes.
- The high learning rate caused unstable optimization and prevented the network from learning effective feature representations.

## Conclusion
Increasing the learning rate to 0.01 significantly reduced the model's performance. The optimizer took excessively large parameter updates, leading to poor convergence and lower training and test accuracy. For this CNN architecture, a learning rate of 0.01 is too large and is not recommended.

---

## Conclusion of Experiment 7
- 0.0001: Stable but requires longer training or early stopping.
- 0.001: Best overall balance for your setup.
- 0.01: Too aggressive and leads to underfitting.

---

# Experiment 8 — Hyperparameter Tuning (Optimizer) 
| Optimizer | Train Acc | Test Acc |
| ------------: | ---------: | ---------: | 
| Adam| 69.58% | 71.89% | 
| SGD | 69.83% | 70.26% |  

---

# Experiment 8A — Optimizer (SGD + Momentum)

## Model
CNN + Data Augmentation + Batch Normalization

## Changes
Replaced the Adam optimizer with SGD using momentum (0.9).

All other training settings remained unchanged.

## Results
- Loss: 0.8617
- Train Accuracy: 69.83%
- Test Accuracy: 70.26%

## Observations
- Training loss decreased smoothly throughout training.
- Training accuracy was slightly higher than with Adam.
- Test accuracy decreased by approximately 1.6% compared with the Adam baseline.
- The train-test accuracy gap remained very small, indicating good generalization.
- SGD required more gradual learning and showed larger fluctuations in test accuracy.

## Conclusion
Replacing Adam with SGD resulted in slightly better training performance but lower test accuracy. Although SGD optimized the training data more effectively, Adam produced better generalization within the 10-epoch training setup. Therefore, Adam remained the preferred optimizer for this project.

---

# Experiment 9 — Hyperparameter Tuning (Batch Size)
| Batch Size | Train Loss | Train Accuracy | Test Accuracy | Assessment |
| ------------: | -----------:|---------------:|--------------:|------------|
| 32| 0.7986 | 72.00% | 73.86% | Best overall balance between optimization and generalization |
| 64 | 0.8846 | 68.83% | 71.89% | Strong baseline performance |
| 128 | 0.7834 | 72.40% | 71.57% | Best training performance but weaker generalization |

---

# Experiment 9A — Batch Size (32)

## Model
CNN + Data Augmentation + Batch Normalization

## Structure
Same architecture as Experiment 6.

## Changes
Reduced the batch size from 64 to 32.

All other training settings remained unchanged.

## Results
- Loss: 0.7986
- Train Accuracy: 72.00%
- Test Accuracy: 73.86%

## Observations
- Achieved the lowest training loss among the batch size experiments.
- Training accuracy increased compared to the baseline.
- Test accuracy improved by nearly 2% over the baseline.
- Training remained stable with smooth convergence.
- The highest test accuracy was reached around Epoch 9, suggesting early stopping could further improve performance.
- The small train-test accuracy gap indicates excellent generalization with no significant overfitting.

## Conclusion
Reducing the batch size to 32 significantly improved both optimization and generalization. More frequent weight updates allowed the model to learn more transferable feature representations, resulting in the highest test accuracy among all hyperparameter tuning experiments. For this CNN architecture, a batch size of 32 provides the best overall balance between learning efficiency and generalization.

---

# Experiment 9B — Batch Size (128)

## Model
CNN + Data Augmentation + Batch Normalization

## Structure
Same architecture as Experiment 6.

## Changes
Increased the batch size from 64 to 128.

All other training settings remained unchanged.

## Results
- Loss: 0.7834
- Train Accuracy: 72.40%
- Test Accuracy: 71.57%

## Observations
- Achieved the lowest training loss and highest training accuracy among the batch size experiments.
- Training converged smoothly with stable optimization.
- Test accuracy was slightly lower than the baseline and noticeably lower than the batch size 32 experiment.
- The train-test accuracy gap remained small, indicating good generalization.
- Better optimization on the training data did not translate into improved performance on unseen images.

## Conclusion
Increasing the batch size to 128 improved optimization of the training data but slightly reduced generalization performance. Although the model achieved the lowest training loss and highest training accuracy, the lower test accuracy suggests that larger batches produced less transferable feature representations. Compared to the tested batch sizes, 128 was not the optimal choice for maximizing classification performance on CIFAR-10.

---

## Overall Conclusion

Among the evaluated batch sizes, 32 produced the best overall performance. Although larger batches optimized the training data more effectively, the smaller batch size achieved the highest test accuracy, indicating better generalization. This suggests that more frequent parameter updates helped the model learn more robust feature representations for unseen data.

---

# Experiment 10 — Final Best Configuration (20 Epochs)

## Model
CNN + Data Augmentation + Batch Normalization

## Structure
Same architecture as Experiment 6.

## Changes
Used the best-performing hyperparameters identified during previous experiments:

- Adam optimizer
- Learning rate = 0.001
- Batch size = 32
- Increased training from 10 to 20 epochs

All other settings remained unchanged.

## Results
- Loss: 0.6743
- Train Accuracy: 76.65%
- Test Accuracy: 77.38%

## Observations
- Achieved the lowest training loss across all experiments.
- Achieved the highest training and test accuracy.
- Training and test accuracy improved steadily throughout all 20 epochs.
- No significant signs of overfitting were observed.
- The confusion matrix showed stronger class separation, with most errors limited to visually similar animal classes.
- Training for additional epochs allowed the network to continue learning meaningful feature representations rather than plateauing after 10 epochs.

## Conclusion
Training the best-performing CNN configuration for 20 epochs produced the strongest overall results. Extending the training duration significantly improved optimization and generalization, increasing the test accuracy from 71.89% in the baseline model to 77.38%. This experiment demonstrates that the original 10-epoch training schedule was insufficient for the network to fully converge, and that additional training yielded the largest performance improvement among all experiments.

---
# Experiment 11: Transfer Learning (ResNet18)

Transfer learning was introduced to compare a pretrained convolutional neural network against the custom CNN built from scratch. Instead of learning all visual features from random initialization, ResNet18 starts with weights pretrained on the ImageNet dataset.

---

## Experiment 11A: Feature Extraction

### Objective

Evaluate how a pretrained ResNet18 performs when only the final classification layer is trained.

### Changes Made

- Loaded a pretrained ResNet18 model.
- Replaced the original fully connected layer:
  - Original: `Linear(512, 1000)`
  - New: `Linear(512, 10)`
- Froze all pretrained layers.
- Trained only the final classifier (`fc`).
- Resized CIFAR-10 images from **32 × 32** to **224 × 224**.
- Used ImageNet normalization values.
- Learning Rate: **0.001**
- Epochs: **5**

### Model Statistics

| Metric | Value |
|---------|------:|
| Total Parameters | 11,181,642 |
| Trainable Parameters | 5,130 |

### Results

| Metric | Value |
|---------|------:|
| Train Accuracy | 80.25% |
| Test Accuracy | **80.24%** |

### Observations

- Only the final classifier was updated during training.
- The pretrained feature extractor significantly outperformed the custom CNN.
- Training converged quickly despite updating less than 0.05% of the model parameters.
- Performance plateaued after several epochs because the feature extractor remained frozen.

### Conclusion

Feature extraction demonstrated the effectiveness of transfer learning. By reusing features learned from ImageNet, the model achieved better performance than the custom CNN while training only the final classification layer.

---

## Experiment 11B: Partial Fine-Tuning

### Objective

Evaluate whether allowing the final residual block to learn CIFAR-10 specific features improves performance over feature extraction.

### Changes Made

- Unfroze **layer4** of ResNet18.
- Continued training the final fully connected layer.
- Kept all earlier layers frozen.
- Reduced learning rate from **0.001** to **0.0001**.
- Trained for **1 epoch**.

### Results

| Metric | Value |
|---------|------:|
| Train Accuracy | 85.67% |
| Test Accuracy | **89.75%** |

### Observations

- Fine-tuning produced a substantial improvement over feature extraction.
- Allowing the final residual block to adapt to CIFAR-10 resulted in much better high-level feature representations.
- The model generalized well despite increasing the number of trainable parameters.
- Even a single epoch of fine-tuning produced a large improvement in accuracy.

### Conclusion

Partial fine-tuning was the best-performing approach. Keeping the low-level ImageNet features while adapting only the highest-level features allowed the model to specialize for CIFAR-10 and achieve the highest test accuracy.

---

# Transfer Learning Summary

| Model | Training Strategy | Test Accuracy |
|--------|------------------|--------------:|
| Custom CNN | Train from Scratch | 77.38% |
| ResNet18 | Feature Extraction | 80.24% |
| ResNet18 | Partial Fine-Tuning | **89.75%** |

### Overall Conclusion

Transfer learning significantly outperformed training a CNN from scratch. Feature extraction alone provided a noticeable improvement over the custom CNN, while partial fine-tuning of the final residual block achieved the best overall performance. This demonstrates that pretrained ImageNet features are highly transferable to CIFAR-10, and allowing higher-level layers to adapt to the new dataset results in substantial gains in classification accuracy while preserving the robust low-level features learned during pretraining.

# Overall Experimental Summary

| Experiment | Modification | Test Accuracy | Outcome |
|------------|--------------|--------------:|---------|
| 1 | MLP Baseline | 49.57% | Established initial baseline |
| 2 | CNN Baseline | 69.61% | Significant improvement over MLP but showed overfitting |
| 3 | Data Augmentation | 71.89% | Improved generalization and reduced overfitting |
| 4 | Dropout | 71.97% | Minimal improvement |
| 5 | Weight Decay | 72.45% | Small improvement in generalization |
| 6 | Batch Normalization | 74.08% | Largest architectural improvement |
| 7A | Learning Rate = 0.0001 | 72.16% | Stable but slower convergence |
| 7B | Learning Rate = 0.01 | 55.82% | Learning rate too high; poor optimization |
| 8 | SGD + Momentum | 70.26% | Adam provided better generalization |
| 9A | Batch Size = 32 | 73.86% | Best hyperparameter change |
| 9B | Batch Size = 128 | 71.57% | Better optimization but weaker generalization |
| 10 | Final Best Configuration (20 Epochs) | 77.38% | Best overall model |

## Final Conclusions

The experiments demonstrated that systematically modifying one component at a time provides valuable insight into CNN performance. Among the architectural changes, Batch Normalization produced the largest improvement, while Dropout had little effect on this relatively small network. Hyperparameter tuning showed that a learning rate of 0.001, the Adam optimizer, and a batch size of 32 provided the best balance between optimization and generalization. Finally, increasing the training duration from 10 to 20 epochs produced the highest overall performance, achieving 77.38% test accuracy without noticeable overfitting. These experiments illustrate the importance of controlled experimentation when improving deep learning models.

