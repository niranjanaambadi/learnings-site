---
title: "FakeFinder: an AI That Detects AI-Generated Images"
date: 2025-12-2
tags: [projects,learnings,automl,CNN]
---

## Building an AI That Detects AI-Generated Images: A Deep Dive into AutoML

*How I built a flexible CNN architecture with automated hyperparameter tuning to distinguish real photos from AI-generated fakes*

---

With tools like Stable Diffusion, MidJourney, and DALL·E making it trivially easy to generate photorealistic images, an important question emerges: can we build AI systems that detect AI-generated content? In this post, I'll walk through how I built **FakeFinder**, a CNN-based classifier that learns to distinguish real images from synthetic ones, and more importantly, how I used **AutoML** to automatically discover the best model architecture.

## The Problem: Real vs. Fake

The dataset I worked with contains images from two categories:
- **Real images**: Photographs from sources like Pexels, Unsplash, and WikiArt
- **Fake images**: AI-generated images from Stable Diffusion, MidJourney, and DALL·E

At first glance, many of these images are indistinguishable to the human eye. But neural networks can learn to pick up on subtle artifacts and patterns that generative models leave behind.

## The Challenge: So Many Choices

When building a CNN for image classification, you're immediately faced with a barrage of decisions:

- How many convolutional layers should I use?
- How many filters in each layer?
- What kernel size works best?
- How much dropout for regularization?
- What learning rate?
- What input resolution?

Traditionally, you'd make educated guesses, train a model, evaluate it, tweak the parameters, and repeat. This is tedious and often leads to suboptimal results because the search space is enormous.

**What if we could automate this entire process?**

## Enter AutoML with Optuna

[Optuna](https://optuna.org/) is a hyperparameter optimization framework that uses Bayesian optimization to intelligently search through the parameter space. Instead of random guessing or grid search, it learns from previous trials to focus on promising regions.

The key insight is this: rather than building one fixed model, we build a **flexible model** that can take on different configurations, then let Optuna figure out which configuration works best.

## Building a Flexible CNN

The heart of this project is the `FlexibleCNN` class. Unlike a traditional CNN where the architecture is hardcoded, this model builds itself dynamically based on parameters passed to it.

### The Core Idea

```python
class FlexibleCNN(nn.Module):
    def __init__(self, n_layers, n_filters, kernel_sizes, dropout_rate, fc_size):
        super().__init__()
        
        self.features = nn.ModuleList()  # Dynamic list of layers
        in_channels = 3  # RGB input
        
        # Build convolutional blocks based on parameters
        for i in range(n_layers):
            block = nn.Sequential(
                nn.Conv2d(in_channels, n_filters[i], kernel_sizes[i], padding='same'),
                nn.BatchNorm2d(n_filters[i]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )
            self.features.append(block)
            in_channels = n_filters[i]
```

Notice how `n_layers`, `n_filters`, and `kernel_sizes` are all parameters. This means the same class can represent:
- A shallow 1-layer network with 8 filters
- A deep 3-layer network with [64, 128, 256] filters
- Anything in between

### The Clever Part: Lazy Classifier Initialization

Here's a challenge: the fully connected classifier needs to know the size of the flattened feature maps. But that size depends on the input resolution AND the number of pooling layers, both of which are hyperparameters we're tuning.

The solution? **Lazy initialization**. We don't create the classifier until the first forward pass, when we can actually measure the feature map size:

```python
def forward(self, x):
    # Apply all conv blocks
    for layer in self.features:
        x = layer(x)
    
    # Flatten
    x = torch.flatten(x, start_dim=1)
    
    # Create classifier on first pass (now we know the size!)
    if self.classifier is None:
        self._create_classifier(x.shape[1])
        self.classifier.to(x.device)
    
    return self.classifier(x)
```

This pattern is powerful because it means our model truly adapts to any combination of hyperparameters without manual calculation.

## Designing the Search Space

With Optuna, you define a **search space** by telling it what parameters to tune and their valid ranges:

```python
def design_search_space(trial):
    # Architecture parameters
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    n_filters = [
        trial.suggest_int(f"n_filters_layer{i}", 8, 64, step=8)
        for i in range(n_layers)
    ]
    
    kernel_sizes = [
        trial.suggest_int(f"kernel_size_layer{i}", 3, 5, step=2)
        for i in range(n_layers)
    ]
    
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    fc_size = trial.suggest_int("fc_size", 64, 512, step=64)
    
    # Training parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    resolution = trial.suggest_categorical("resolution", [16, 32, 64])
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    
    return { ... }
```

A few things to note:

1. **`suggest_int` with `step`**: For filters, we search in multiples of 8 (8, 16, 24, ..., 64). This reduces the search space while keeping sensible values.

2. **`log=True` for learning rate**: Learning rates vary across orders of magnitude (0.0001 vs 0.01), so we search on a log scale. This gives equal probability to each order of magnitude.

3. **`suggest_categorical` for resolution**: Some parameters aren't continuous—we only want to try specific image sizes.

4. **Dynamic list lengths**: Notice how `n_filters` and `kernel_sizes` are lists whose length depends on `n_layers`. Optuna handles this elegantly.

## The Objective Function

Optuna needs an **objective function** that takes a trial, builds a model with the suggested parameters, trains it, and returns a score to maximize (or minimize):

```python
def objective_function(trial, device, dataset_path, n_epochs=4):
    # 1. Get hyperparameters
    params = design_search_space(trial)
    
    # 2. Create data transforms with suggested resolution
    transform = transforms.Compose([
        transforms.Resize((params["resolution"], params["resolution"])),
        transforms.ToTensor(),
    ])
    
    # 3. Build model with suggested architecture
    model = FlexibleCNN(
        n_layers=params["n_layers"],
        n_filters=params["n_filters"],
        kernel_sizes=params["kernel_sizes"],
        dropout_rate=params["dropout_rate"],
        fc_size=params["fc_size"]
    ).to(device)
    
    # 4. Train
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    for epoch in range(n_epochs):
        train_one_epoch(model, train_loader, optimizer)
    
    # 5. Evaluate and return accuracy
    accuracy = evaluate(model, val_loader)
    return accuracy
```

Optuna calls this function repeatedly, each time with different suggested parameters. It tracks which combinations lead to high accuracy and uses that information to make smarter suggestions in subsequent trials.

## Running the Optimization

With everything set up, running the optimization is straightforward:

```python
study = optuna.create_study(direction="maximize")  # We want high accuracy
study.optimize(objective_function, n_trials=25)

print(f"Best accuracy: {study.best_trial.value:.2%}")
print(f"Best parameters: {study.best_trial.params}")
```

After 25 trials (each taking about 3 minutes), Optuna found a configuration achieving **73.2% accuracy**.

## What Optuna Discovered

The best configuration was surprisingly compact:

| Parameter | Value |
|-----------|-------|
| n_layers | 2 |
| n_filters | [32, 24] |
| kernel_sizes | [5, 3] |
| dropout_rate | 0.17 |
| fc_size | 384 |
| learning_rate | 0.000732 |
| resolution | 16 |
| batch_size | 16 |

Some interesting observations:

1. **Smaller is better**: The best model used only 16×16 resolution and 2 layers. Larger models actually performed worse, likely due to overfitting on the limited training data.

2. **Low dropout**: With a small model, aggressive dropout (0.5) hurt performance. The optimal 0.17 provided just enough regularization.

3. **Decreasing filters**: The filter pattern [32, 24] is unusual—typically we increase filters in deeper layers. But Optuna found this worked best for this specific problem.

This is the power of AutoML: it can discover counterintuitive configurations that a human might never try.

## Beyond Accuracy: Model Efficiency

Accuracy isn't everything. In production, you might care about:
- **Model size**: Fewer parameters = smaller deployment footprint
- **Inference speed**: Faster predictions = better user experience
- **Training cost**: Smaller models train faster and cheaper

I added a simple function to count trainable parameters:

```python
def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Among the top-5 models by accuracy, there was significant variation in size. This lets you make informed tradeoffs—maybe you accept 1% lower accuracy for a model that's 10x smaller.

## Lessons Learned

1. **Flexibility enables automation**: By designing a model that accepts architecture parameters, we turned a design problem into an optimization problem that Optuna could solve.

2. **Search space design matters**: The choices of ranges, steps, and scales significantly impact what Optuna can discover. Domain knowledge helps here.

3. **Start small**: Running 25 trials with 4 epochs each is manageable. You can always run more trials or more epochs once you've narrowed down promising regions.

4. **Trust the process**: AutoML might find configurations you wouldn't have tried. That's the point—it explores without human bias.

## Try It Yourself

The complete code is available on [GitHub](https://github.com/niranjanaambadi/FakeFinder). To run your own optimization:

```bash
git clone <>
cd <>
pip install -r requirements.txt
python train.py --data-path ./data --n-trials 25
```

## What's Next?

There's plenty of room for improvement:

- **Data augmentation**: Flips, rotations, and color jitter could help the model generalize better
- **Transfer learning**: Starting from a pretrained backbone (ResNet, EfficientNet) would likely boost accuracy significantly
- **Multi-objective optimization**: Optuna supports optimizing for multiple objectives (accuracy AND model size) simultaneously
- **Pruning**: Optuna can stop unpromising trials early, saving compute

The techniques here—flexible architectures, hyperparameter optimization, efficiency analysis—apply far beyond fake image detection. Any deep learning project can benefit from automating the tedious process of architecture search.

---

*Have questions or suggestions? Feel free to reach out or open an issue on the GitHub repo!*
