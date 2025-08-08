# Name the Dataset or Name the Label: Dataset Bias Investigation

This project examines dataset bias through the lens of **Name the Dataset** — a task introduced by Torralba & Efros (*CVPR 2011*) in which a classifier predicts the source dataset of an image — and **Name the Label**, a semantic classification task.  

The study decomposes dataset bias into **semantic** (generalizable and transferable) and **non-semantic** components, revealing their roles via linear probe experiments and showing that different neural architectures mediate bias in distinct ways.  

Two Domain-Adversarial Neural Network (**DANN**) variants are proposed — **one-hot minimax** and **uniform double minimax** — which incorporate a *Name the Dataset* loss into the total loss function to mitigate bias.  

Experiments on a combined **TinyImageNet + CIFAR-100** dataset, training **ResNet-18** for 273-class semantic classification, achieve ~2% improvement in classification accuracy; training **ViT-B/32** under the same setting yields a smaller gain of ~0.5%.  

While the approach suppresses both semantic and non-semantic components (despite only non-semantic suppression being desirable), **fortunately, in this setting, the non-semantic component is dominant**.
