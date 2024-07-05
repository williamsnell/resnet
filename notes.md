# July 5th

- Ran a sweep of hyperparameters on a default ResNet34 
    implementation and CIFAR-10 dataset.
- Only one epoch was attempted, to try and find learning rate
    and batch size hyperparameters. 
- Low learning rate and small batch sizes seemed to perform
    the best - by a lot. Both correlate with longer runtime.
- GPU usage was very low with the ideal parameters (small batch size),
    which suggests an H100 is probably not a good choice (which makes
    sense, given the dataset is only ~130MB compressed.)

- Going to take the best learning rate (~0.0007) and a small batch size
    (~64) and try train across a lot of epochs.
- Also going to try the same settings with a model that's twice as large
    and see how it performs.
