# CSCI-566 Assignment 2

This assignment took me ~2.5 days to complete. It was amazing to get to apply high-level concepts learned in lecture to low-level implementation of each of the problems here. Below are some great learning points from each of the problems.

![puppy](problem_1_CNN/cs566/notebook_images/puppy.jpg)

## Problem 1: Convolutional Neural Networks

### `cnn.py`

- initializing a cnn and storing its weights in a dictionary
- backprop gradients for linear, max pool, relu, and convolution all derived by hand 🔥

### `layers.py`

- affine: fancy term for $Y = Wx + b$
- relu forward & backward clean code
- handwritten `softmax_loss` before I found out again about the helper function
- `batchnorm_forward` fun to implement
- `batchnorm_backward` was derived by hand, but learned about *indirect contribution* of gradients (effect of $x$ on $\mu$ and $\sigma^2$)
- conv, max pool forward & backward naive was cool but probably won't use it
  - deriving backprop by hand was pretty cool again
- spatial functions just an extension of their non-spatial versions but with a few insights:
  - choosing which axes to sum/mean/var across
  - broadcasting a particular dimension
  - transposing multiple dimensions

### `optim.py`

- implemented adam before, but good reminder of first, second moment, unbias

### ipynb

- really cool *TDD* approach to testing each small piece of a CNN
- `eval_numerical_gradient`: interesting way to calculate gradients with a point estimate and an applied small $\delta h$
- cool filter visualization
- lots of other layers in `fast_layer.py` to learn from

## Problem 2: Transformers

- calculating *attention* with $Q$, $K$, $V$ 😮‍💨
- multi-head attention layer
  - parallelization strength comes from here
- transformer blocks
- positional encoding
- BERT Model
- pre-training with MLM
  - randomly replace words in the text with mask tokens
  - predict the mask tokens
- fine-tuning -- it's always all just
  1. load data
  2. load model and tokenizer (or build your own)
  3. train the model: [predict, calculate loss, backprop, step optimizer]
  4. evaluate on test set
  5. save model
  6. run inference
- this also built the foundation for my very first self-fine-tuned model, [scottsus/mamba-2.8b-instruct-hf](https://huggingface.co/scottsus/mamba-2.8b-instruct-hf)

## Problem 3: Graph Convolutional Networks

- understood the `torch_geometric` library
- used the Open Graph Benchmark
- built a custom GCN, trained, and tested it
