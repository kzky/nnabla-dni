# Decoupled Neural Interfaces with NNabla

Decoupled Neural Interfaces usnig Synthetic Gradients uses Gradient Synthesizer which takes the input of the result of the forward pass at a certain depth, then generate gradients, and in the backward pass it uses these synthetic gradients. Feeddforward neural networks can be separated at this depth, thus it is called `Decoupled`.

Here is the reproduction of DNI using MNIST dataset. MLP and CNN are used as network architectures. The experimantal settings, e.g., layers used in a network, batch size, and etc., are a bit different from the ones described in the paper, but it actually works.

Run as the following, 

```sh

$ python mnist_main.py -c "cuda.cudnn" -d 0 -b 32

```

## References
- Max Jaderberg, Wojciech Marian Czarnecki, Simon Osindero, Oriol Vinyals, Alex Graves, David Silver, and Koray Kavukcuoglu, "Decoupled Neural Interfaces using Synthetic Gradients", https://arxiv.org/abs/1608.05343
