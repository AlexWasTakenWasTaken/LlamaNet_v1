<h1 align="center">LlamaNet</h1>
<h3 align="center">A custom built neural network library built from scratch in C++</h3>

<h3 align="left">Description:</h3>
Hi 👋! I'm Alex, and I developed a Neural Network deep learning library in C++ from scratch, using only the standard C++ libraries. The library features cutting-edge capabilities for building, training, and deploying neural networks and incorporates key advancements, such as the Adaptive Moment Estimation algorithm for adaptive learning rate adjustments and a fully customizable neural network architecture.

<h3 align="left">Details:</h3>

- Custom Neural Network Architecture: Designed and implemented fully customizable layers and activation functions (ReLU, Sigmoid, Tanh, etc.), allowing for custom network topology and training parameters.

- Optimization: Implemented Adam optimization for efficient and adaptive gradient descent, ensuring faster convergence, alongside Xavier initialization to improve weight initialization and promote stable gradient flow during training.

- Efficient Training: Enabled randomized mini-batch gradient descent with advanced backpropagation, supporting loss/cost functions like Huber Loss.

- Scalable Design: Built a modular framework supporting different topologies, making it easy to expand and adapt for various datasets and applications.

<h3 align="left">Testing:</h3>

- Achieved 98.10% accuracy over 10,000 test cases on the MNIST Handwritten Digits Classification Test

- Achieved 96.26% accuracy after a single epoch.

- See the Excel spreadsheet for a more in-depth analysis of the library's test results :D

<h3 align="left">Future Changes:</h3>

- Implement parallelization to improve computing time

- Implement cross-entropy cost function and softmax to optimize classification tasks

- Implement SVMs


You'll probably see these changes in v2 of the LlamaNet, where I will also implement a custom image processing library to be able to detect and classify hotdogs 🌭🌭🌭🌭. Hope to see you there!
