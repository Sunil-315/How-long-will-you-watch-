Based on the provided document and supplementing with standard technical context from the allowed sources (GeeksforGeeks/Tutorialspoint) where necessary to fully flesh out the concepts, here is a more elaborate and detailed discussion on the Activation Functions used in Artificial Neural Networks.

### The Critical Role of Activation Functions
Before diving into the specific types, it is vital to appreciate the mechanism described in the slides. In a biological brain, a neuron doesn't just pass a signal along; it decides *whether* to pass it along based on the strength of the input.

In an Artificial Neural Network (ANN), the activation function acts as this decision-making filter. As detailed in **Slide 17**, a neuron first calculates a weighted sum of all inputs ($v = \sum w_i x_i$). However, this sum is just a raw number. To make it useful, we pass this sum through a **non-linear filter**—the activation function ($y = f(v)$).

As **Slide 13** emphasizes, without these functions, a neural network is nothing more than a Linear Regression model. Real-world data—such as images, human speech, or stock market trends—is rarely linear. You cannot draw a straight line through it. Activation functions introduce **non-linearity**, giving the network the "expressive power" to map curves, twists, and complex relationships.

Here is a detailed breakdown of the six specific functions highlighted in the document.

---

### 1. Linear Function
The Linear Function is the most basic activation type, often referred to as the identity function.

*   **Mathematical Definition:**
    As shown in **Slide 19**, the equation is defined as $f(v) = a + v$. In many contexts, if we ignore the bias for a moment, it is simply proportional to the input ($f(x) = x$).
*   **Visual Representation:**
    The graph on the slide displays a straight diagonal line passing through the axes. There are no curves or bends. If the input increases, the output increases by the exact same proportion.
*   **Detailed Analysis:**
    While simple, the linear function is severely limited. **Slide 14** explains that "a linear combination of linear functions is still linear." This means that even if you stack 100 layers of neurons that use linear activation, the entire network behaves exactly like a single layer. It cannot learn complex patterns. It is generally only used in the final output layer of a network when we are trying to predict a continuous numerical value (like predicting the price of a house), but never in the hidden layers.

### 2. Heaviside Step Function
The Heaviside Step function serves as a binary classifier, mimicking the "all-or-nothing" firing mechanism of a biological neuron.

*   **Mathematical Definition:**
    According to **Slide 22**, the function relies on a threshold value, denoted as $t$.
    $$ f(v) = \begin{cases} 1 & \text{if } t \leq v \\ 0 & \text{otherwise} \end{cases} $$
*   **Visual Representation:**
    The graph is characterized by a sharp, vertical "step." The output line stays flat at 0 until it hits the threshold, at which point it jumps immediately to 1 and stays there.
*   **Detailed Analysis:**
    This function is a hard decider. It looks at the weighted sum ($v$) and compares it to the threshold ($t$). If the input is strong enough to cross the threshold, the neuron fires (1); if it falls even slightly short, the neuron remains completely inactive (0). While this is conceptually easy to understand, it is rarely used in modern deep learning because it is not differentiable—the sudden jump means we cannot calculate the "gradient" (slope) needed for the network to learn and update its weights during backpropagation.

### 3. Sigmoid Function
The Sigmoid function (or Logistic function) is historically one of the most significant activation functions, famous for its smooth, "S-shaped" curve.

*   **Mathematical Definition:**
    **Slide 23** provides the formula:
    $$ f(v) = \frac{1}{1 + e^{-v}} $$
*   **Visual Representation:**
    The graph shows a curve that starts near 0, rises smoothly, and flattens out near 1. Unlike the Step function, the transition is gradual.
*   **Detailed Analysis:**
    The Sigmoid function acts as a "squashing" function because it compresses (squashes) any input value, no matter how massive or tiny, into a strict range between **0 and 1**.
    This property makes it incredibly useful for probability prediction. For example, if the output is 0.95, the network is 95% confident. However, a major drawback (often cited in supplementary literature like GeeksforGeeks) is that for very high or very low inputs, the curve becomes almost flat. When the curve is flat, the gradient is near zero, which can cause the network to stop learning—a phenomenon known as the "Vanishing Gradient Problem."

### 4. Hyperbolic Tangent (tanh) Function
The `tanh` function is mathematically very similar to the Sigmoid function but solves one of its primary limitations regarding the range of the output.

*   **Mathematical Definition:**
    As presented on **Slide 26**:
    $$ f(v) = \frac{e^v - e^{-v}}{e^v + e^{-v}} $$
*   **Visual Representation:**
    Visually, the `tanh` graph looks like a taller Sigmoid curve. It is still S-shaped and smooth.
*   **Detailed Analysis:**
    The crucial difference lies in the range. While Sigmoid outputs 0 to 1, `tanh` outputs values between **-1 and 1**.
    This means the function is **zero-centered**. In many neural network architectures, having outputs that can be negative (indicating an inhibitory signal) rather than just zero (indicating no signal) helps the model center the data, often leading to faster learning compared to the Sigmoid function.

### 5. Rectified Linear Unit (ReLU) Function
ReLU has become the default activation function for most modern deep learning models due to its simplicity and effectiveness.

*   **Mathematical Definition:**
    **Slide 24** defines it as:
    $$ f(v) = \begin{cases} 0 & \text{if } v \leq 0 \\ v & \text{if } 0 \leq v \end{cases} $$
*   **Visual Representation:**
    The graph looks like a bent line. For the left side (negative inputs), the line is flat along the x-axis (output is 0). At the origin (0,0), it angles upward linearly.
*   **Detailed Analysis:**
    The logic here is "Rectification." The function allows positive values to pass through unchanged (linear behavior) but completely blocks negative values by turning them into zero.
    This offers two massive benefits:
    1.  **Computational Efficiency:** The computer only needs to check if a number is positive; it doesn't need to calculate complex exponents like in Sigmoid or Tanh.
    2.  **Sparsity:** By outputting actual zeros, it allows the network to ignore irrelevant neurons, making the model lighter.
    However, the range is **0 to $\infty$**, meaning output values can become very large.

### 6. Leaky Rectified Linear Unit (Leaky ReLU) Function
Leaky ReLU is a specialized modification of the standard ReLU designed to fix a specific problem known as the "Dying ReLU" problem.

*   **Mathematical Definition:**
    As shown in **Slide 25**:
    $$ f(v) = \begin{cases} av & \text{if } v \leq 0 \\ v & \text{if } 0 \leq v \end{cases} $$
    Here, **$a$** is a small positive number (e.g., 0.01).
*   **Visual Representation:**
    The graph looks almost identical to ReLU, but with a key difference on the left side. Instead of a flat line at zero, there is a slight, downward slope for negative values.
*   **Detailed Analysis:**
    In standard ReLU, if a neuron receives a negative input, it outputs 0. If it outputs 0, the gradient is 0, and the neuron essentially stops learning—it "dies."
    Leaky ReLU solves this by allowing a small, non-zero leak for negative values ($av$). This ensures that even for negative inputs, there is still a small gradient, keeping the neuron active and capable of learning. The possible output range extends from **$-\infty$ to $\infty$**.
