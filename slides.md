### Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations

Maziar Raissi, Paris Perdikaris, and George Em Karniadakis

---

### Small data regime of complex systems
- In complex systems analysis, the cost of data acquisition is prohibitive.
- We are forced to draw conclusions and make decisions under partial information.
- In this small data regime, the majority of state-of the art ML techniques (e.g. CNN and RNN) lack robustness and certainty of convergence.

--

### Amplify the information context utilizing prior knowledge
- While modeling physical and biological systems, there is a large amount of prior knowledge that can act as a regularization agent; currently not being utilized in modern ML practice.
- By taking into account this prior knowledge (e.g. physical laws), the space of admissible solutions can be constrained to a manageable size.
- Encoding such structured information, results in amplifying the information content of the available data, enabling quick identification of the right solution, and good generalization with only a few training samples.

---

### Limitations of Gaussian processes
- Gaussian processes (that also elegantly exploit such structured information), pose two important limitations when treating nonlinear problems.
1) Need for local linearization of nonlinear terms with respect to time, and thus limiting the applicability to discrete-time domains.
2) Certain prior assumptions are required, limiting the representation capacity of the model.

--

### Bypassing these limitations
- These limitations are bypassed by employing deep neural networks, leveraging their capability to act as universal function approximators.
- In this setting, nonlinear problems can be directly tackled without the need of any prior assumptions, linearizations or local time-stepping.

---

### Physics Informed Neural Networks (PINNs)
- Exploiting automatic differentiation (Baydin, 2015), leads to neural networks constrained to respect symmetries, invariances and/or conservation principles.
- This simple, but powerful, construction allows a wide range of problems to be tackled.

--
### Automatic Differentiation
- Technique to compute exact derivatives of a function specified by a computer program
- Based on the chain rule, it overcomes the limitations of both symbolic (complexity) and numerical (inaccuracy) differentiation.
$$
\frac{\partial f}{\partial x} = \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h}
\cdot \frac{\partial h}{\partial x}
$$
- It is not a numerical approximation, but an exact computation of the derivative

--
### The Computational Graph
$$
f(x_1, x_2) = x_1 x_2 + \sin(x_1)
$$
<p>$$ v_1 = x_1 $$</p>
<p>$$ v_2 = x_2 $$</p>
<p>$$ v_3 = v_1 \cdot v_2 $$</p>
<p>$$ v_4 = \sin(v_1) $$</p>
<p>$$ y = v_5 = v_3 + v_4 $$</p>
- AD works by propagating derivatives through this graph.

--

### Forward Mode AD
- We compute the function's value and its derivative simultaneously, moving forward through the graph.
- Tangent tuples of the form ⟨value,derivative⟩
- Standard derivative rules:
<p>
$$
\langle u, \dot{u} \rangle + \langle v, \dot{v} \rangle = \langle u+v, \dot{u}+\dot{v} \rangle
$$
</p>

<p>
$$
\langle u, \dot{u} \rangle \cdot \langle v, \dot{v} \rangle = \langle u \cdot v, u\dot{v} + v\dot{u} \rangle
$$
</p>

<p>
$$
\sin(\langle u, \dot{u} \rangle) = \langle \sin(u), \cos(u) \cdot \dot{u} \rangle
$$
</p>

--

### Walkthrough Example
<p>
Let's compute $\frac{\partial f}{\partial x_1}$ for the function $f(x_1, x_2) = x_1 x_2 + \sin(x_1)$ at the point $(x_1, x_2) = (2, 5)$.
</p>

<p>
We set the initial "seeds": $\dot{x_1} = 1$ (since $\frac{\partial x_1}{\partial x_1} = 1$) and $\dot{x_2} = 0$ (since $\frac{\partial x_2}{\partial x_1} = 0$).
</p>

<!-- The evaluation trace -->
<p style="margin-top: 30px;">
Evaluation Trace:
</p>

$$
v_1 = x_1 \quad \rightarrow \quad \langle 2, 1 \rangle
$$

$$
v_2 = x_2 \quad \rightarrow \quad \langle 5, 0 \rangle
$$

$$
v_3 = v_1 \cdot v_2 \quad \rightarrow \quad \langle 2 \cdot 5, (2 \cdot 0 + 5 \cdot 1) \rangle = \langle 10, 5 \rangle
$$

$$
v_4 = \sin(v_1) \quad \rightarrow \quad \langle \sin(2), \cos(2) \cdot 1 \rangle \approx \langle 0.91, -0.42 \rangle
$$

$$
y = v_5 = v_3 + v_4 \quad \rightarrow \quad \langle 10+0.91, 5-0.42 \rangle = \langle 10.91, 4.58 \rangle
$$

<p style="margin-top: 30px;">
The final derivative is the second component of the result: $\frac{\partial f}{\partial x_1} \approx 4.58$.
</p>
--
<h4>Reverse Mode AD (Adjoint Mode)</h4>

<p>A highly efficient <strong>two-pass process</strong> for computing gradients:</p>

<ul>
  <li><strong>1. Forward Pass:</strong> Compute the value of every node in the graph and cache the results.</li>
  <li><strong>2. Backward Pass:</strong> Propagate gradients from the final output back to the inputs.</li>
</ul>

<p>
The <strong>adjoint</strong>, $\bar{v}$, represents the gradient of the final output $y$ with respect to any intermediate node $v$:
</p>
$$
\bar{v} = \frac{\partial y}{\partial v}
$$

<p class="fragment">
The core update rule accumulates these gradients backward through the graph:
</p>
<p class="fragment">
$$
\bar{u} \leftarrow \bar{u} + \bar{v} \cdot \frac{\partial v}{\partial u}
$$
</p>

--

### Reverse Mode - Walkthrough Example

<p>
Let's again compute $\frac{\partial f}{\partial x_1}$ for $f(x_1, x_2) = x_1 x_2 + \sin(x_1)$ at $(x_1, x_2) = (2, 5)$.
</p>

<!-- FORWARD PASS -->
<h5 style="margin-top:25px;">1. Forward Pass (calculate values)</h5>
$$
v_1 = 2, \quad v_2 = 5
$$
$$
v_3 = v_1 \cdot v_2 = 10
$$
$$
v_4 = \sin(v_1) = \sin(2) \approx 0.91
$$
$$
y = v_5 = v_3 + v_4 \approx 10.91
$$

--

### Backwards Pass (Part 1)
<!-- BACKWARD PASS -->
<h5 style="margin-top:25px;">2. Backward Pass (propagate gradients)</h5>
<p>
Start at the end: $\bar{y} = \bar{v}_5 = 1$ (since $\frac{\partial y}{\partial y} = 1$). Initialize all other $\bar{v}_i = 0$.
</p>

<p><strong>Node $v_5$ ($y = v_3 + v_4$):</strong></p>
$$ \bar{v}_3 = \bar{v}_5 \cdot \frac{\partial v_5}{\partial v_3} = 1 \cdot 1 = 1 $$
$$ \bar{v}_4 = \bar{v}_5 \cdot \frac{\partial v_5}{\partial v_4} = 1 \cdot 1 = 1 $$

<p><strong>Node $v_4$ ($v_4 = \sin(v_1)$):</strong></p>
$$ \bar{v}_1 \leftarrow \bar{v}_1 + \left(\bar{v}_4 \cdot \frac{\partial v_4}{\partial v_1}\right) = 0 + (1 \cdot \cos(2)) \approx -0.42 $$

--

### Backward Pass (Part 2)
<p><strong>Node $v_3$ ($v_3 = v_1 \cdot v_2$):</strong></p>
$$ \bar{v}_1 \leftarrow \bar{v}_1 + \left(\bar{v}_3 \cdot \frac{\partial v_3}{\partial v_1}\right) \approx -0.42 + (1 \cdot v_2) = -0.42 + 5 = 4.58 $$
$$ \bar{v}_2 \leftarrow \bar{v}_2 + \left(\bar{v}_3 \cdot \frac{\partial v_3}{\partial v_2}\right) = 0 + (1 \cdot v_1) = 2 $$

<!-- FINAL GRADIENTS -->
<h5 style="margin-top:25px;">Final Gradients:</h5>
$$
\bar{x}_1 = \bar{v}_1 = 4.58
$$
$$
\bar{x}_2 = \bar{v}_2 = 2
$$

--

### Why Reverse Mode is Key for Deep Learning

<div class="r-stack">
  <div class="fragment fade-in-then-out">
    <h5>Forward Mode</h5>
    <ul>
      <li><strong>How it works:</strong> One full pass through the graph is needed <em>for each input parameter</em>.</li>
      <li><strong>Cost:</strong> Proportional to the number of inputs ($N_{\text{in}}$).</li>
      <li><strong>Best for:</strong> Functions with few inputs and many outputs.<br>
          $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ where $n \ll m$.
      </li>
    </ul>
  </div>

  <div class="fragment fade-in-then-out">
    <h5>Reverse Mode</h5>
    <ul>
      <li><strong>How it works:</strong> Computes the gradient for <em>all inputs</em> in a single forward/backward pass.</li>
      <li><strong>Cost:</strong> Proportional to the number of outputs ($N_{\text{out}}$).</li>
      <li><strong>Best for:</strong> Functions with many inputs and few outputs.<br>
          $f: \mathbb{R}^n \rightarrow \mathbb{R}^m$ where $m \ll n$.
      </li>
    </ul>
  </div>

  <div class="fragment">
    <h5>The Deep Learning Connection</h5>
    <p>
      Training a neural network is the ultimate "many inputs, one output" problem:
    </p>
    <p style="text-align: center; font-size: 1.2em; margin-top: 25px;">
      $f: \mathbb{R}^{\text{millions of parameters}} \rightarrow \mathbb{R}^{\text{1 loss value}}$
    </p>
    <p class="fragment" style="margin-top: 25px;">
      Reverse Mode AD is dramatically more efficient for this task. 
    </p>
  </div>
</div>

---

### The Governing Equation
$$
u_t + \mathcal{N}[u] = 0, \quad x \in \Omega, \ t \in [0, T]
$$

--
### Neural Network Approximation
$$
u(t, x) \approx NN(t, x; \theta)
$$

--

### The PDE residual
$$
f(t, x) := u_t + \mathcal{N}[u]
$$

--
### The composite Loss Function
<p>$$ \text{MSE} = \text{MSE}_u + \text{MSE}_f $$</p>

<p>$$ \text{MSE}_u = \frac{1}{N_u} \sum_{i=1}^{N_u} |u(t_u^i, x_u^i) - u^i|^2 $$</p>

<p>$$ \text{MSE}_f = \frac{1}{N_f} \sum_{i=1}^{N_f} |f(t_f^i, x_f^i)|^2 $$</p>
---

### Physics Informed Neural Networks (PINNs)

- A new class of universal function approximators that is capable of encoding any underlying physical laws that can be described by partial differential equations.
- The implementation simplicity greatly favors rapid development and testing of new ideas, potentially opening the path for a new era in data-driven scientific computing.
- One pressing question involves addressing the problem of quantifying the uncertainty associated with the neural network predictions.


---

### Thank You!

