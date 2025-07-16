# Presentation slides with Reveal.js

Stylianos (Stelios) Anastasakis, 2025

---

# Why Reveal.js?

- Cross-platform
- Markdown-based
- Supports math with LaTeX
- Looks great

---

# Inline Math

Einstein's formula: $E = mc^2$

---

# Block Math

$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

---

# Vertical slides

This is the parent of vertical slides.

--
## Vertical slide 1

This is a vertical slide under the parent slide.

--
## Vertical slide 2

Another vertical slide under the parent slide.

---

# Add figures

Add a figure with Markdown code

```markdown
    ![Histogram of the solution of a bistable ODE](figures/demo.png)
```

![Histogram](figures/demo.png)

--

or with HTML code for more control

```html
<img src="figures/demo.png" alt="Histogram" width="400">
```

<img src="figures/demo.png" alt="Histogram" width="400">

--

or with percentage

```html
<img src="figures/demo.png" alt="Histogram" style="width:40%">
```

<img src="figures/demo.png" alt="Histogram" style="width:40%">

--

You can add a caption like this
```html
<figure>
  <img src="figures/demo.png" alt="Time series" style="width:70%">
  <figcaption>Figure 1: Histogram of the solution of a bistable ODE</figcaption>
</figure>
```

<figure>
  <img src="figures/demo.png" alt="Time series" style="width:70%">
  <figcaption>Figure 1: Histogram of the solution of a bistable ODE</figcaption>
</figure>

---

# Show a video

```html
<video src="media/video.mp4" autoplay muted loop style="width: 60%"></video>
```

<video src="media/video.mp4" autoplay muted loop style="width: 60%"></video>


---

# Code blocks

<pre><code class="language-python" data-trim>
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
</code></pre>


--

# Code blocks with highlighting

<pre><code class="language-python" data-trim data-line-numbers="3,5-6,10">
import numpy as np
import matplotlib.pyplot as plt

def simulate_ode(f, y0, t):
    """Simple forward Euler ODE solver."""
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return y

# Example usage
f = lambda t, y: -0.5 * y
t = np.linspace(0, 10, 100)
y = simulate_ode(f, 1.0, t)

plt.plot(t, y)
plt.title("Exponential Decay")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.grid()
plt.show()
</code></pre>


--

<section>
  <h3>Code blocks with animations</h3>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
import numpy as np
import matplotlib.pyplot as plt
    </code></pre>
  </div>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
def simulate_ode(f, y0, t):
    """Simple forward Euler ODE solver."""
    y = np.zeros_like(t)
    y[0] = y0
    for i in range(1, len(t)):
        dt = t[i] - t[i-1]
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return y
    </code></pre>
  </div>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
f = lambda t, y: -0.5 * y
t = np.linspace(0, 10, 100)
y = simulate_ode(f, 1.0, t)
    </code></pre>
  </div>

  <div class="fragment">
    <pre><code class="language-python" data-trim data-line-numbers>
plt.plot(t, y)
plt.title("Exponential Decay")
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.grid()
plt.show()
    </code></pre>
  </div>
</section>



---

### 🦧 That is all 🦧


---

---
<!---

--

### Main problem classes
1) Data-driven solution
2) Data-driven discovery
$$
\ u_t + N[u; \lambda] = 0 \
$$
- This setup encapsulates a wide range of problems in mathematical physics
1) conservation laws
2) diffusion processes
3) advection-diffusion-reaction systems
4) kinetic equations

--

### 1-D Burgers' equation
$$
\ N[u; \lambda] = \underbrace{\lambda_1 u u_x}_{\text{Nonlinear Advection}} - \underbrace{\lambda_2 u_{xx}}_{\text{Diffusion (Viscosity)}} \
$$
- Given noisy measurments, we want to solve two distinct problems:
1) Predictive inference, filtering and smoothing of PDEs
	- given fixed λ, what can be said about the hidden state?
2) Learning and system identification of PDEs
	- which λ best describe the observed data?


-->
