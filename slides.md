# Small data regime of complex systems
In complex systems analysis, the cost of data acquisition is prohibitive.
We are forced to draw conclusions and make decisions under partial information.
In this small data regime, the majority of state-of the art ML techniques (e.g. CNN and RNN) lack robustness and certainty of convergence.

--

# Amplify the information context utilizing prior knowledge
While modeling physical and biological systems, there is a large amount of prior knowledge that can act as a regularization agent; currently not being utilized in modern ML practice.
By taking into account this prior knowledge (e.g. physical laws), the space of admissible solutions can be constrained to a manageable size.
Encoding such structured information, results in amplifying the information content of the available data, enabling quick identification of the right solution, and good generalization with only a few training samples.

---

# Limitations of Gaussian processes
Gaussian processes (that also elegantly exploit such structured information), pose two important limitations when treating nonlinear problems.
1. Need for local linearization of nonlinear terms with respect to time, and thus limiting the applicability to discrete-time domains.
2. Certain prior assumptions are required, limiting the representation capacity of the model.

--

# Bypassing these limitations
These limitations are bypassed by employing deep neural networks, leveraging their capability to act as universal function approximators.
In this setting, nonlinear problems can be directly tackled without the need of any prior assumptions, linearizations or local time-stepping.

---

# Physics Informed Neural Networks (PINNs)
Exploiting automatic differentiation (Baydin, 2015), leads to neural networks constrained to respect symmetries, invariances and/or conservation principles.
This simple, but powerful, construction allows a wide range of problems to be tackled.

--

# Main problem classes
1. Data-driven solution
2. Data-driven siscovery

(du/dt) + N[u;λ] = 0
This setup encapsulates a wide range of problems in mathematical physics
1. conservation laws
2. diffusion processes
3. advection-diffusion-reaction systems
4. kinetic equations

--

# 1-D Burgers' equation
N [u; λ] = λ1uux − λ2uxx and λ = (λ1, λ2)
Given noisy measurments, we want to solve two distinct problems:
1. Predictive inference, filtering and smoothing of PDEs
	- given fixed λ, what can be said about the hidden state?
2. Learning and system identification of PDEs
	- which λ best describe the observed data?

