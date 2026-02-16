Spires is a high performance spiking reservoir library. Spiking reservoirs are class of reservoir computing (RC) systems configured to utilize the temporal dynamics of spiking activity for computation. In general, a spiking reservoir includes $N$ interconnected spiking neurons, such as those based on the leaky integrate-and-fire neuron model, where each neuron integrates input from other respectively connected neurons over time, accumulating
a membrane potential $v\_i$:

$$ v_i(t+1)=(1-\eta) v_i(t)+\sum\_{j=1}^{N} W\_{ij} S_j(t) + \gamma W\_{in,i} u(t), $$

where $\eta$ is the leak rate, $S\_j(t)$ are the output spike impulses of other neurons, $\gamma$ is the input strength, $W\_{in,i}$ is the input weight, and $u(t)$ is the input signal.

When the membrane potential $v\_i$ surpasses an activation threshold $\theta$, the neuron transmits an impulse to downstream neurons:

$$
S\_i(t+1) =
\begin{cases}
    1, & \text{if } v\_i(t+1) > \theta, \\
    0, & \text{otherwise}.
\end{cases}
$$

After transmitting a spike, a neuron returns to its resting potential $v\_i = v\_{rest}$ for a refractory period $T\_{ref}$. This neuron model at least partially captures essential dynamics of biological neurons, thereby allowing the reservoir to exhibit biologically plausible dynamics while maintaining computational simplicity.

The neurons are connected together with weights $W$:

$$
W = \rho \frac{W\_0}{\lambda\_{max}(W\_0)},
$$

where $W\_0$ is a random weight matrix with a sparse connectivity $c$, $\lambda\_{max}(W\_0)$ is the largest eigenvalue of $W\_0$, and $\rho$ is the desired spectral radius.

A readout layer with weights $W_{out}$ maps the high-dimensional, dynamical state of the reservoir into an output $y(t)$:

$$
y(t) = W\_{out} \cdot \mathbf{v}(t).
$$

The readout layer is trained to minimize the error $\epsilon(t)$ between the target, $y\_{target}(t)$ and the predicted output $y(t)$: $\epsilon(t) = y\_{target}(t) - y(t)$, while the internal
reservoir weights remain fixed. The readout layer is trained by updating $W\_{out}$:

$$
W\_{out} \leftarrow W\_{out} + \mu \cdot \epsilon(t) \cdot \mathbf{v}(t)^\intercal,
$$

where $\mu$ is the learning rate.
