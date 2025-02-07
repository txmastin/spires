Spiking reservoirs are class of reservoir computing (RC) systems configured to utilize the temporal dynamics of spiking activity for computation. In general, a spiking reservoir includes $N$ interconnected spiking neurons, such as those based on the leaky integrate-and-fire neuron model, where each neuron integrates input from other respectively connected neurons over time, accumulating
a membrane potential $v_i$:


$$ v_i(t+1)=(1-\eta) v_i(t)+\sum_{j=1}^{N} W_{ij} S_j(t) + \gamma W_{in,i} u(t), $$

where $\eta$ is the leak rate, $S_j(t)$ are the output spike impulses of other neurons, $\gamma$ is the input strength, $W_{in,i}$ is the input weight, and $u(t)$ is the input signal.

When the membrane potential $v_i$  surpasses an activation threshold $\theta$, the neuron transmits an impulse to downstream neurons:


$$
S_i(t+1) =
\begin{cases}
    1, & \text{if } v_i(t+1) > \theta, \\
    0, & \text{otherwise}.
\end{cases}
$$

After transmitting a spike, a neuron returns to its resting potential $v_i = v_{rest}$ for a refractory period $T_{ref}$. This neuron model at least partially captures essential dynamics of biological neurons, thereby allowing the reservoir to exhibit biologically plausible dynamics while maintaining computational simplicity.

The neurons are connected together with weights $W$:

$$
W = \rho \frac{W_0}{\lambda_{max}(W_0)},
$$

where $W_0$ is a random weight matrix with a sparse connectivity $c$, $\lambda_{max}(W_0)$ is the largest eigenvalue of $W_0$, and $\rho$ is the desired spectral radius.


A readout layer with weights $W_{out}$ maps the high-dimensional, dynamical state of the reservoir into an output $y(t)$:

$$
y(t) = W_{out} \cdot \mathbf{v}(t).
$$

The readout layer is trained to minimize the error $\epsilon(t)$ between the target, $y_{target}(t)$ and the predicted output $y(t)$: $\epsilon(t) = y_{target}(t) - y(t)$, while the internal
reservoir weights remain fixed. The readout layer is trained by updating $W_{out}$:

$$
W_{out} \leftarrow W_{out} + \mu \cdot \epsilon(t) \cdot \mathbf{v}(t)^\intercal,
$$

where $\mu$ is the learning rate.
