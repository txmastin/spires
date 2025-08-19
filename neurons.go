package spires

import (
	"fmt"
	"math"
	"math/rand"
)

// initNeuron creates a new neuron of the specified type
func initNeuron(neuronType NeuronType, params []float64, dt float64) (neuron, error) {
	switch neuronType {
	case NeuronLIFDiscrete:
		return newLIFDiscreteNeuron(params, dt)
	case NeuronLIFBio:
		return newLIFBioNeuron(params, dt)
	case NeuronFLIFCaputo:
		return newFLIFCaputoNeuron(params, dt)
	case NeuronFLIFGL:
		return newFLIFGLNeuron(params, dt)
	case NeuronFLIFDiffusive:
		return newFLIFDiffusiveNeuron(params, dt)
	default:
		return nil, fmt.Errorf("unknown neuron type: %d", neuronType)
	}
}

// LIF Discrete Neuron Implementation
type lifDiscreteNeuron struct {
	V         float64 // Membrane potential
	VTh       float64 // Threshold potential
	V0        float64 // Resting potential
	leakRate  float64 // Leak rate
	spike     float64 // Spike output
	bias      float64 // Bias current
	dt        float64 // Time step
}

func newLIFDiscreteNeuron(params []float64, dt float64) (*lifDiscreteNeuron, error) {
	if len(params) < 5 {
		return nil, fmt.Errorf("LIF discrete neuron requires at least 5 parameters")
	}
	
	return &lifDiscreteNeuron{
		V:        params[0], // Initial membrane potential
		VTh:      params[1], // Threshold
		V0:       params[2], // Resting potential
		leakRate: params[3], // Leak rate
		bias:     params[4], // Bias current
		dt:       dt,
	}, nil
}

func (n *lifDiscreteNeuron) update(input float64, dt float64) {
	// Update membrane potential
	n.V += dt * (-n.leakRate*(n.V-n.V0) + input + n.bias)
	
	// Check for spike
	if n.V >= n.VTh {
		n.spike = 1.0
		n.V = n.V0 // Reset to resting potential
	} else {
		n.spike = 0.0
	}
}

func (n *lifDiscreteNeuron) getState() float64 {
	return n.V
}

func (n *lifDiscreteNeuron) getSpike() float64 {
	return n.spike
}

func (n *lifDiscreteNeuron) reset() {
	n.V = n.V0
	n.spike = 0.0
}

// LIF Biological Neuron Implementation
type lifBioNeuron struct {
	V         float64 // Membrane potential
	VTh       float64 // Threshold potential
	V0        float64 // Resting potential
	leakRate  float64 // Leak rate
	spike     float64 // Spike output
	bias      float64 // Bias current
	refractoryPeriod float64 // Refractory period
	refractoryTimer  float64 // Current refractory timer
	dt        float64 // Time step
}

func newLIFBioNeuron(params []float64, dt float64) (*lifBioNeuron, error) {
	if len(params) < 6 {
		return nil, fmt.Errorf("LIF biological neuron requires at least 6 parameters")
	}
	
	return &lifBioNeuron{
		V:               params[0], // Initial membrane potential
		VTh:             params[1], // Threshold
		V0:              params[2], // Resting potential
		leakRate:        params[3], // Leak rate
		bias:            params[4], // Bias current
		refractoryPeriod: params[5], // Refractory period
		dt:              dt,
	}, nil
}

func (n *lifBioNeuron) update(input float64, dt float64) {
	// Update refractory timer
	if n.refractoryTimer > 0 {
		n.refractoryTimer -= dt
		n.spike = 0.0
		return
	}
	
	// Update membrane potential
	n.V += dt * (-n.leakRate*(n.V-n.V0) + input + n.bias)
	
	// Check for spike
	if n.V >= n.VTh {
		n.spike = 1.0
		n.V = n.V0 // Reset to resting potential
		n.refractoryTimer = n.refractoryPeriod
	} else {
		n.spike = 0.0
	}
}

func (n *lifBioNeuron) getState() float64 {
	return n.V
}

func (n *lifBioNeuron) getSpike() float64 {
	return n.spike
}

func (n *lifBioNeuron) reset() {
	n.V = n.V0
	n.spike = 0.0
	n.refractoryTimer = 0.0
}

// Fractional LIF with Caputo Derivative Implementation
type flifCaputoNeuron struct {
	V         float64 // Membrane potential
	VTh       float64 // Threshold potential
	V0        float64 // Resting potential
	leakRate  float64 // Leak rate
	spike     float64 // Spike output
	bias      float64 // Bias current
	alpha     float64 // Fractional order
	history   []float64 // History for fractional derivative
	maxHistory int     // Maximum history length
	dt        float64 // Time step
}

func newFLIFCaputoNeuron(params []float64, dt float64) (*flifCaputoNeuron, error) {
	if len(params) < 6 {
		return nil, fmt.Errorf("FLIF Caputo neuron requires at least 6 parameters")
	}
	
	maxHistory := 100 // Adjust based on requirements
	
	return &flifCaputoNeuron{
		V:         params[0], // Initial membrane potential
		VTh:       params[1], // Threshold
		V0:        params[2], // Resting potential
		leakRate:  params[3], // Leak rate
		bias:      params[4], // Bias current
		alpha:     params[5], // Fractional order
		history:   make([]float64, maxHistory),
		maxHistory: maxHistory,
		dt:        dt,
	}, nil
}

func (n *flifCaputoNeuron) update(input float64, dt float64) {
	// Update history (shift and add current value)
	for i := n.maxHistory - 1; i > 0; i-- {
		n.history[i] = n.history[i-1]
	}
	n.history[0] = n.V
	
	// Compute fractional derivative (simplified)
	fractionalDerivative := 0.0
	for i := 1; i < n.maxHistory; i++ {
		if n.history[i] != 0 {
			coeff := math.Pow(float64(i), -n.alpha)
			fractionalDerivative += coeff * (n.history[0] - n.history[i])
		}
	}
	
	// Update membrane potential with fractional derivative
	n.V += dt * (-n.leakRate*(n.V-n.V0) + input + n.bias + fractionalDerivative)
	
	// Check for spike
	if n.V >= n.VTh {
		n.spike = 1.0
		n.V = n.V0 // Reset to resting potential
	} else {
		n.spike = 0.0
	}
}

func (n *flifCaputoNeuron) getState() float64 {
	return n.V
}

func (n *flifCaputoNeuron) getSpike() float64 {
	return n.spike
}

func (n *flifCaputoNeuron) reset() {
	n.V = n.V0
	n.spike = 0.0
	for i := range n.history {
		n.history[i] = 0.0
	}
}

// Fractional LIF with Grünwald-Letnikov Implementation
type flifGLNeuron struct {
	V         float64 // Membrane potential
	VTh       float64 // Threshold potential
	V0        float64 // Resting potential
	leakRate  float64 // Leak rate
	spike     float64 // Spike output
	bias      float64 // Bias current
	alpha     float64 // Fractional order
	history   []float64 // History for GL derivative
	maxHistory int     // Maximum history length
	dt        float64 // Time step
}

func newFLIFGLNeuron(params []float64, dt float64) (*flifGLNeuron, error) {
	if len(params) < 6 {
		return nil, fmt.Errorf("FLIF GL neuron requires at least 6 parameters")
	}
	
	maxHistory := 100 // Adjust based on requirements
	
	return &flifGLNeuron{
		V:         params[0], // Initial membrane potential
		VTh:       params[1], // Threshold
		V0:        params[2], // Resting potential
		leakRate:  params[3], // Leak rate
		bias:      params[4], // Bias current
		alpha:     params[5], // Fractional order
		history:   make([]float64, maxHistory),
		maxHistory: maxHistory,
		dt:        dt,
	}, nil
}

func (n *flifGLNeuron) update(input float64, dt float64) {
	// Update history
	for i := n.maxHistory - 1; i > 0; i-- {
		n.history[i] = n.history[i-1]
	}
	n.history[0] = n.V
	
	// Compute Grünwald-Letnikov fractional derivative
	glDerivative := 0.0
	for i := 1; i < n.maxHistory; i++ {
		if n.history[i] != 0 {
			coeff := math.Pow(-1, float64(i)) * binomialCoeff(n.alpha, float64(i))
			glDerivative += coeff * n.history[i]
		}
	}
	glDerivative /= math.Pow(dt, n.alpha)
	
	// Update membrane potential
	n.V += dt * (-n.leakRate*(n.V-n.V0) + input + n.bias + glDerivative)
	
	// Check for spike
	if n.V >= n.VTh {
		n.spike = 1.0
		n.V = n.V0
	} else {
		n.spike = 0.0
	}
}

func (n *flifGLNeuron) getState() float64 {
	return n.V
}

func (n *flifGLNeuron) getSpike() float64 {
	return n.spike
}

func (n *flifGLNeuron) reset() {
	n.V = n.V0
	n.spike = 0.0
	for i := range n.history {
		n.history[i] = 0.0
	}
}

// Fractional LIF with Diffusive Implementation
type flifDiffusiveNeuron struct {
	V         float64 // Membrane potential
	VTh       float64 // Threshold potential
	V0        float64 // Resting potential
	leakRate  float64 // Leak rate
	spike     float64 // Spike output
	bias      float64 // Bias current
	alpha     float64 // Fractional order
	diffusionCoeff float64 // Diffusion coefficient
	dt        float64 // Time step
}

func newFLIFDiffusiveNeuron(params []float64, dt float64) (*flifDiffusiveNeuron, error) {
	if len(params) < 6 {
		return nil, fmt.Errorf("FLIF diffusive neuron requires at least 6 parameters")
	}
	
	return &flifDiffusiveNeuron{
		V:             params[0], // Initial membrane potential
		VTh:           params[1], // Threshold
		V0:            params[2], // Resting potential
		leakRate:      params[3], // Leak rate
		bias:          params[4], // Bias current
		alpha:         params[5], // Fractional order
		diffusionCoeff: params[6], // Diffusion coefficient
		dt:            dt,
	}, nil
}

func (n *flifDiffusiveNeuron) update(input float64, dt float64) {
	// Add diffusion term (simplified)
	diffusionTerm := n.diffusionCoeff * math.Sqrt(2*dt) * (rand.Float64()*2 - 1)
	
	// Update membrane potential
	n.V += dt * (-n.leakRate*(n.V-n.V0) + input + n.bias) + diffusionTerm
	
	// Check for spike
	if n.V >= n.VTh {
		n.spike = 1.0
		n.V = n.V0
	} else {
		n.spike = 0.0
	}
}

func (n *flifDiffusiveNeuron) getState() float64 {
	return n.V
}

func (n *flifDiffusiveNeuron) getSpike() float64 {
	return n.spike
}

func (n *flifDiffusiveNeuron) reset() {
	n.V = n.V0
	n.spike = 0.0
}

// Helper function for binomial coefficient
func binomialCoeff(n, k float64) float64 {
	if k == 0 {
		return 1
	}
	if k == 1 {
		return n
	}
	if k > n/2 {
		return binomialCoeff(n, n-k)
	}
	
	result := 1.0
	for i := 0; i < int(k); i++ {
		result *= (n - float64(i)) / (float64(i) + 1)
	}
	return result
}
