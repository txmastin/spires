package spires

import (
	"math"
)

// Status codes for operations
type Status int

const (
	StatusOK Status = iota
	StatusErrInvalidArg
	StatusErrAlloc
	StatusErrInternal
)

func (s Status) String() string {
	switch s {
	case StatusOK:
		return "OK"
	case StatusErrInvalidArg:
		return "Invalid argument"
	case StatusErrAlloc:
		return "Allocation error"
	case StatusErrInternal:
		return "Internal error"
	default:
		return "Unknown status"
	}
}

// Connectivity types for the reservoir
type ConnectivityType int

const (
	ConnRandom ConnectivityType = iota
	ConnSmallWorld
	ConnScaleFree
)

// Neuron types supported by the library
type NeuronType int

const (
	NeuronLIFDiscrete NeuronType = iota
	NeuronLIFBio
	NeuronFLIFCaputo
	NeuronFLIFGL
	NeuronFLIFDiffusive
)

// Configuration for creating a reservoir
type ReservoirConfig struct {
	NumNeurons        int
	NumInputs         int
	NumOutputs        int
	SpectralRadius    float64
	EIRatio           float64
	InputStrength     float64
	Connectivity      float64
	Dt                float64
	ConnectivityType  ConnectivityType
	NeuronType        NeuronType
	NeuronParams      []float64
}

// Reservoir represents a spiking neural reservoir
type Reservoir struct {
	impl *reservoir
}

// Create creates a new reservoir with the given configuration
func Create(cfg *ReservoirConfig) (*Reservoir, Status) {
	if cfg == nil {
		return nil, StatusErrInvalidArg
	}

	// Validate dt (must evenly divide 1.0)
	if cfg.Dt <= 0.0 || cfg.Dt > 1.0 {
		return nil, StatusErrInvalidArg
	}
	stepsD := 1.0 / cfg.Dt
	steps := int(math.Round(stepsD))
	if steps < 1 || math.Abs(stepsD-float64(steps)) > 1e-12 {
		return nil, StatusErrInvalidArg
	}

	impl, err := createReservoir(
		cfg.NumNeurons,
		cfg.NumInputs,
		cfg.NumOutputs,
		cfg.SpectralRadius,
		cfg.EIRatio,
		cfg.InputStrength,
		cfg.Connectivity,
		cfg.Dt,
		cfg.ConnectivityType,
		cfg.NeuronType,
		cfg.NeuronParams,
	)
	if err != nil {
		return nil, StatusErrInternal
	}

	if err := initReservoir(impl); err != nil {
		freeReservoir(impl)
		return nil, StatusErrInternal
	}

	return &Reservoir{impl: impl}, StatusOK
}

// Destroy frees the resources associated with the reservoir
func (r *Reservoir) Destroy() {
	if r != nil && r.impl != nil {
		freeReservoir(r.impl)
		r.impl = nil
	}
}

// Reset resets the reservoir to its initial state
func (r *Reservoir) Reset() Status {
	if r == nil || r.impl == nil {
		return StatusErrInvalidArg
	}
	resetReservoir(r.impl)
	return StatusOK
}

// Step advances the reservoir by one time step with the given input
func (r *Reservoir) Step(u []float64) Status {
	if r == nil || r.impl == nil {
		return StatusErrInvalidArg
	}

	var input []float64
	if u != nil {
		input = make([]float64, len(u))
		copy(input, u)
	}
	stepReservoir(r.impl, input)
	return StatusOK
}

// Run runs the reservoir on a series of inputs
func (r *Reservoir) Run(inputSeries []float64, seriesLength int) ([]float64, Status) {
	if r == nil || r.impl == nil || inputSeries == nil {
		return nil, StatusErrInvalidArg
	}
	
	outputs := runReservoir(r.impl, inputSeries, seriesLength)
	if outputs == nil {
		return nil, StatusErrInternal
	}
	return outputs, StatusOK
}

// TrainOnline performs online training with a target vector
func (r *Reservoir) TrainOnline(targetVec []float64, lr float64) Status {
	if r == nil || r.impl == nil || targetVec == nil {
		return StatusErrInvalidArg
	}
	trainOutputIteratively(r.impl, targetVec, lr)
	return StatusOK
}

// TrainRidge performs ridge regression training
func (r *Reservoir) TrainRidge(inputSeries, targetSeries []float64, seriesLength int, lambda float64) Status {
	if r == nil || r.impl == nil || inputSeries == nil || targetSeries == nil {
		return StatusErrInvalidArg
	}
	trainOutputRidgeRegression(r.impl, inputSeries, targetSeries, seriesLength, lambda)
	return StatusOK
}

// ReadStateCopy returns a copy of the current neuron states
func (r *Reservoir) ReadStateCopy() ([]float64, Status) {
	if r == nil || r.impl == nil {
		return nil, StatusErrInvalidArg
	}
	state := readReservoirState(r.impl)
	if state == nil {
		return nil, StatusErrInternal
	}
	return state, StatusOK
}

// ComputeOutput computes the current output
func (r *Reservoir) ComputeOutput(out []float64) Status {
	if r == nil || r.impl == nil || out == nil {
		return StatusErrInvalidArg
	}
	if err := computeOutput(r.impl, out); err != nil {
		return StatusErrInternal
	}
	return StatusOK
}

// NumNeurons returns the number of neurons in the reservoir
func (r *Reservoir) NumNeurons() int {
	if r == nil || r.impl == nil {
		return 0
	}
	return r.impl.numNeurons
}

// NumInputs returns the number of inputs to the reservoir
func (r *Reservoir) NumInputs() int {
	if r == nil || r.impl == nil {
		return 0
	}
	return r.impl.numInputs
}

// NumOutputs returns the number of outputs from the reservoir
func (r *Reservoir) NumOutputs() int {
	if r == nil || r.impl == nil {
		return 0
	}
	return r.impl.numOutputs
}
