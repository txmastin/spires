package spires

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// Memory pool for temporary arrays to reduce GC pressure
var (
	inputBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]float64, 0, 1000) // Pre-allocate with capacity
		},
	}
	stateBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]float64, 0, 1000)
		},
	}
	outputBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]float64, 0, 1000)
		},
	}
	tempBufferPool = sync.Pool{
		New: func() interface{} {
			return make([]float64, 0, 1000)
		},
	}
)

// Internal reservoir structure
type reservoir struct {
	neurons           []neuron
	numNeurons        int
	numInputs         int
	numOutputs        int
	spectralRadius    float64
	eiRatio           float64
	inputStrength     float64
	connectivity      float64
	dt                float64
	connectivityType  ConnectivityType
	neuronType        NeuronType
	neuronParams      []float64
	WIn               []float64
	WOut              []float64
	W                 []float64
	
	// Pre-allocated buffers for performance
	inputBuffer       []float64
	stateBuffer       []float64
	outputBuffer      []float64
	tempBuffer1       []float64
	tempBuffer2       []float64
	tempBuffer3       []float64
	
	// Parallel processing settings
	numWorkers        int
	useParallel       bool
}

// neuron interface for different neuron types
type neuron interface {
	update(input float64, dt float64)
	getState() float64
	getSpike() float64
	reset()
}

// createReservoir creates a new reservoir instance
func createReservoir(numNeurons, numInputs, numOutputs int,
	spectralRadius, eiRatio, inputStrength, connectivity, dt float64,
	connectivityType ConnectivityType, neuronType NeuronType, neuronParams []float64) (*reservoir, error) {

	// Determine number of workers for parallel processing
	numWorkers := runtime.NumCPU()
	useParallel := numNeurons > 100 && numWorkers > 1 // Only use parallel for larger reservoirs

	r := &reservoir{
		numNeurons:       numNeurons,
		numInputs:        numInputs,
		numOutputs:       numOutputs,
		spectralRadius:   spectralRadius,
		eiRatio:          eiRatio,
		inputStrength:    inputStrength,
		connectivity:     connectivity,
		dt:               dt,
		connectivityType: connectivityType,
		neuronType:       neuronType,
		neuronParams:     make([]float64, len(neuronParams)),
		WIn:              make([]float64, numNeurons*numInputs),
		WOut:             make([]float64, numOutputs*numNeurons),
		W:                make([]float64, numNeurons*numNeurons),
		neurons:          make([]neuron, numNeurons),
		
		// Pre-allocate buffers
		inputBuffer:      make([]float64, numInputs),
		stateBuffer:      make([]float64, numNeurons),
		outputBuffer:     make([]float64, numOutputs),
		tempBuffer1:      make([]float64, numNeurons),
		tempBuffer2:      make([]float64, numNeurons),
		tempBuffer3:      make([]float64, numNeurons),
		
		// Parallel processing settings
		numWorkers:       numWorkers,
		useParallel:      useParallel,
	}

	// Copy neuron parameters
	copy(r.neuronParams, neuronParams)

	// Initialize neurons
	for i := 0; i < numNeurons; i++ {
		neuron, err := initNeuron(neuronType, neuronParams, dt)
		if err != nil {
			return nil, fmt.Errorf("failed to initialize neuron %d: %w", i, err)
		}
		r.neurons[i] = neuron
	}

	return r, nil
}

// initReservoir initializes the reservoir weights and connections
func initReservoir(r *reservoir) error {
	if r == nil {
		return fmt.Errorf("reservoir not created")
	}

	if err := initWeights(r); err != nil {
		return fmt.Errorf("failed to initialize weights: %w", err)
	}
	if err := rescaleWeights(r); err != nil {
		return fmt.Errorf("failed to rescale weights: %w", err)
	}
	if err := randomizeOutputLayer(r); err != nil {
		return fmt.Errorf("failed to randomize output layer: %w", err)
	}

	return nil
}

// initWeights initializes the weight matrices
func initWeights(r *reservoir) error {
	// Initialize input weights
	for i := 0; i < r.numNeurons; i++ {
		for j := 0; j < r.numInputs; j++ {
			r.WIn[i*r.numInputs+j] = (rand.Float64()*2 - 1) * r.inputStrength
		}
	}

	// Initialize recurrent weights based on connectivity type
	switch r.connectivityType {
	case ConnRandom:
		initRandomWeights(r)
	case ConnSmallWorld:
		initSmallWorldWeights(r)
	case ConnScaleFree:
		initScaleFreeWeights(r)
	default:
		return fmt.Errorf("unknown connectivity type: %d", r.connectivityType)
	}

	// Initialize output weights (will be trained)
	for i := 0; i < r.numOutputs; i++ {
		for j := 0; j < r.numNeurons; j++ {
			r.WOut[i*r.numNeurons+j] = 0.0
		}
	}

	return nil
}

// initRandomWeights creates random connections
func initRandomWeights(r *reservoir) {
	for i := 0; i < r.numNeurons; i++ {
		for j := 0; j < r.numNeurons; j++ {
			if i != j && rand.Float64() < r.connectivity {
				r.W[i*r.numNeurons+j] = (rand.Float64()*2 - 1)
			} else {
				r.W[i*r.numNeurons+j] = 0.0
			}
		}
	}
}

// initSmallWorldWeights creates small-world network connections
func initSmallWorldWeights(r *reservoir) {
	// Start with a regular ring lattice
	k := int(r.connectivity * float64(r.numNeurons) / 2)
	for i := 0; i < r.numNeurons; i++ {
		for j := 1; j <= k; j++ {
			idx1 := (i + j) % r.numNeurons
			idx2 := (i - j + r.numNeurons) % r.numNeurons
			r.W[i*r.numNeurons+idx1] = (rand.Float64()*2 - 1)
			r.W[i*r.numNeurons+idx2] = (rand.Float64()*2 - 1)
		}
	}

	// Rewire with some probability
	for i := 0; i < r.numNeurons; i++ {
		for j := 0; j < r.numNeurons; j++ {
			if r.W[i*r.numNeurons+j] != 0 && rand.Float64() < 0.1 {
				r.W[i*r.numNeurons+j] = 0.0
				newTarget := rand.Intn(r.numNeurons)
				if newTarget != i {
					r.W[i*r.numNeurons+newTarget] = (rand.Float64()*2 - 1)
				}
			}
		}
	}
}

// initScaleFreeWeights creates scale-free network connections
func initScaleFreeWeights(r *reservoir) {
	// Preferential attachment model
	degrees := make([]int, r.numNeurons)
	
	// Start with a few connected nodes
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			r.W[i*r.numNeurons+j] = (rand.Float64()*2 - 1)
			r.W[j*r.numNeurons+i] = (rand.Float64()*2 - 1)
			degrees[i]++
			degrees[j]++
		}
	}

	// Add remaining nodes with preferential attachment
	for i := 3; i < r.numNeurons; i++ {
		// Connect to existing nodes with probability proportional to degree
		totalDegree := 0
		for _, d := range degrees {
			totalDegree += d
		}

		for j := 0; j < i; j++ {
			if rand.Float64() < float64(degrees[j])/float64(totalDegree)*r.connectivity {
				r.W[i*r.numNeurons+j] = (rand.Float64()*2 - 1)
				r.W[j*r.numNeurons+i] = (rand.Float64()*2 - 1)
				degrees[i]++
				degrees[j]++
			}
		}
	}
}

// rescaleWeights rescales the recurrent weights to achieve the desired spectral radius
func rescaleWeights(r *reservoir) error {
	currentRho := calcSpectralRadius(r.W, r.numNeurons)
	if currentRho > 0 {
		scale := r.spectralRadius / currentRho
		for i := 0; i < r.numNeurons*r.numNeurons; i++ {
			r.W[i] *= scale
		}
	}
	return nil
}

// randomizeOutputLayer initializes random output weights
func randomizeOutputLayer(r *reservoir) error {
	for i := 0; i < r.numOutputs*r.numNeurons; i++ {
		r.WOut[i] = (rand.Float64()*2 - 1) * 0.1
	}
	return nil
}

// stepReservoir advances the reservoir by one time step
func stepReservoir(r *reservoir, input []float64) {
	// Prepare input vector using pre-allocated buffer
	if input != nil {
		copy(r.inputBuffer, input)
	} else {
		// Zero the input buffer
		for i := range r.inputBuffer {
			r.inputBuffer[i] = 0.0
		}
	}

	// Compute input contribution: W_in * u using pre-allocated buffer
	MatVecMult(r.WIn, r.inputBuffer, r.tempBuffer1, r.numNeurons, r.numInputs)

	// Get current neuron states using pre-allocated buffer
	if r.useParallel {
		r.getNeuronStatesParallel()
	} else {
		r.getNeuronStatesSequential()
	}
	
	// Compute recurrent contribution: W * x using pre-allocated buffer
	MatVecMult(r.W, r.stateBuffer, r.tempBuffer2, r.numNeurons, r.numNeurons)

	// Update each neuron using pre-allocated buffer for total input
	if r.useParallel {
		r.updateNeuronsParallel()
	} else {
		r.updateNeuronsSequential()
	}
}

// getNeuronStatesSequential gets neuron states sequentially
func (r *reservoir) getNeuronStatesSequential() {
	for i := 0; i < r.numNeurons; i++ {
		r.stateBuffer[i] = r.neurons[i].getState()
	}
}

// getNeuronStatesParallel gets neuron states in parallel
func (r *reservoir) getNeuronStatesParallel() {
	var wg sync.WaitGroup
	neuronsPerWorker := r.numNeurons / r.numWorkers
	
	for w := 0; w < r.numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * neuronsPerWorker
			end := start + neuronsPerWorker
			if workerID == r.numWorkers-1 {
				end = r.numNeurons
			}
			
			for i := start; i < end; i++ {
				r.stateBuffer[i] = r.neurons[i].getState()
			}
		}(w)
	}
	wg.Wait()
}

// updateNeuronsSequential updates neurons sequentially
func (r *reservoir) updateNeuronsSequential() {
	for i := 0; i < r.numNeurons; i++ {
		r.tempBuffer3[i] = r.tempBuffer1[i] + r.tempBuffer2[i]
		r.neurons[i].update(r.tempBuffer3[i], r.dt)
	}
}

// updateNeuronsParallel updates neurons in parallel
func (r *reservoir) updateNeuronsParallel() {
	var wg sync.WaitGroup
	neuronsPerWorker := r.numNeurons / r.numWorkers
	
	for w := 0; w < r.numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * neuronsPerWorker
			end := start + neuronsPerWorker
			if workerID == r.numWorkers-1 {
				end = r.numNeurons
			}
			
			for i := start; i < end; i++ {
				r.tempBuffer3[i] = r.tempBuffer1[i] + r.tempBuffer2[i]
				r.neurons[i].update(r.tempBuffer3[i], r.dt)
			}
		}(w)
	}
	wg.Wait()
}

// runReservoir runs the reservoir on a series of inputs
func runReservoir(r *reservoir, inputSeries []float64, seriesLength int) []float64 {
	if r == nil || r.dt <= 0.0 {
		return nil
	}

	numInputs := r.numInputs
	numOutputs := r.numOutputs

	outputSeries := make([]float64, numOutputs*seriesLength)

	for i := 0; i < seriesLength; i++ {
		currentInput := inputSeries[i*numInputs : (i+1)*numInputs]
		stepReservoir(r, currentInput)

		currentOutput := outputSeries[i*numOutputs : (i+1)*numOutputs]
		computeOutput(r, currentOutput)
	}

	return outputSeries
}

// computeOutput computes the current output
func computeOutput(r *reservoir, out []float64) error {
	// Get current neuron states using pre-allocated buffer
	if r.useParallel {
		r.getNeuronStatesParallel()
	} else {
		r.getNeuronStatesSequential()
	}

	// Compute output: W_out * state using pre-allocated buffer
	MatVecMult(r.WOut, r.stateBuffer, out, r.numOutputs, r.numNeurons)
	return nil
}

// readReservoirState returns the current neuron states
func readReservoirState(r *reservoir) []float64 {
	// Use pre-allocated buffer and copy to avoid external modifications
	state := make([]float64, r.numNeurons)
	for i := 0; i < r.numNeurons; i++ {
		state[i] = r.neurons[i].getState()
	}
	return state
}

// trainOutputIteratively performs online training
func trainOutputIteratively(r *reservoir, targetVec []float64, lr float64) {
	// Get current output using pre-allocated buffer
	computeOutput(r, r.outputBuffer)

	// Get current state using pre-allocated buffer
	if r.useParallel {
		r.getNeuronStatesParallel()
	} else {
		r.getNeuronStatesSequential()
	}

	// Update output weights: W_out += lr * (target - output) * state^T
	for i := 0; i < r.numOutputs; i++ {
		error := targetVec[i] - r.outputBuffer[i]
		for j := 0; j < r.numNeurons; j++ {
			r.WOut[i*r.numNeurons+j] += lr * error * r.stateBuffer[j]
		}
	}
}

// trainOutputRidgeRegression performs ridge regression training
func trainOutputRidgeRegression(r *reservoir, inputSeries, targetSeries []float64, seriesLength int, lambda float64) {
	// Collect states for all time steps
	states := make([][]float64, seriesLength)
	for t := 0; t < seriesLength; t++ {
		// Reset and run up to time t
		resetReservoir(r)
		for i := 0; i <= t; i++ {
			currentInput := inputSeries[i*r.numInputs : (i+1)*r.numInputs]
			stepReservoir(r, currentInput)
		}
		states[t] = readReservoirState(r)
	}

	// Solve ridge regression: W_out = (X^T * X + λI)^(-1) * X^T * Y
	// where X is the state matrix and Y is the target matrix
	solveRidgeRegression(r, states, targetSeries, seriesLength, lambda)
}

// solveRidgeRegression solves the ridge regression problem
func solveRidgeRegression(r *reservoir, states [][]float64, targets []float64, seriesLength int, lambda float64) {
	// This is a simplified implementation
	// In practice, you might want to use a more efficient solver
	
	// For now, we'll use a simple iterative approach
	// This could be optimized with proper matrix operations
	
	// Initialize W_out with small random values
	for i := 0; i < r.numOutputs*r.numNeurons; i++ {
		r.WOut[i] = (rand.Float64()*2 - 1) * 0.01
	}

	// Simple gradient descent
	lr := 0.001
	epochs := 1000
	
	for epoch := 0; epoch < epochs; epoch++ {
		for t := 0; t < seriesLength; t++ {
			// Compute current output using pre-allocated buffer
			MatVecMult(r.WOut, states[t], r.outputBuffer, r.numOutputs, r.numNeurons)
			
			// Compute error
			error := targets[t] - r.outputBuffer[0] // Assuming single output for simplicity
			
			// Update weights
			for j := 0; j < r.numNeurons; j++ {
				r.WOut[j] += lr * (error*states[t][j] - lambda*r.WOut[j])
			}
		}
	}
}

// resetReservoir resets all neurons to their initial state
func resetReservoir(r *reservoir) {
	if r.useParallel {
		r.resetNeuronsParallel()
	} else {
		r.resetNeuronsSequential()
	}
}

// resetNeuronsSequential resets neurons sequentially
func (r *reservoir) resetNeuronsSequential() {
	for i := 0; i < r.numNeurons; i++ {
		r.neurons[i].reset()
	}
}

// resetNeuronsParallel resets neurons in parallel
func (r *reservoir) resetNeuronsParallel() {
	var wg sync.WaitGroup
	neuronsPerWorker := r.numNeurons / r.numWorkers
	
	for w := 0; w < r.numWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			start := workerID * neuronsPerWorker
			end := start + neuronsPerWorker
			if workerID == r.numWorkers-1 {
				end = r.numNeurons
			}
			
			for i := start; i < end; i++ {
				r.neurons[i].reset()
			}
		}(w)
	}
	wg.Wait()
}

// freeReservoir frees the reservoir resources
func freeReservoir(r *reservoir) {
	if r != nil {
		r.neurons = nil
		r.WIn = nil
		r.WOut = nil
		r.W = nil
		r.neuronParams = nil
		
		// Clear pre-allocated buffers
		r.inputBuffer = nil
		r.stateBuffer = nil
		r.outputBuffer = nil
		r.tempBuffer1 = nil
		r.tempBuffer2 = nil
		r.tempBuffer3 = nil
	}
}

// matVecMult performs matrix-vector multiplication: y = A * x
func matVecMult(A, x, y []float64, rows, cols int) {
	for i := 0; i < rows; i++ {
		y[i] = 0
		for j := 0; j < cols; j++ {
			y[i] += A[i*cols+j] * x[j]
		}
	}
}

// calcSpectralRadius calculates the spectral radius of a matrix
func calcSpectralRadius(A []float64, n int) float64 {
	// Use the optimized version from math_utils.go
	return CalcSpectralRadius(A, n)
}
