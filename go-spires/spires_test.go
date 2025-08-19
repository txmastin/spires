package spires

import (
	"math"
	"testing"
)

func TestStatusString(t *testing.T) {
	tests := []struct {
		status Status
		want   string
	}{
		{StatusOK, "OK"},
		{StatusErrInvalidArg, "Invalid argument"},
		{StatusErrAlloc, "Allocation error"},
		{StatusErrInternal, "Internal error"},
		{Status(99), "Unknown status"},
	}

	for _, tt := range tests {
		if got := tt.status.String(); got != tt.want {
			t.Errorf("Status(%d).String() = %v, want %v", tt.status, got, tt.want)
		}
	}
}

func TestCreateReservoir(t *testing.T) {
	// Valid configuration
	config := &ReservoirConfig{
		NumNeurons:      10,
		NumInputs:       2,
		NumOutputs:      1,
		SpectralRadius:  0.9,
		EIRatio:         0.8,
		InputStrength:   0.1,
		Connectivity:    0.1,
		Dt:              0.01,
		ConnectivityType: ConnRandom,
		NeuronType:      NeuronLIFDiscrete,
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0},
	}

	reservoir, status := Create(config)
	if status != StatusOK {
		t.Errorf("Create() failed with status: %s", status)
		return
	}
	defer reservoir.Destroy()

	if reservoir == nil {
		t.Error("Create() returned nil reservoir")
		return
	}

	// Test introspection methods
	if got := reservoir.NumNeurons(); got != config.NumNeurons {
		t.Errorf("NumNeurons() = %d, want %d", got, config.NumNeurons)
	}

	if got := reservoir.NumInputs(); got != config.NumInputs {
		t.Errorf("NumInputs() = %d, want %d", got, config.NumInputs)
	}

	if got := reservoir.NumOutputs(); got != config.NumOutputs {
		t.Errorf("NumOutputs() = %d, want %d", got, config.NumOutputs)
	}
}

func TestCreateReservoirInvalidConfig(t *testing.T) {
	// Test nil config
	_, status := Create(nil)
	if status != StatusErrInvalidArg {
		t.Errorf("Create(nil) returned status %s, want %s", status, StatusErrInvalidArg)
	}

	// Test invalid dt
	config := &ReservoirConfig{
		NumNeurons:      10,
		NumInputs:       2,
		NumOutputs:      1,
		SpectralRadius:  0.9,
		EIRatio:         0.8,
		InputStrength:   0.1,
		Connectivity:    0.1,
		Dt:              1.1, // Invalid: dt > 1.0
		ConnectivityType: ConnRandom,
		NeuronType:      NeuronLIFDiscrete,
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0},
	}

	_, status = Create(config)
	if status != StatusErrInvalidArg {
		t.Errorf("Create(invalid dt) returned status %s, want %s", status, StatusErrInvalidArg)
	}
}

func TestReservoirStep(t *testing.T) {
	config := &ReservoirConfig{
		NumNeurons:      10,
		NumInputs:       2,
		NumOutputs:      1,
		SpectralRadius:  0.9,
		EIRatio:         0.8,
		InputStrength:   0.1,
		Connectivity:    0.1,
		Dt:              0.01,
		ConnectivityType: ConnRandom,
		NeuronType:      NeuronLIFDiscrete,
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0},
	}

	reservoir, _ := Create(config)
	defer reservoir.Destroy()

	// Test step with input
	input := []float64{1.0, 0.5}
	status := reservoir.Step(input)
	if status != StatusOK {
		t.Errorf("Step() failed with status: %s", status)
	}

	// Test step with nil input (should work)
	status = reservoir.Step(nil)
	if status != StatusOK {
		t.Errorf("Step(nil) failed with status: %s", status)
	}
}

func TestReservoirRun(t *testing.T) {
	config := &ReservoirConfig{
		NumNeurons:      10,
		NumInputs:       2,
		NumOutputs:      1,
		SpectralRadius:  0.9,
		EIRatio:         0.8,
		InputStrength:   0.1,
		Connectivity:    0.1,
		Dt:              0.01,
		ConnectivityType: ConnRandom,
		NeuronType:      NeuronLIFDiscrete,
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0},
	}

	reservoir, _ := Create(config)
	defer reservoir.Destroy()

	// Test run with input series
	inputSeries := []float64{1.0, 0.0, 0.0, 1.0, 1.0, 1.0} // 3 time steps, 2 inputs each
	seriesLength := 3

	outputs, status := reservoir.Run(inputSeries, seriesLength)
	if status != StatusOK {
		t.Errorf("Run() failed with status: %s", status)
		return
	}

	expectedLength := config.NumOutputs * seriesLength
	if len(outputs) != expectedLength {
		t.Errorf("Run() returned %d outputs, want %d", len(outputs), expectedLength)
	}
}

func TestReservoirReset(t *testing.T) {
	config := &ReservoirConfig{
		NumNeurons:      10,
		NumInputs:       2,
		NumOutputs:      1,
		SpectralRadius:  0.9,
		EIRatio:         0.8,
		InputStrength:   0.1,
		Connectivity:    0.1,
		Dt:              0.01,
		ConnectivityType: ConnRandom,
		NeuronType:      NeuronLIFDiscrete,
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0},
	}

	reservoir, _ := Create(config)
	defer reservoir.Destroy()

	// Step the reservoir
	input := []float64{1.0, 0.5}
	reservoir.Step(input)

	// Reset
	status := reservoir.Reset()
	if status != StatusOK {
		t.Errorf("Reset() failed with status: %s", status)
	}
}

func TestReservoirTraining(t *testing.T) {
	config := &ReservoirConfig{
		NumNeurons:      10,
		NumInputs:       2,
		NumOutputs:      1,
		SpectralRadius:  0.9,
		EIRatio:         0.8,
		InputStrength:   0.1,
		Connectivity:    0.1,
		Dt:              0.01,
		ConnectivityType: ConnRandom,
		NeuronType:      NeuronLIFDiscrete,
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0},
	}

	reservoir, _ := Create(config)
	defer reservoir.Destroy()

	// Test online training
	target := []float64{0.5}
	learningRate := 0.01

	status := reservoir.TrainOnline(target, learningRate)
	if status != StatusOK {
		t.Errorf("TrainOnline() failed with status: %s", status)
	}

	// Test ridge regression training
	inputSeries := []float64{1.0, 0.0, 0.0, 1.0, 1.0, 1.0}
	targetSeries := []float64{0.5, 0.3, 0.7}
	seriesLength := 3
	lambda := 0.01

	status = reservoir.TrainRidge(inputSeries, targetSeries, seriesLength, lambda)
	if status != StatusOK {
		t.Errorf("TrainRidge() failed with status: %s", status)
	}
}

func TestReservoirStateAccess(t *testing.T) {
	config := &ReservoirConfig{
		NumNeurons:      10,
		NumInputs:       2,
		NumOutputs:      1,
		SpectralRadius:  0.9,
		EIRatio:         0.8,
		InputStrength:   0.1,
		Connectivity:    0.1,
		Dt:              0.01,
		ConnectivityType: ConnRandom,
		NeuronType:      NeuronLIFDiscrete,
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0},
	}

	reservoir, _ := Create(config)
	defer reservoir.Destroy()

	// Test state reading
	state, status := reservoir.ReadStateCopy()
	if status != StatusOK {
		t.Errorf("ReadStateCopy() failed with status: %s", status)
		return
	}

	if len(state) != config.NumNeurons {
		t.Errorf("ReadStateCopy() returned %d states, want %d", len(state), config.NumNeurons)
	}

	// Test output computation
	output := make([]float64, config.NumOutputs)
	status = reservoir.ComputeOutput(output)
	if status != StatusOK {
		t.Errorf("ComputeOutput() failed with status: %s", status)
		return
	}

	if len(output) != config.NumOutputs {
		t.Errorf("ComputeOutput() returned %d outputs, want %d", len(output), config.NumOutputs)
	}
}

func TestNeuronCreation(t *testing.T) {
	// Test LIF discrete neuron
	params := []float64{-65.0, -55.0, -65.0, 0.1, 0.0}
	neuron, err := initNeuron(NeuronLIFDiscrete, params, 0.01)
	if err != nil {
		t.Errorf("Failed to create LIF discrete neuron: %v", err)
		return
	}

	if neuron == nil {
		t.Error("initNeuron() returned nil neuron")
		return
	}

	// Test neuron interface methods
	neuron.update(1.0, 0.01)
	state := neuron.getState()
	spike := neuron.getSpike()
	
	// Basic sanity checks
	if math.IsNaN(state) || math.IsInf(state, 0) {
		t.Error("Neuron state is NaN or Inf")
	}
	
	if spike != 0.0 && spike != 1.0 {
		t.Errorf("Neuron spike should be 0 or 1, got %f", spike)
	}

	neuron.reset()
}

func TestMathUtils(t *testing.T) {
	// Test matrix-vector multiplication
	A := []float64{1.0, 2.0, 3.0, 4.0} // 2x2 matrix
	x := []float64{1.0, 2.0}
	y := make([]float64, 2)

	MatVecMult(A, x, y, 2, 2)

	expected := []float64{5.0, 11.0} // [1*1+2*2, 3*1+4*2]
	for i, val := range y {
		if math.Abs(val-expected[i]) > 1e-10 {
			t.Errorf("MatVecMult result[%d] = %f, want %f", i, val, expected[i])
		}
	}

	// Test vector operations
	a := []float64{1.0, 2.0, 3.0}
	b := []float64{4.0, 5.0, 6.0}
	result := make([]float64, 3)

	VectorAdd(a, b, result)
	expectedAdd := []float64{5.0, 7.0, 9.0}
	for i, val := range result {
		if math.Abs(val-expectedAdd[i]) > 1e-10 {
			t.Errorf("VectorAdd result[%d] = %f, want %f", i, val, expectedAdd[i])
		}
	}

	dot := VectorDot(a, b)
	expectedDot := 1.0*4.0 + 2.0*5.0 + 3.0*6.0
	if math.Abs(dot-expectedDot) > 1e-10 {
		t.Errorf("VectorDot = %f, want %f", dot, expectedDot)
	}
}
