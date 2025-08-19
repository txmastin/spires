# SPIRES - Go Implementation

A Go implementation of the SPIRES (Spiking Neural Reservoir) library for neuromorphic computing and reservoir computing applications.

## Overview

SPIRES is a library for creating and simulating spiking neural reservoirs with various neuron models and connectivity patterns. This Go implementation provides a clean, idiomatic Go API while maintaining the core functionality of the original C library.

## Features

- **Multiple Neuron Types**: Support for LIF (Leaky Integrate-and-Fire) neurons, biological neurons, and fractional LIF neurons
- **Connectivity Patterns**: Random, small-world, and scale-free network topologies
- **Training Methods**: Online learning and ridge regression training
- **Memory Management**: Automatic memory management with Go's garbage collector
- **Type Safety**: Strong typing and error handling throughout the API

## Installation

```bash
go get github.com/spires/spires
```

## Dependencies

- Go 1.21 or later
- `gonum.org/v1/gonum` for advanced mathematical operations

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/spires/spires"
)

func main() {
    // Create reservoir configuration
    config := &spires.ReservoirConfig{
        NumNeurons:     100,
        NumInputs:      2,
        NumOutputs:     1,
        SpectralRadius: 0.9,
        EIRatio:        0.8,
        InputStrength:  0.1,
        Connectivity:   0.1,
        Dt:             0.01,
        ConnectivityType: spires.ConnRandom,
        NeuronType:      spires.NeuronLIFDiscrete,
        NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0}, // V, VTh, V0, leakRate, bias
    }

    // Create reservoir
    reservoir, status := spires.Create(config)
    if status != spires.StatusOK {
        fmt.Printf("Failed to create reservoir: %s\n", status)
        return
    }
    defer reservoir.Destroy()

    // Create input sequence
    inputSeries := []float64{1.0, 0.0, 0.0, 1.0, 0.0, 0.0} // 3 time steps, 2 inputs each
    seriesLength := 3

    // Run reservoir
    outputs, status := reservoir.Run(inputSeries, seriesLength)
    if status != spires.StatusOK {
        fmt.Printf("Failed to run reservoir: %s\n", status)
        return
    }

    fmt.Printf("Outputs: %v\n", outputs)
}
```

## API Reference

### Core Types

#### ReservoirConfig

Configuration structure for creating reservoirs:

```go
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
```

#### Reservoir

Main reservoir interface:

```go
type Reservoir struct {
    // ... implementation details
}
```

### Main Functions

#### Create

Creates a new reservoir:

```go
func Create(cfg *ReservoirConfig) (*Reservoir, Status)
```

#### Step

Advances the reservoir by one time step:

```go
func (r *Reservoir) Step(u []float64) Status
```

#### Run

Runs the reservoir on a series of inputs:

```go
func (r *Reservoir) Run(inputSeries []float64, seriesLength int) ([]float64, Status)
```

#### TrainOnline

Performs online training:

```go
func (r *Reservoir) TrainOnline(targetVec []float64, lr float64) Status
```

#### TrainRidge

Performs ridge regression training:

```go
func (r *Reservoir) TrainRidge(inputSeries, targetSeries []float64, seriesLength int, lambda float64) Status
```

### Neuron Types

- `NeuronLIFDiscrete`: Basic LIF neuron
- `NeuronLIFBio`: Biological LIF neuron with refractory period
- `NeuronFLIFCaputo`: Fractional LIF with Caputo derivative
- `NeuronFLIFGL`: Fractional LIF with Grünwald-Letnikov derivative
- `NeuronFLIFDiffusive`: Fractional LIF with diffusion term

### Connectivity Types

- `ConnRandom`: Random network topology
- `ConnSmallWorld`: Small-world network topology
- `ConnScaleFree`: Scale-free network topology

## Examples

### Basic Usage

See the `examples/` directory for complete working examples.

### Training a Reservoir

```go
// Create training data
inputData := []float64{/* your input data */}
targetData := []float64{/* your target data */}

// Train using ridge regression
lambda := 0.01
status := reservoir.TrainRidge(inputData, targetData, len(inputData)/2, lambda)
if status != spires.StatusOK {
    fmt.Printf("Training failed: %s\n", status)
}
```

### Online Learning

```go
// Single step training
target := []float64{0.5}
learningRate := 0.01
status := reservoir.TrainOnline(target, learningRate)
```

## Performance Considerations

- The library uses Go's built-in memory management
- Matrix operations are optimized for the specific use cases
- Consider using Go's profiling tools for performance analysis
- For large-scale simulations, consider using goroutines for parallel processing

## Error Handling

All functions return a `Status` value indicating success or failure:

- `StatusOK`: Operation completed successfully
- `StatusErrInvalidArg`: Invalid argument provided
- `StatusErrAlloc`: Memory allocation failed
- `StatusErrInternal`: Internal error occurred

## Contributing

Contributions are welcome! Please ensure:

- Code follows Go conventions
- Tests are included for new functionality
- Documentation is updated

## License

This project is licensed under the same terms as the original SPIRES library.

## Acknowledgments

This Go implementation is based on the original C SPIRES library, adapted to provide a modern, idiomatic Go API for spiking neural reservoir computing.
