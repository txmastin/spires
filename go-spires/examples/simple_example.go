package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/spires/spires"
)

func main() {
	// Seed random number generator
	rand.Seed(time.Now().UnixNano())

	fmt.Println("SPIRES Go Implementation - Simple Example")
	fmt.Println("========================================")

	// Create reservoir configuration
	config := &spires.ReservoirConfig{
		NumNeurons:      50,  // Small reservoir for demonstration
		NumInputs:       2,   // 2 input channels
		NumOutputs:      1,   // 1 output
		SpectralRadius:  0.9, // Critical for reservoir dynamics
		EIRatio:         0.8, // Excitatory-inhibitory ratio
		InputStrength:   0.1, // Input weight strength
		Connectivity:    0.1, // 10% connection density
		Dt:              0.01, // Time step
		ConnectivityType: spires.ConnRandom,
		NeuronType:      spires.NeuronLIFBio, // Changed to biological LIF
		NeuronParams:    []float64{-65.0, -55.0, -65.0, 0.1, 0.0, 0.005}, // Added refractory period
	}

	fmt.Printf("Creating reservoir with %d neurons...\n", config.NumNeurons)

	// Create reservoir
	reservoir, status := spires.Create(config)
	if status != spires.StatusOK {
		fmt.Printf("Failed to create reservoir: %s\n", status)
		return
	}
	defer reservoir.Destroy()

	fmt.Println("Reservoir created successfully!")

	// Create a simple input sequence (3 time steps, 2 inputs each)
	inputSeries := []float64{
		1.0, 0.0, // Time step 1: [1, 0]
		0.0, 1.0, // Time step 2: [0, 1]
		1.0, 1.0, // Time step 3: [1, 1]
	}
	seriesLength := 3

	fmt.Printf("Running reservoir on input sequence...\n")
	fmt.Printf("Input series: %v\n", inputSeries)

	// Run reservoir
	outputs, status := reservoir.Run(inputSeries, seriesLength)
	if status != spires.StatusOK {
		fmt.Printf("Failed to run reservoir: %s\n", status)
		return
	}

	fmt.Printf("Outputs: %v\n", outputs)

	// Demonstrate single step operation
	fmt.Println("\nDemonstrating single step operation...")
	
	// Reset reservoir
	reservoir.Reset()
	
	// Step through inputs one by one
	for i := 0; i < seriesLength; i++ {
		input := inputSeries[i*2 : (i+1)*2]
		fmt.Printf("Step %d: Input = %v\n", i+1, input)
		
		status := reservoir.Step(input)
		if status != spires.StatusOK {
			fmt.Printf("Step failed: %s\n", status)
			continue
		}
		
		// Get current output
		output := make([]float64, config.NumOutputs)
		status = reservoir.ComputeOutput(output)
		if status != spires.StatusOK {
			fmt.Printf("Output computation failed: %s\n", status)
			continue
		}
		
		fmt.Printf("  Output = %v\n", output)
		
		// Get reservoir state
		state, status := reservoir.ReadStateCopy()
		if status == spires.StatusOK {
			// Calculate average membrane potential
			sum := 0.0
			for _, v := range state {
				sum += v
			}
			avgPotential := sum / float64(len(state))
			fmt.Printf("  Average membrane potential = %.3f\n", avgPotential)
		}
	}

	// Demonstrate reservoir behavior without training
	fmt.Println("\nDemonstrating reservoir dynamics...")
	
	// Reset reservoir
	reservoir.Reset()
	
	// Show how reservoir responds to different input patterns
	testInputs := [][]float64{
		{1.0, 0.0},  // Only first input active
		{0.0, 1.0},  // Only second input active
		{1.0, 1.0},  // Both inputs active
		{0.5, 0.5},  // Moderate inputs
		{0.0, 0.0},  // No input
	}
	
	for i, input := range testInputs {
		fmt.Printf("Test %d: Input = %v\n", i+1, input)
		
		// Step the reservoir
		status := reservoir.Step(input)
		if status != spires.StatusOK {
			fmt.Printf("  Step failed: %s\n", status)
			continue
		}
		
		// Get current output
		output := make([]float64, config.NumOutputs)
		status = reservoir.ComputeOutput(output)
		if status != spires.StatusOK {
			fmt.Printf("  Output computation failed: %s\n", status)
			continue
		}
		
		fmt.Printf("  Output = %v\n", output)
		
		// Get reservoir state
		state, status := reservoir.ReadStateCopy()
		if status == spires.StatusOK {
			// Calculate average membrane potential
			sum := 0.0
			for _, v := range state {
				sum += v
			}
			avgPotential := sum / float64(len(state))
			fmt.Printf("  Average membrane potential = %.3f\n", avgPotential)
			
			// Count active neurons (above threshold)
			activeCount := 0
			for _, v := range state {
				if v > -60.0 { // Threshold for "active"
					activeCount++
				}
			}
			fmt.Printf("  Active neurons: %d/%d (%.1f%%)\n", activeCount, len(state), float64(activeCount)/float64(len(state))*100)
		}
	}

	fmt.Println("\nExample completed successfully!")
}

// Helper function to create a sine wave input
func createSineWaveInput(frequency float64, amplitude float64, numSteps int, dt float64) []float64 {
	input := make([]float64, numSteps*2) // 2 input channels
	
	for i := 0; i < numSteps; i++ {
		time := float64(i) * dt
		value := amplitude * math.Sin(2*math.Pi*frequency*time)
		
		// Channel 1: sine wave
		input[i*2] = value
		// Channel 2: cosine wave (90 degrees out of phase)
		input[i*2+1] = amplitude * math.Cos(2*math.Pi*frequency*time)
	}
	
	return input
}
