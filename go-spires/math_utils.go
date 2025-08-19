package spires

import (
	"fmt"
	"math"
	"math/rand"
)

// Matrix operations and mathematical utilities

// MatVecMult performs matrix-vector multiplication: y = A * x
// A is stored in row-major order: A[i*cols + j] = A[i][j]
func MatVecMult(A, x, y []float64, rows, cols int) {
	for i := 0; i < rows; i++ {
		y[i] = 0
		for j := 0; j < cols; j++ {
			y[i] += A[i*cols+j] * x[j]
		}
	}
}

// MatMatMult performs matrix-matrix multiplication: C = A * B
// All matrices are stored in row-major order
func MatMatMult(A, B, C []float64, r1, c1, c2 int) {
	for i := 0; i < r1; i++ {
		for j := 0; j < c2; j++ {
			C[i*c2+j] = 0
			for k := 0; k < c1; k++ {
				C[i*c2+j] += A[i*c1+k] * B[k*c2+j]
			}
		}
	}
}

// MatTranspose computes the transpose of matrix A: A_T = A^T
func MatTranspose(A, A_T []float64, rows, cols int) {
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			A_T[j*rows+i] = A[i*cols+j]
		}
	}
}

// CalcSpectralRadius calculates the spectral radius (largest eigenvalue magnitude) of a matrix
// This is a simplified implementation using power iteration
func CalcSpectralRadius(A []float64, n int) float64 {
	if n <= 0 {
		return 0.0
	}

	// Initialize random vector
	x := make([]float64, n)
	for i := range x {
		x[i] = 2*rand.Float64() - 1
	}

	// Power iteration
	maxIter := 100
	tolerance := 1e-10
	
	for iter := 0; iter < maxIter; iter++ {
		// Compute y = A * x
		y := make([]float64, n)
		MatVecMult(A, x, y, n, n)
		
		// Find maximum magnitude
		maxMag := 0.0
		for i := range y {
			mag := math.Abs(y[i])
			if mag > maxMag {
				maxMag = mag
			}
		}
		
		if maxMag == 0 {
			return 0.0
		}
		
		// Normalize y
		for i := range y {
			y[i] /= maxMag
		}
		
		// Check convergence
		diff := 0.0
		for i := range x {
			diff += math.Abs(y[i] - x[i])
		}
		
		if diff < tolerance {
			return maxMag
		}
		
		// Update x
		copy(x, y)
	}
	
	// Return approximate spectral radius
	return 0.0
}

// RescaleMatrix rescales matrix A to achieve target spectral radius
func RescaleMatrix(A []float64, n int, targetRho float64) {
	currentRho := CalcSpectralRadius(A, n)
	if currentRho > 0 {
		scale := targetRho / currentRho
		for i := range A {
			A[i] *= scale
		}
	}
}

// SolveLinearSystemLUD solves Ax = b using LU decomposition
// This is a simplified implementation
func SolveLinearSystemLUD(A []float64, b, x []float64, n int) error {
	if n <= 0 {
		return fmt.Errorf("invalid matrix size")
	}
	
	// Create copies to avoid modifying original data
	A_copy := make([]float64, len(A))
	copy(A_copy, A)
	b_copy := make([]float64, len(b))
	copy(b_copy, b)
	
	// LU decomposition with partial pivoting
	lu, pivots, err := luDecomposition(A_copy, n)
	if err != nil {
		return err
	}
	
	// Solve Ly = Pb (forward substitution)
	y := make([]float64, n)
	for i := 0; i < n; i++ {
		y[i] = b_copy[pivots[i]]
		for j := 0; j < i; j++ {
			y[i] -= lu[i*n+j] * y[j]
		}
	}
	
	// Solve Ux = y (backward substitution)
	for i := n - 1; i >= 0; i-- {
		x[i] = y[i]
		for j := i + 1; j < n; j++ {
			x[i] -= lu[i*n+j] * x[j]
		}
		x[i] /= lu[i*n+i]
	}
	
	return nil
}

// luDecomposition performs LU decomposition with partial pivoting
func luDecomposition(A []float64, n int) ([]float64, []int, error) {
	lu := make([]float64, len(A))
	copy(lu, A)
	pivots := make([]int, n)
	
	for i := range pivots {
		pivots[i] = i
	}
	
	for k := 0; k < n-1; k++ {
		// Find pivot
		maxRow := k
		maxVal := math.Abs(lu[k*n+k])
		for i := k + 1; i < n; i++ {
			val := math.Abs(lu[i*n+k])
			if val > maxVal {
				maxVal = val
				maxRow = i
			}
		}
		
		if maxVal == 0 {
			return nil, nil, fmt.Errorf("singular matrix")
		}
		
		// Swap rows if necessary
		if maxRow != k {
			// Swap pivot rows
			for j := 0; j < n; j++ {
				lu[k*n+j], lu[maxRow*n+j] = lu[maxRow*n+j], lu[k*n+j]
			}
			// Update pivot array
			pivots[k], pivots[maxRow] = pivots[maxRow], pivots[k]
		}
		
		// Eliminate column k
		for i := k + 1; i < n; i++ {
			factor := lu[i*n+k] / lu[k*n+k]
			lu[i*n+k] = factor
			for j := k + 1; j < n; j++ {
				lu[i*n+j] -= factor * lu[k*n+j]
			}
		}
	}
	
	return lu, pivots, nil
}

// Vector operations
func VectorAdd(a, b, result []float64) {
	for i := range a {
		result[i] = a[i] + b[i]
	}
}

func VectorSub(a, b, result []float64) {
	for i := range a {
		result[i] = a[i] - b[i]
	}
}

func VectorScale(a []float64, scalar float64, result []float64) {
	for i := range a {
		result[i] = a[i] * scalar
	}
}

func VectorDot(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func VectorNorm(a []float64) float64 {
	return math.Sqrt(VectorDot(a, a))
}

// Random number generation utilities
func RandomMatrix(rows, cols int, min, max float64) []float64 {
	result := make([]float64, rows*cols)
	range_ := max - min
	for i := range result {
		result[i] = min + rand.Float64()*range_
	}
	return result
}

func RandomVector(n int, min, max float64) []float64 {
	result := make([]float64, n)
	range_ := max - min
	for i := range result {
		result[i] = min + rand.Float64()*range_
	}
	return result
}
