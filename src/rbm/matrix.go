package rbm

import (
	"math"
	"math/rand"
)

func Multiply(M, x, y []float64) {
	ln := len(x)
	for i := range y { y[i] = 0 }
	for i := range M {
		y[i/ln] += M[i] * x[i%ln]
	}
}

func MultiplyT(M, x, y []float64) {
	ln := len(y)
	for i := range y { y[i] = 0 }
	for i := range M {
		y[i%ln] += M[i] * x[i/ln]
	}
}

func Logistic(x float64) float64{
	return 1.0 / (1.0 + math.Exp(-x))
}

func Sample(V []float64, B []float64) {
	for i, v := range V {
		if rand.Float64() < Logistic(v) {
			B[i] = 1.0
		} else {
			B[i] = -1.0
		}
	}
}

func RandomMatrix(M []float64, sd float64) {
	for i := range M {
		M[i] = rand.NormFloat64()*sd
	}
}

