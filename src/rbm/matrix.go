package rbm

import (
	"math"
	"math/rand"
)

func Logistic(x float64) float64{
	return 1.0 / (1.0 + math.Exp(-x))
}

func Transfer(M, x, y []float64) {
	ln := len(x)
	for i := range y {
		y[i] = 0
	}
	for i := range M {
		y[i/ln] += M[i] * x[i%ln]
	}
	for i := range y {
		y[i] = Logistic(y[i])
	}
}

func TransferT(M, x, y []float64) {
	ln := len(y)
	for i := range y {
		y[i] = 0
	}
	for i := range M {
		y[i%ln] += M[i] * x[i/ln]
	}
	for i := range y {
		y[i] = Logistic(y[i])
	}
}

func Sample(V []float64, B []float64) {
	for i, v := range V {
		if rand.Float64() < v {
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