package main

import "testing"
import "reflect"

func TestRBM(t *testing.T) {

	matrix := []float64{0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 0., 1., 1., 0., 0.}
	vector := []float64{1., 1., 0., -1.}
	result := make([]float64, 6)

	Multiply(matrix, vector, result)

	if !reflect.DeepEqual(result, []float64{0., 1., 1., 1., 2., 2.}) {
		t.Error("Multiplication broken")
	}

	vector = []float64{0., 1., 0., -1., -1., 0.}
	result = make([]float64, 4)
	
	MultiplyT(matrix, vector, result)

	if !reflect.DeepEqual(result, []float64{-2., 0., 0., 0.}) {
		t.Error("Transposed multiplication broken")
	}

	prob := []float64{0.0}
	bit := []float64{0.0}
	val := 0.0
	for j := 0 ; j < 65536; j++ {
		Sample(prob, bit)
		val += bit[0]
	}
	val /= 65536.0
	if val > 0.01 || val < -0.01 {
		t.Errorf("Sampling broken: %v", val)
	}
}