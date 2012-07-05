package rbm

import (
	"fmt"
)

// M rows == NH, M cols == NV
func Train(M, V1 []float64, e float64) {
	NV := len(V1)
	NH := len(M) / NV
	H1 := make([]float64, NH)
	P1 := make([]float64, NH)
	H2 := make([]float64, NH)
	V2 := make([]float64, NV)
	P3 := make([]float64, NH)
	V3 := make([]float64, NV)

	for j := 0; j < 4; j++ {
		Multiply(M, V1, P1)
		Sample(P1, H1)
		
		MultiplyT(M, H1, V2)
		Sample(V2, V2)
		V2[NV-1] = 1.0
		
		Multiply(M, V2, H2)	
		Sample(H2, H2)

		MultiplyT(M, H2, V3)
		Sample(V3, V3)
		V3[NV-1] = 1.0

		Multiply(M, V3, P3)	

		for i := range M {
			M[i] += e * V1[i % NV] * (P1[i / NV] - P3[i / NV])
		}
	}

	for i := range M {
		M[i] *= (1 - e / 16)  // weight decay
	}
}

func CalculateError(M, V1 []float64) ( err float64 ) {
	NV := len(V1)
	NH := len(M) / NV
	H1 := make([]float64, NH)
	V2 := make([]float64, NV)
	H2 := make([]float64, NH)
	V3 := make([]float64, NV)

	for j := 0; j < 512; j++ {
		Multiply(M, V1, H1)
		Sample(H1, H1)	

		MultiplyT(M, H1, V2)
		Sample(V2, V2)

		Multiply(M, V2, H2)
		Sample(H2, H2)	

		MultiplyT(M, H2, V3)
		Sample(V3, V3)

		for i := range V1 {
			if i != len(V1)-1 && V1[i] * V3[i] < 0 { err++ }
		}
	}

	err /= 512
	return
}

func MtoS(X []float64) string {
	s := ""
	for _, x := range X {
		s += fmt.Sprintf("%+.3f ", x)
	}
	return s
}
