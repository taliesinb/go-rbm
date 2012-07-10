package rbm

import (
	"math"
	"math/rand"
)

const zero = -1
const bias = 1

type Vector []float64

// from -1 to 1: this is quite deliberate
func Logistic(x float64) float64{
	return 2.0 / (1.0 + math.Exp(-x)) - 1.0
}

// there is some unnecessary clamping of biases here,
// mainly a symptom of overzealous debugging
func Transfer(M, x, y Vector) {
	lenx, leny := len(x), len(y)
	for i := 0 ; i < leny ; i++ {
		y[i] = 0
	}
	for i := range M {
		y[i/lenx] += M[i] * x[i%lenx]
	}
	for i := 0 ; i < leny-1 ; i++ {
		y[i] = Logistic(y[i])
	}
	y[leny-1] = bias
}

func TransferT(M, x, y Vector) {
	lenx, leny := len(x), len(y)
	x[lenx-1] = 0.0 // disable hidden unit bias
	for i := range y {
		y[i] = 0
	}
	for i := range M {
		y[i%leny] += M[i] * x[i/leny]
	}
	for i := 0 ; i < leny-1 ; i++ {
		y[i] = Logistic(y[i])
	}
	x[lenx-1] = bias
	y[leny-1] = bias
}

func Sample(V, B Vector) {
	for i, v := range V {
		if 2 * rand.Float64() - 1 < v {
			B[i] = 1.0
		} else {
			B[i] = zero
		}
	}
	B[len(V)-1] = bias
}

func RandomMatrix(n int, sd float64) (M Vector) {
	M = make(Vector, n)
	for i := range M {
		M[i] = rand.NormFloat64()*sd
	}
	return
}

func (vec Vector) String() (str string) {
	for _, v := range vec {
		switch {
		case v > 0.51: str += "+" 
		case v < 0.49: str += "-" 
		case v > 0:    str += "."
		default:       str += "X"
		}
	}
	return
}

func (vec Vector) MatrixString(n int) (str string) {
	for i := 0 ; i < len(vec); i += n {
		str += Vector(vec[i:i+n]).String()
		str += "\n"
	}
	return 
}

func HammingError(vec1, vec2 Vector) (err float64) {
	for i := range vec1 {
		if (vec1[i] - 0.5) * (vec2[i] - 0.5) < 0.001 {
			err += 1.0
		}
	}
	return
}

func RMSError(vec1, vec2 Vector) (err float64) {
	if len(vec1) != len(vec2) { panic("Different length vectors") }
	for i := range vec1 {
		f := (vec1[i] - vec2[i])/(1 - zero)
		err += f * f
	}
	return
}

func DelBias(v Vector) Vector {
	return v[:len(v)-1]
}

func AddBias(v Vector) (t Vector) {
	t = make(Vector, len(v)+1)
	copy(t, v)
	t[len(v)] = bias
	return
}