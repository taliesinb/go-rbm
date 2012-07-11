package rbm

import (
	"math/rand"
	"fmt"
)

type TrainingOptions struct {
	Rate, Decay float64
	Rounds int
	Monitor *StepMonitor
}

type RBM struct {
	S []int    // shape: number of units in each of the D layers
	W []Vector // list of D weight matrices
	U []Vector // list of D+1 unit activations
}

func CreateRBM(numv int, W []Vector) (rbm *RBM) {

	rbm = new(RBM)

	rbm.W = W
	rbm.S = make([]int, len(W)+1)
	rbm.U = make([]Vector, len(W)+1)

	numv++
	rbm.U[0] = make(Vector, numv)
	rbm.S[0] = numv
	
	for i, w := range W {
		numv = len(w) / numv
		rbm.U[i+1] = make(Vector, numv)
		rbm.U[i+1][numv-1] = one
		rbm.S[i+1] = numv
	}

	return
}

func LoadRBM(numv int, path string) (rbm *RBM) {
	W := ReadArrayFile(path)
	rbm = CreateRBM(numv, W)
	fmt.Printf("Loaded RBM of shape %v\n", rbm.S)
	return
}

func (m *RBM) WriteFile(path string) {
	WriteArrayFile(path, m.W) 
}

func (m *RBM) Weights() (str string) {
	depth := len(m.U)-1
	for j := depth ; j >= 0 ; j-- {
		if j < depth { str += "\n" }
		str += DelBias(m.U[j]).String()
	}
	return
}

func (m *RBM) String() (str string) {
	depth := len(m.U)-1
	for j := depth ; j >= 0 ; j-- {
		if j < depth { str += "|" }
		str += DelBias(m.U[j]).String()
	}
	return
}

func (m *RBM) WeightsString() (str string) {
	for i := range m.W {
		str += m.W[i].MatrixString(m.S[i])
	}
	return
}

func (m *RBM) Reconstruct(T Vector, iters int, sample bool) (R Vector) {

	R = make(Vector, len(T))

	d := len(m.W)

	for k := 0 ; k < iters ; k++ {

		copy(m.U[0], T)
		m.U[0][len(T)] = 1.0
		
		for i := 0 ; i < d ; i++ {
			Transfer(m.W[i], m.U[i], m.U[i+1])
			Sample(m.U[i+1], m.U[i+1])
		}

		for i := d; i > 0; i-- {
			TransferT(m.W[i-1], m.U[i], m.U[i-1])
			if i > 1 || sample { Sample(m.U[i-1], m.U[i-1]) }
		}

		for i := range R {
			R[i] += m.U[0][i]
		}

	}
	for i := range R {
		R[i] /= float64(iters)
	}

	return
}

func (m *RBM) Error(T []Vector, sample bool) (totalerr float64) {
	
	// add bias unit to traiiteravgning vectors
	B := make([]Vector, len(T))
	for j := range T {
		B[j] = AddBias(T[j])
	}

	for i, t := range T {
		err := 0.0
		for j := 0 ; j < 64 ; j++ {
			r := m.Reconstruct(t, 1, sample)
			err += RMSError(r, t)
		}
		err /= 64.0
		if i < 128 || i > len(T) - 128 {
			fmt.Printf("\t%d: %s -> %s, avg err = %f\n",
				i, t.String(), m.Reconstruct(t, 1, sample).String(), err)
		}
		totalerr += err
	}
	totalerr /= float64(len(T))
	return
}

func (m *RBM) Train(T []Vector, opts TrainingOptions) {

	// add bias unit to training vectors
	B := make([]Vector, len(T))
	for j := range T {
		B[j] = AddBias(T[j])
		//fmt.Printf("%s\n", B[j])
	}

	rate := opts.Rate
	rounds := opts.Rounds
	decay := opts.Decay
	
	// train successive layers
	for i := range m.W {

		nv := m.S[i]
		nh := m.S[i+1]
		P := make(Vector, nh)
		Q := make(Vector, nh)

		W := m.W[i]
		V := m.U[i]
		H := m.U[i+1]

		fmt.Printf("Training layer %d: %d visible units -> %d hidden units\n", i, len(V), len(H))

		if opts.Monitor != nil {
			opts.Monitor.Reset()
		}

		for r := 0 ; r < rounds ; r++ {

			if r % 16 == 0 && opts.Monitor != nil {
				if opts.Monitor.Tick(r, rounds) {
					fmt.Printf("\t%s\n", opts.Monitor)
				}
			}
			
			// select a random training vector
			n := rand.Intn(len(B))

			// sample our way up to the desired layer			
			Transfer(W, B[n], P)
			Sample(P, H)

			// sample down
			TransferT(W, H, V)
			Sample(V, V)

			// sample up
			Transfer(W, V, H)
			Sample(H, H)

			// sample down
			TransferT(W, H, V)
			Sample(V, V)
			
			// CD works better on probabilities, so last sample not needed
			Transfer(W, V, Q)

			// contrastive divergence
			for i := range W {
				W[i] += rate * B[n][i % nv] * (P[i / nv] - Q[i / nv])
			}

			if r % 32 == 0 && decay > 0 {
				for i := range W {
					W[i] *= 1 - decay
				}
			}
		}

		// create the next layer of bias vector
		for i := range B {
			b2 := make(Vector, nh)
			Transfer(W, B[i], b2)
			B[i] = b2
		}
	}
}