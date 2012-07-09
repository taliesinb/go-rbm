package rbm

type RBM struct {
	W []float64 // weight matrix
	H []float64 // hidden units
	V []float64 // visible units
	P []float64 // initial hidden probs
	Q []float64 // final hidden probs
}

func NewBiased(n int) []float64 {
	t := make([]float64, n+1)
	t[n] = 1.0
	return t[:n]
}

func (rbm *RBM) Init(numh, numv int) {
	
	if numh * numv != len(rbm.W) {
		panic("Internal error: Bad matrix shape")
	}

	rbm.H = NewBiased(numh)
	rbm.P = NewBiased(numh)
	rbm.Q = NewBiased(numh)
	rbm.V = make([]float64, numv)
}

func (m *RBM) Down(n int) {

	bias := len(m.V)-1
	for ;n > 0; n-- {
		TransferT(m.W, m.H, m.V)
		Sample(m.V, m.V)
		m.V[bias] = +1

		Transfer(m.W, m.V, m.H)
		Sample(m.H, m.H)
	}

	Transfer(m.W, m.H, m.V)
	Sample(m.V, m.V)
	m.V[bias] = +1.0
}

func (m *RBM) Up(n int, V []float64) {

	Transfer(m.W, V, m.P) 
	Sample(m.P, m.H)

	bias := len(m.V)-1
	for ;n > 0; n-- {
		// go down
		TransferT(m.W, m.H, m.V)
		Sample(m.V, m.V)
		m.V[bias] = +1.0
		
		// go up
		Transfer(m.W, m.V, m.Q)
		Sample(m.Q, m.H)		
	}
}

func (m *RBM) LearnVector(visible []float64, rate float64) {

	nv := len(m.V)

	if visible[nv-1] != 1.0 {
		panic("NO BIAS")
	}

    // 4 sub-rounds
	for j := 0; j < 4; j++ {

        // markov chain
		m.Up(3, visible)

		// contrastive divergence
		for i := range m.W {
			m.W[i] += rate * visible[i % nv] * (m.P[i / nv] - m.Q[i / nv])
		}
	}
}

func (m *RBM) Error(visible []float64) ( err float64 ) {

	for j := 0; j < 512; j++ {

		m.Up(1, visible)
	
		for i := range visible {
			if visible[i] * m.V[i] < 0 {
				err++
			}
		}
	}

	err /= 512
	return
}

func (m *RBM) Norm() ( norm float64 ) {
	for _, x := range m.W {
		norm += x * x
	}
	return
}