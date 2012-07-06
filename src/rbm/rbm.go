package rbm

type RBM struct {
	W []float64       // weight matrix
	H []float64       // hidden units
	V []float64       // visible units
	P1, P2 []float64  // hidden unit probabilities
}

var badSize string = "Not enough information to determine shape of matrix. Specify number of hidden and/or visible units."

func NewMachine(numv, numh int) *RBM {

	if numv * numh == 0 {
		panic(badSize) 
	}
	
	return &RBM{
		make([]float64, numh * numv),
		make([]float64, numh),
		make([]float64, numv),
		make([]float64, numh),
		make([]float64, numh),
	}
}

func LoadMachine(numv, numh int, path string) ( rbm *RBM ) {

	if numv == 0 && numh == 0 {
		panic(badSize)
	}

	list := ReadArrayFile(path)

	if list == nil || len(list) != 1 {
		panic("Cannot read weights from file \"" + path + "\".")
	}

	weights := list[0]

	if n := numv * numh ; n != 0 && n != len(weights) {
		panic("Shape of weight matrix inconsistent with specified number of hidden and visible units.")
	}

	if numv == 0 { numv = len(weights) / numh }
	if numh == 0 { numh = len(weights) / numv }

	rbm = NewMachine(numv, numh)
	rbm.W = weights

	return 
}

func (m *RBM) Iterate(n int, visible []float64) {

	if visible != nil {
		Transfer(m.W, visible, m.P1) // initially go up
		Sample(m.P1, m.H)
	}

	bias := len(m.V)-1
	for i := 0; i < n; i++ {

		// go down
		TransferT(m.W, m.H, m.V)
		Sample(m.V, m.V)

		// clamp the bias unit
		m.V[bias] = +1

		// go up
		Transfer(m.W, m.V, m.P2)
		Sample(m.P2, m.H)
	}

}

func (m *RBM) Train(visible []float64, rate float64, decay float64) {

	nv := len(m.V)
	
    // 4 sub-rounds
	for j := 0; j < 4; j++ {

        // markov chain
		m.Iterate(3, visible)

		// contrastive divergence
		for i := range m.W {
			m.W[i] += rate * visible[i % nv] * (m.P1[i / nv] - m.P2[i / nv])
		}
	}

    // weight decay
	if decay > 0 {
		for i := range m.W {
			m.W[i] *= 1 - decay
		}
	}
}

func (m *RBM) Error(visible []float64) ( err float64 ) {

	n := len(visible)-1
	for j := 0; j < 512; j++ {
	
		m.Iterate(1, visible) // 2?
	
		for i := 0; i < n; i++ {
			if visible[i] * m.V[i] < 0 {
				err++
			}
		}
	}

	err /= 512
	return
}