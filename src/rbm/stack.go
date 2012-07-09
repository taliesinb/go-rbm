package rbm

import "fmt"
import "math/rand"

type StackedRBM []RBM

func LoadStackedRBM(numv int, path string) ( stack StackedRBM ) {

	weights := ReadArrayFile(path)
	if weights == nil || len(weights) == 0 {
		panic("Cannot read weights from file \"" + path + "\".")
	}

	stack = make([]RBM, len(weights))

	numv++
	for i, W := range weights {

		if len(W) % numv != 0 {
			panic(fmt.Sprintf("Shape of weight matrix %d is inconsistent with specified number of visible units on previous layer", i))
		}

		numh := len(W) / numv
		stack[i].W = W
		stack[i].Init(numh, numv)
		numv = numh + 1
	}
	
	return 
}

func RandomStackedRBM(numv int, width []int) ( stack StackedRBM ) {

	stack = make([]RBM, len(width))

	numv++
	for i := range width {
		
		numh := width[i]
		stack[i].W = RandomMatrix(numh * numv, 1.0)
		stack[i].Init(numh, numv)
		numv = numh + 1
	}

	return
}

func (stack StackedRBM) WriteFile(path string) {

	weights := make([][]float64, len(stack))

	for i := range stack {
		weights[i] = stack[i].W
	}

	WriteArrayFile(path, weights)
}

type TrainingOptions struct {
	Rate, Decay float64
	Rounds int
	Monitor *StepMonitor
}

func (stack StackedRBM) String() (str string) {
	for _, rbm := range stack {
		str += fmt.Sprintf("%v\n", rbm)
	}
	return
}

func AddBias(in []float64) []float64 {
	out := make([]float64, len(in)+1)
	copy(out, in)
	out[len(in)] = 1.0
	return out
}

func (stack StackedRBM) Train(vectors [][]float64, opts TrainingOptions) {
	
	// one set of training vectors for each layer
	training := make([][][]float64, len(stack))

	// construct the first set of training vectors by just adding bias
	// units to the provided vectors
	training[0] = make([][]float64, len(vectors))
	for i := range vectors {
		training[0][i] = AddBias(vectors[i])
	}

	rounds := opts.Rounds
	totalRounds := opts.Rounds * len(stack)
	for i, rbm := range stack {

		fmt.Printf("Training layer %d\n", i)

		// training vectors for layer i
		vectors = training[i] 

		// train layer i on the training vectors 
		for r := 0; r < rounds; r++ {

			// select a random training vector
			n := rand.Intn(len(vectors))

			// train against the selected vector
			rbm.LearnVector(vectors[n], opts.Rate)

			// give a progress update periodically\
			if r % 512 == 0 && opts.Monitor != nil {
				if opts.Monitor.Tick(i * rounds + r, totalRounds) {
					fmt.Printf("%s, norm = %f\n", opts.Monitor, rbm.Norm())
				}
			}

			// periodically decay the weights
			if r % 16 == 0 && opts.Decay > 0 {
				for i := range rbm.W {
					rbm.W[i] *= 1 - opts.Decay
				}
			}
		}

		// construct the visible units for the next layer
		if i < len(stack)-1 {

			training[i+1] = make([][]float64, len(vectors))
			
			for j, vec := range vectors {

				// calculate hidden probabilities
				rbm.Up(1, vec)

				// copy into training vector
				training[i+1][j] = AddBias(rbm.Q)
			}
		}
	}
}

func GetBias(x []float64) []float64 {
	return x[:len(x)+1]
}

func (stack StackedRBM) Reconstruct(vector []float64) []float64 {

	vector = AddBias(vector)

	// go up to top
	for i := 0 ; i < len(stack) ; i++ {
		Transfer(stack[i].W, vector, stack[i].H) 
		Sample(stack[i].P, stack[i].H)
		vector = GetBias(stack[i].H)
	}

	// go down to bottom
	for i := len(stack)-1 ; i > 0 ; i-- {
		TransferT(stack[i].W, stack[i].H, stack[i].V)
		Sample(stack[i].V, stack[i].V)	
		copy(GetBias(stack[i-1].H), stack[i].V)
	}
	
	TransferT(stack[0].W, stack[0].H, stack[0].V) 
	Sample(stack[0].V, stack[0].V)

	return stack[0].V
}

func (stack StackedRBM) Error(v []float64) (err float64) {

	for j := 0 ; j < 512 ; j++ {

		v2 := stack.Reconstruct(v)

		for i := range v {
			if v2[i] * v[i] < 0 {
				err++
			}
		}
	}
	
	err /= 512
	return
}