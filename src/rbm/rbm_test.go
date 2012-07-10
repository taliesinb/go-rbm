package rbm

import (
	"testing"
	"fmt"
	"math/rand"
)

func Print(args... interface{}) {
	fmt.Print(args)
}

func TestRBM(t *testing.T) {

	rand.Seed(5)

	weights := []Vector{
		RandomMatrix(5 * 3, 0.5),
	}

	training := []Vector{
		Vector{1,1,0,0},
		Vector{0,0,1,1},
	};

	rbm := CreateRBM(4, weights)

	fmt.Printf("SHAPE = %#v\n", rbm)

	rbm.Train(training, TrainingOptions{Rate:0.01, Rounds:65535})

	for j := 0; j < 25; j++ {
		rbm.Reconstruct(Vector{0.5, 0.5, 0.5, 0.8}, 1)
		fmt.Printf("%s\n", rbm)
	}

	rbm.Error(training)
}