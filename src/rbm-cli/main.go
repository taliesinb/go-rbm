package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
)

import . "rbm"

func PrintUsage() {
	fmt.Println("Usage: rbm-cli action <input1> <input2> >output< [options]")
	fmt.Println("Actions:")
	fmt.Println("  train <visibles> >weights<")
	fmt.Println("  error <visibles> <weights>")
	fmt.Println("  sampleup    <weights> <visibles> >hiddens<")
	fmt.Println("  sampledown  <weights> <hiddens>  >visibles<")
	fmt.Println("  reconstruct <weights> <visibles> >visibles<")
	fmt.Println("Options:")
	flag.PrintDefaults()
}

func main() {

	var numHidden, numVisible, numRounds int
	var rate float64
	var randSeed int64
	
	flag.IntVar(&numHidden, "hidden", 0, "number of hidden units")
	flag.IntVar(&numVisible, "visible", 0, "number of hidden units")
	flag.IntVar(&numRounds, "rounds", 1024, "number of rounds of learning or iteration")
	flag.Int64Var(&randSeed, "seed", 1, "random seed to use")
	flag.Float64Var(&rate, "rate", 0.001, "learning rate")

	flag.Parse()

	nargs := flag.NArg()
	if nargs < 1 {
		PrintUsage()
		return
	}

	cmd := flag.Arg(0)

	var validCmd bool
	switch cmd { case "train", "sampledown", "sampleup", "reconstruct", "error": validCmd = true }
	
	if !validCmd {
		PrintUsage()
		return 
	}

	rand.Seed(randSeed)

	switch cmd {
	case "train":

		if nargs < 2 || nargs > 3 {
			PrintUsage()
			return
		}
		
		visibles := ReadArrayFile(flag.Arg(1))

		if visibles == nil || len(visibles) <= 1 {
			panic("Invalid or non-existent training vectors")
		}

		if numVisible == 0 { numVisible = len(visibles[0]) }

		if numVisible != len(visibles[0]) {
			panic("--numVisible doesn't agree with training vector shape")
		}

		if numHidden == 0 {
			fmt.Println("--numHidden not specified, using 2")
			numHidden = 2
		}

		fmt.Printf("Generating random %dx%d weight matrix\n", numHidden, numVisible)
		weights := make([]float64, numHidden * numVisible)
		RandomMatrix(weights, 1.0)

		fmt.Printf("Commencing %d rounds of training\n", numRounds)
		for i := 0; i < numRounds; i++ {
			j := rand.Intn(len(visibles))
			Train(weights, visibles[j], rate)
			if i == numRounds / 2 {
				rate *= 0.25
				fmt.Printf("Halfway through, reducing learning rate\n")
			}

			if i == (3 * numRounds) / 4 {
				rate *= 0.25
				fmt.Printf("Three quarters through, reducing learning rate\n")
			}
		}

		layers := [][]float64{weights}
		if nargs == 3 {
			outfile := flag.Arg(2)
			fmt.Printf("Writing weight matrix to \"%s\"\n", outfile)
			WriteArrayFile(outfile, layers)
		} else {
			fmt.Printf("No output file specified, printing weights to stdout\n")
			WriteArray(os.Stdout, ".tsv", layers)
		}

	case "sampledown":

		if nargs < 2 || nargs > 3 {
			PrintUsage()
			return
		}

		weightsList := ReadArrayFile(flag.Arg(1))

		if weightsList == nil || len(weightsList) != 1 {
			panic("Invalid or non-existent weight matrix")
		}

		weights := weightsList[0]

		if numHidden == 0 {
			panic("Unknown number of hidden units")
		}

		NH := numHidden
		NV := len(weights) / NH
		H1 := make([]float64, NH)
		H2 := make([]float64, NH)
		V2 := make([]float64, NV)
		V3 := make([]float64, NV)

		for i := 0; i < NH; i++ {
			fmt.Printf("Samples via hidden unit %2d:\n", i)
			for j := range H1 {
				H1[j] = -1.0
			}
			H1[i] = 1.0

			for t := 0 ; t < 16; t++ {

				MultiplyT(weights, H1, V2)
				Sample(V2, V2)
				//V2[NV-1] = 1.0
		
				Multiply(weights, V2, H2)	
				Sample(H2, H2)

				MultiplyT(weights, H2, V3)
				Sample(V3, V3)
				//V3[NV-1] = 1.0

				WriteTextSigns(os.Stdout, V2)
				fmt.Print("\n")
			}
		}

	case "sampleup":

		if nargs != 3 {
			PrintUsage()
			return
		}

		visibles := ReadArrayFile(flag.Arg(1))

		if visibles == nil || len(visibles) <= 1 {
			panic("Invalid or non-existent training vectors")
		}

		if numVisible == 0 { numVisible = len(visibles[0]) }

		if numVisible != len(visibles[0]) {
			panic("--numVisible doesn't agree with training vector shape")
		}

		weightsList := ReadArrayFile(flag.Arg(2))

		if weightsList == nil || len(weightsList) != 1 {
			panic("Invalid or non-existent weight matrix")
		}

		weights := weightsList[0]		

		NV := numVisible
		NH := len(weights) / NV
		H1 := make([]float64, NH)
		H2 := make([]float64, NH)
		V2 := make([]float64, NV)

		for i, V1 := range visibles {
			fmt.Printf("Hidden unit samples from training vector %2d: ", i)
			WriteTextSigns(os.Stdout, V1)
			fmt.Print("\n")

			for t := 0 ; t < 16; t++ {

				Multiply(weights, V1, H1)
				Sample(H1, H1)

				MultiplyT(weights, H1, V2)
				Sample(V2, V2)
				V2[NV-1] = 1.0
		
				Multiply(weights, V2, H2)	
				Sample(H2, H2)
				
				WriteTextSigns(os.Stdout, H2)
				fmt.Print("\n")
			}
		}
		
	case "error":

		if nargs != 3 {
			PrintUsage()
			return
		}

		visibles := ReadArrayFile(flag.Arg(1))

		if visibles == nil || len(visibles) <= 1 {
			panic("Invalid or non-existent training vectors")
		}

		if numVisible == 0 { numVisible = len(visibles[0]) }

		if numVisible != len(visibles[0]) {
			panic("--numVisible doesn't agree with training vector shape")
		}

		weightsList := ReadArrayFile(flag.Arg(2))

		if weightsList == nil || len(weightsList) != 1 {
			panic("Invalid or non-existent weight matrix")
		}

		weights := weightsList[0]

		for i, vec := range visibles {
			error := CalculateError(weights, vec)
			fmt.Printf("Average error of example %d: %f\n", i, error)
		}
		
	default:
		PrintUsage()
		return
	}
}