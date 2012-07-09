package main

import (
	"math/rand"
	"flag"
	"fmt"
	"os"
	"strings"
	"strconv"
)

import . "rbm"

func PrintUsage() {
	fmt.Println("Usage: rbm-cli action <input1> <input2> >output< [options]")
	fmt.Println("Actions:")
	fmt.Println("  train <visibles> >weights<")
	fmt.Println("  error <visibles> <weights>")
	fmt.Println("  sampleup    <weights> <visibles> [>hiddens<]")
	fmt.Println("  sampledown  <weights> <hiddens>  [>visibles<]")
	fmt.Println("  reconstruct <weights> <visibles> [>visibles<]")
	fmt.Println("Options:")
	flag.PrintDefaults()
}

func StringToInts(str string) (ints []int) {
	for i, str := range strings.Split(str, ",") {
		n, err := strconv.Atoi(str)
		if err != nil {
			panic(fmt.Sprintf("Couldn't convert %d'th string \"%s\" to an integer", i, str))
		}
		ints = append(ints, n)
	}
	return
}

func main() {

	var opts TrainingOptions
	var numv int
	var numhstr string
	var seed int64
	
	flag.StringVar(&numhstr, "hidden", "", "number of hidden units (comma-separated list)")
	flag.IntVar(&numv, "visible", 0, "number of visible units")
	flag.Int64Var(&seed, "seed", 1, "random seed to use")
	flag.Float64Var(&opts.Rate, "rate", 0.0001, "learning rate")
	flag.Float64Var(&opts.Decay, "decay", 0.0000001, "weight decay")
	flag.IntVar(&opts.Rounds, "rounds", 1024, "number of rounds of learning or iteration")

	opts.Monitor = new(StepMonitor)

	flag.Parse()

	if flag.NArg() < 1 {
		PrintUsage()
		return
	}

	cmd := flag.Arg(0)
	args := flag.Args()[1:]

	switch fmt.Sprintf("%s%d", cmd, len(args)) {
	case "train2", "sampledown1", "sampleup2", "reconstruct2", "error2":
	default:
		fmt.Println("Invalid usage")
		PrintUsage()
		return 
	}

	rand.Seed(seed)

	numh := StringToInts(numhstr)

	switch cmd {
	case "train":

		visibles, numv := LoadVectors(args[0], numv, "Visible")

		prev := numv + 1
		for _, h := range numh {
			fmt.Printf("Generating random %dx%d weight matrix\n", prev, h)
			prev = h + 1
		}
		stack := RandomStackedRBM(numv, numh)

		fmt.Printf("Commencing %d rounds of learning\n", opts.Rounds)
		stack.Train(visibles, opts)

		fmt.Printf("Writing weight matrix to \"%s\"\n", args[1])
		stack.WriteFile(args[1])

		average := 0.0
		fmt.Println("Error\tVector (Sample)")
		for _, vis := range visibles {

			error := stack.Error(vis)
			average += error

			fmt.Printf("%.2f\t", error)
			WriteTextSigns(os.Stdout, vis)
			fmt.Print(" (")
			WriteTextSigns(os.Stdout, stack[0].V[:len(vis)])
			fmt.Print(")\n")
		}
		average /= float64(len(visibles))
		fmt.Printf("\nTotal average error: %f\n", average)

	case "error":

		visible, numv := LoadVectors(args[0], numv, "Visible")
		rbm := LoadStackedRBM(numv, args[1])

		average := 0.0
		fmt.Println("Error\tVector (Sample)")
		for _, vis := range visible {

			error := rbm.Error(vis)
			average += error

			fmt.Printf("%.2f\t", error)
			WriteTextSigns(os.Stdout, vis)
			fmt.Print(" (")
			WriteTextSigns(os.Stdout, rbm[0].V[:len(vis)])
			fmt.Print(")\n")
		}
		average /= float64(len(visible))
		fmt.Printf("\nTotal average error: %f\n", average)

		/*

	case "sampledown":

		rbm := LoadMachine(numv, numh, flag.Arg(1))

		for i := range rbm.H {
			
			fmt.Printf("Visible samples via hidden unit %d:\n", i)

			for j := range rbm.H {
				rbm.H[j] = -1.0
			}
			rbm.H[i] = 1.0

			for t := 0 ; t < 16; t++ {
				
				rbm.Iterate(4, nil)

				WriteTextSigns(os.Stdout, rbm.V)

				fmt.Print("\n")
			}
		}

	case "reconstruct":

		visible, numv := LoadVectors(flag.Arg(1), numv, "Visible")
		
		rbm := LoadMachine(numv, numh, flag.Arg(2))
		for i, vis := range visible {
			
			fmt.Printf("\nSample vector %d\n", i)
			WriteTextSigns(os.Stdout, vis)
			fmt.Printf("\n\n")

			for t := 0 ; t < 16; t++ {
				rbm.Iterate(2, vis)
				WriteTextSigns(os.Stdout, rbm.V)
				fmt.Print("\n")
			}
		}

	case "sampleup":

		visible, numv := LoadVectors(flag.Arg(1), numv, "Visible")
		
		rbm := LoadMachine(numv, numh, flag.Arg(2))

		for i, vis := range visible {
			
			fmt.Printf("Hidden unit samples from visible vector %d:\n", i)
			WriteTextSigns(os.Stdout, vis)
			fmt.Print("\n")

			for t := 0 ; t < 16; t++ {

				rbm.Iterate(4, vis)
					
				WriteTextSigns(os.Stdout, rbm.H)
				fmt.Print("\n")
			}
		}
		
	}
		*/
	}
}