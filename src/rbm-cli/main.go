package main

import (
	"math/rand"
	"flag"
	"time"
	"fmt"
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

type ProgressMeter struct {
	start time.Time
	last float64
}

func (pm *ProgressMeter) Print(i, N int) {
	since := time.Since(pm.start).Seconds()
	if since > pm.last + 1.5 {
		pm.last = since
		sz := 0
		for j := i; j > 0; j >>= 1 { sz++ }
		fmt.Printf("%02d%% complete, %s remaining, #%d\n",
			100 * i / N,
			time.Second * time.Duration(
				float64(N-i) * (since/float64(i))),
			i,
			)
	}
}

func main() {

	var numh, numv, rounds int
	var rate, decay float64
	var seed int64
	
	flag.IntVar(&numh, "hidden", 0, "number of hidden units")
	flag.IntVar(&numv, "visible", 0, "number of hidden units")
	flag.IntVar(&rounds, "rounds", 1024, "number of rounds of learning or iteration")
	flag.Int64Var(&seed, "seed", 1, "random seed to use")
	flag.Float64Var(&rate, "rate", 0.0001, "learning rate (log base 2)")
	flag.Float64Var(&decay, "decay", 0.0000001, "weight decay (log base 2)")

	flag.Parse()

	if flag.NArg() < 1 {
		PrintUsage()
		return
	}

	cmd := flag.Arg(0)

	switch fmt.Sprintf("%s%d", cmd, flag.NArg()) {
	case "train2", "train3", "sampledown2", "sampleup3", "reconstruct3", "error3":
	default:
		fmt.Println("Invalid usage")
		PrintUsage()
		return 
	}

	rand.Seed(seed)

	switch cmd {
	case "train":

		visibles, numv := LoadVectors(flag.Arg(1), numv, "Visible")

		rbm := NewMachine(numv, numh)

		fmt.Printf("Generating random %dx%d weight matrix\n", numh, numv)

		RandomMatrix(rbm.W, 1.0)

		fmt.Printf("Commencing %d rounds of learning\n", rounds)

		prog := ProgressMeter{time.Now(), 0}
		for i := 0; i < rounds; i++ {

			// select a random learning vector and learn it
			j := rand.Intn(len(visibles))
			rbm.Train(visibles[j], rate, decay)

			// give a progress update every second
			if i % 512 == 0 {
				prog.Print(i, rounds)
			}
		}

		layers := [][]float64{rbm.W}
		if flag.NArg() == 3 {
			outfile := flag.Arg(2)
			fmt.Printf("Writing weight matrix to \"%s\"\n", outfile)
			WriteArrayFile(outfile, layers)
			
		} else {
			
			fmt.Printf("No output file specified, printing weights to stdout\n")
			WriteArray(os.Stdout, ".tsv", layers)
		}

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
		
	case "error":

		visible, numv := LoadVectors(flag.Arg(1), numv, "Visible")
		rbm := LoadMachine(numv, numh, flag.Arg(2))

		average := 0.0
		fmt.Println("Error\tVector (Sample)")
		for _, vis := range visible {

			error := rbm.Error(vis)
			average += error

			fmt.Printf("%.2f\t", error)
			WriteTextSigns(os.Stdout, vis)
			fmt.Print(" (")
			WriteTextSigns(os.Stdout, rbm.V)
			fmt.Print(")\n")
		}
		average /= float64(len(visible))
		fmt.Printf("\nTotal average error: %f\n", average)
		
	}
}