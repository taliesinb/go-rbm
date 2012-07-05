package main

import (
	"io"
	"os"
	"math"
	"math/rand"
	"path"
	"fmt"
	"flag"
	"unsafe"
	"reflect"
	"strconv"
)

func Multiply(M, x, y []float64) {
	ln := len(x)
	for i := range y { y[i] = 0 }
	for i := range M {
		y[i/ln] += M[i] * x[i%ln]
	}
}

func MultiplyT(M, x, y []float64) {
	ln := len(y)
	for i := range y { y[i] = 0 }
	for i := range M {
		y[i%ln] += M[i] * x[i/ln]
	}
}

func Sample(V []float64, B []float64) {
	for i, v := range V {
		if rand.Float64() < 1.0 / (1.0 + math.Exp(-v)) {
			B[i] = 1.0
		} else {
			B[i] = -1.0
		}
	}
}

func RandomMatrix(M []float64, sd float64) {
	for i := range M {
		M[i] = rand.NormFloat64()*sd
	}
}

// M rows == NH, M cols == NV
func Train(M, V1 []float64, e float64) {
	NV := len(V1)
	NH := len(M) / NV
	H1 := make([]float64, NH)
	H2 := make([]float64, NH)
	V2 := make([]float64, NV)
	H3 := make([]float64, NH)
	V3 := make([]float64, NV)

	for j := 0; j < 4; j++ {
		Multiply(M, V1, H1)
		Sample(H1, H1)	
		
		MultiplyT(M, H1, V2)
		Sample(V2, V2)
		V2[NV-1] = 1.0
		
		Multiply(M, V2, H2)	
		Sample(H2, H2)

		MultiplyT(M, H2, V3)
		Sample(V3, V3)
		V3[NV-1] = 1.0

		Multiply(M, V3, H3)	
		Sample(H3, H3)

		for i := range M {
			M[i] += e * V1[i % NV] * (H1[i / NV] - H2[i / NV])
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

func ReadInt(reader io.Reader) int {
	var b [4]byte
	reader.Read(b[:])
	return int(b[0]) + int(b[1]) << 8 + int(b[2]) << 16 + int(b[3]) << 24
}

func WriteInt(writer io.Writer, x int) {
	var b [4]byte
	b[0] = byte(x)
	b[1] = byte(x >> 8)
	b[2] = byte(x >> 16)
	b[3] = byte(x >> 24)
	writer.Write(b[:])
}

func ReadFloats(reader io.Reader, n int) []float64 {
	bytes := make([]byte, n * 8)
	reader.Read(bytes)
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&bytes))
	header.Len /= 8
	header.Cap /= 8
	return *(*[]float64)(unsafe.Pointer(&header))
}

func WriteFloats(writer io.Writer, slice []float64) {
	header := *(*reflect.SliceHeader)(unsafe.Pointer(&slice))
	header.Len *= 8
	header.Cap *= 8
	writer.Write(*(*[]byte)(unsafe.Pointer(&header)))
}

func ReadSigns(reader io.Reader, n int) (slice []float64) {
	temp := make([]byte, (n + 7) / 8)
	reader.Read(temp)
	for i := range temp {
		slice[i] = 2.0 * float64(temp[i / 8] >> uint(i % 8)) - 1.0
	}
	return
}

func WriteSigns(writer io.Writer, slice []float64) {
	var b byte
	for i := range slice {
		b <<= 1
		if slice[i] > 0.0 { b |= 1 }
		if i % 8 == 0 || i == len(slice) {
			writer.Write([]byte{b})
		}
	}
}

func ReadTextSigns(reader io.Reader) (slice []float64) {
	single := []byte{0}
	for {
		n, err := reader.Read(single)
		if n == 0 || err == io.EOF { return }
		switch single[0] {
		case '\n':
			return
		case '0':
			slice = append(slice, -1.0)
		case '1':
			slice = append(slice, +1.0)
		}
	}
	return
}

func WriteTextSigns(writer io.Writer, slice []float64) {
	bytes := make([]byte, len(slice))
	for i := range slice {
		if slice[i] > 0.0 {
			bytes[i] = '1'
		} else {
			bytes[i] = '0'
		}
	}
	writer.Write(bytes)
}

// this is different from the others in that it doesn't need to know in advance how many elements are in a line
func ReadTextFloats(reader io.Reader) (slice []float64) {
	single := []byte{0}
	buffer := make([]byte, 0, 64) // no float64 will be more than this
	done := false 
	for !done {
		buffer = buffer[:0]
		for {
			n, err := reader.Read(single)
			if n == 0 || err == io.EOF { break }
			switch single[0] {
			case '\n': 
				done = true
				goto got_float
			case '\t':
				goto got_float
			default:
				buffer = append(buffer, single[0])
			}
		}
		got_float:
		if len(buffer) == 0 { break }
		f, err := strconv.ParseFloat(string(buffer), 64)
		if err != nil {
			panic("Couldn't parse a float")
		}
		slice = append(slice, f)
	}
	return
}

var newline []byte = []byte{'\n'}
func WriteTextFloats(writer io.Writer, data []float64) {
	for i := range data {
		if i == 0 {
			fmt.Fprintf(writer, "%f", data[i])
		} else {
			fmt.Fprintf(writer, "\t%f", data[i])
		}
	}
	writer.Write(newline)
}


func WriteArray(writer io.Writer, format string, data [][]float64) {
	switch format {
	case ".sgn":
		WriteInt(writer, len(data))
		WriteInt(writer, len(data[0]))
		for i := range data {
			WriteSigns(writer, data[i])
		}
	case ".flt":	
		WriteInt(writer, len(data))
		WriteInt(writer, len(data[0]))
		for i := range data {
			WriteFloats(writer, data[i])
		}
	case ".txt":
		for i := range data {
			WriteTextSigns(writer, data[i])
		}
	case ".tsv":
		for i := range data {
			WriteTextFloats(writer, data[i])
		}
	default:
		panic("Unknown array format \"" + format + "\"") 
	}	
}

func ReadArray(reader io.Reader, format string) (data [][]float64) {
	switch format {
	case ".sgn", ".flt":
		n := ReadInt(reader)
		sz := ReadInt(reader)
		data = make([][]float64, 0, n)
		fn := ReadFloats
		if format == ".sgn" { fn = ReadSigns }
		for i := 0; i < n; i++ {
			row := fn(reader, sz)
			data = append(data, row)
		}
		
	case ".txt",".tsv":
		fn := ReadTextFloats
		if format == ".txt" { fn = ReadTextSigns }
		sz := 0
		for {
			row := fn(reader)
			if row == nil { break }
			if sz == 0 { sz = len(row) }
			if sz != 0 && sz != len(row) {
				panic("Unequal lengths in table")
			}
			data = append(data, row)
		}
	}
	return
}


func ReadArrayFile(filePath string) (data [][]float64) {

	if filePath == "" { return nil }

	file, err := os.Open(filePath)
	defer file.Close()
	
	if err != nil {
		fmt.Printf("Cannot open \"%s\"", filePath)
		return
	}

	return ReadArray(file, path.Ext(filePath))
}

func WriteArrayFile(filePath string, data [][]float64) {

	file, err := os.Create(filePath)

	if err != nil {
		panic("Couldn't create file \"" + filePath + "\"")
	}

	WriteArray(file, path.Ext(filePath), data)
}

func PrintUsage() {
	fmt.Println("Usage: rbm action <input1> <input2> >output< [options]")
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

		fmt.Printf("%v\n", weights)

		for i, vec := range visibles {
			error := CalculateError(weights, vec)
			fmt.Printf("Average error of example %d: %f\n", i, error)
		}
		
	default:
		PrintUsage()
		return
	}
}