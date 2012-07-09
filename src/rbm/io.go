package rbm

import (
	"os"
	"io"
	"fmt"
	"path"
	"unsafe"
	"reflect"
	"strconv"
)

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


func ReadArray(reader io.Reader, format string) (data [][]float64) {
	switch format {
	case ".sgn", ".flt":
		n := ReadInt(reader)
		data = make([][]float64, 0, n)
		fn := ReadFloats
		if format == ".sgn" { fn = ReadSigns }
		for i := 0; i < n; i++ {
			sz := ReadInt(reader)
			row := fn(reader, sz)
			data = append(data, row)
		}
		
	case ".txt",".tsv":
		fn := ReadTextFloats
		if format == ".txt" { fn = ReadTextSigns }
		for {
			row := fn(reader)
			if row == nil { break }
			data = append(data, row)
		}
	}
	return
}

func WriteArray(writer io.Writer, format string, data [][]float64) {
	switch format {
	case ".sgn":
		WriteInt(writer, len(data))
		for i := range data {
			WriteInt(writer, len(data[i]))
			WriteSigns(writer, data[i])
		}
	case ".flt":	
		WriteInt(writer, len(data))
		for i := range data {
			WriteInt(writer, len(data[i]))
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

func LoadVectors(path string, n int, t string) ( [][]float64, int ) {
	
	visibles := ReadArrayFile(path)

	if visibles == nil || len(visibles) <= 1 {
		panic("Invalid or non-existent " + t + " file \"" + path + "\"")
	}

	if n == 0 {
		n = len(visibles[0])	
	} else if n != len(visibles[0]) {
		panic("--num" + t + " doesn't agree with training vector shape")
	}

	return visibles, n
}
