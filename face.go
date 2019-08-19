package goface

// #cgo CXXFLAGS: -std=c++1z -Wall -O3 -DNDEBUG -march=native
// #cgo LDFLAGS: -ljpeg -ldlib
// #include <stdlib.h>
// #include <stdint.h>
// #include "facerec.h"
import "C"
import (
	"image"
	"io/ioutil"
	"math"
	"os"
	"unsafe"
)

const (
	rectLen    = 4
	featureLen = 68 * 2
	vectLen    = 128
)

// A Recognizer creates face vectors for provided images and can also classify
// them into categories
type Recognizer struct {
	ptr    *_Ctype_struct_facerec
	closed bool
}

// Face holds coordinates and vector of a face
type Face struct {
	Rectangle image.Rectangle `json:"rectangle"`
	Features  []image.Point   `json:"features"`
	Vector    Vector          `json:"vector"`
}

// Vector holds 128-dimensional feature vector
type Vector [128]float64

// Vector32 holds 128-dimensional feature vector
type Vector32 [128]float32

// New creates new face with the provided parameters
func NewFace(r image.Rectangle, d Vector) Face {
	return Face{
		Rectangle: r,
		Vector:    d,
	}
}

// NewRecognizer returns a new recognizer. modelDir points to the dir with:
//  - shape_predictor_5_face_landmarks.dat
//  - dlib_face_recognition_resnet_model_v1.dat
func NewRecognizer(modelDir string) (rec *Recognizer, err error) {
	cModelDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cModelDir))
	ptr := C.facerec_init(cModelDir)

	if ptr.err_str != nil {
		defer C.facerec_free(ptr)
		defer C.free(unsafe.Pointer(ptr.err_str))
		err = makeError(C.GoString(ptr.err_str), int(ptr.err_code))
		return
	}

	rec = &Recognizer{ptr, false}
	return
}

// recognize is the private recognize function
func (rec *Recognizer) recognize(imgData []byte, maxFaces, jitter int) (faces []Face, err error) {
	if len(imgData) == 0 {
		err = ImageLoadError("Empty image")
		return
	}
	cImgData := (*C.uint8_t)(&imgData[0])
	cLen := C.int(len(imgData))
	cMaxFaces := C.int(maxFaces)
	cJitter := C.int(jitter)
	ret := C.facerec_recognize(rec.ptr, cImgData, cLen, cMaxFaces, cJitter)
	defer C.free(unsafe.Pointer(ret))

	if ret.err_str != nil {
		defer C.free(unsafe.Pointer(ret.err_str))
		err = makeError(C.GoString(ret.err_str), int(ret.err_code))
		return
	}

	// No faces found
	numFaces := int(ret.num_faces)
	if numFaces == 0 {
		return
	}

	// Copy c++ faces data to Go struct
	// descriptors is the var for a face vector in the c++ code, because we use  std::vector
	defer C.free(unsafe.Pointer(ret.rectangles))
	defer C.free(unsafe.Pointer(ret.features))
	defer C.free(unsafe.Pointer(ret.descriptors))

	rDataLen := numFaces * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

	fDataLen := numFaces * featureLen
	fDataPtr := unsafe.Pointer(ret.features)
	fData := (*[1 << 30]C.long)(fDataPtr)[:fDataLen:fDataLen]
	features := calcFeaturePoints(fData)

	dDataLen := numFaces * vectLen
	dDataPtr := unsafe.Pointer(ret.descriptors)
	dData := (*[1 << 30]float32)(dDataPtr)[:dDataLen:dDataLen]

	for i := 0; i < numFaces; i++ {
		face := Face{}
		x0 := int(rData[i*rectLen])
		y0 := int(rData[i*rectLen+1])
		x1 := int(rData[i*rectLen+2])
		y1 := int(rData[i*rectLen+3])
		face.Rectangle = image.Rect(x0, y0, x1, y1)
		face.Features = features[i]
		vect64 := castTo64(dData[i*vectLen : (i+1)*vectLen])
		copy(face.Vector[:], vect64)
		faces = append(faces, face)
	}
	return
}

func calcFeaturePoints(featureData []C.long) [][]image.Point {
	features := make([][]image.Point, len(featureData)/featureLen)
	f := make([]C.long, featureLen)
	// iterate over faces in features
	numFaces := len(featureData)/featureLen
	for i := 0; i < numFaces; i++ {
		f = featureData[:featureLen]
		featureData = featureData[featureLen:len(featureData)]
		ps := make([]image.Point, featureLen/2)
		for j, k := 0, 0; j < featureLen; j, k = j+2, k+1 {
			p := image.Pt(int(f[j]), int(f[j+1]))
			ps[k] = p
		}
		features[i] = ps
	}
	return features
}

func castTo64(data []float32) (ret []float64) {
	ret = make([]float64, len(data))
	for i, val := range data {
		ret[i] = float64(val)
	}
	return
}

func castTo32(data Vector) (ret Vector32) {
	for i, val := range data {
		ret[i] = float32(val)
	}
	return
}

// recognizeFile is the private file recognize faces private function
func (rec *Recognizer) recognizeFile(imgPath string, maxFaces, jitter int) (face []Face, err error) {
	fd, err := os.Open(imgPath)
	if err != nil {
		return
	}
	imgData, err := ioutil.ReadAll(fd)
	if err != nil {
		return
	}
	return rec.recognize(imgData, maxFaces, jitter)
}

// Recognize takes the bytes of a JPEG and a int to specify the number of times to jitter the faces,
//  and returns all faces found on the provided image, sorted from left to right. It returns a empty
// slice if there are no faces, error is returned if there was a error while decoding/processing image.
// This is thread-safe.
func (rec *Recognizer) Recognize(imgData []byte, jitter int) (faces []Face, err error) {
	if !rec.closed {
		return rec.recognize(imgData, 0, jitter)
	}
	err = closedError
	return
}

// RecognizeSingle takes the bytes of a JPEGand a int to specify the number of times to jitter the faces,
//  and returns a face if it's the only face on the image otherwise it returns nil. This is thread-safe.
func (rec *Recognizer) RecognizeSingle(imgData []byte, jitter int) (face *Face, err error) {
	var faces []Face
	if !rec.closed {
		faces, err = rec.recognize(imgData, 1, jitter)
		if err != nil || len(faces) != 1 {
			return
		}
		face = &faces[0]
		return
	}
	err = closedError
	return
}

// RecognizeFile takes the path of a JPEG and a int to specify the number of times to jitter the faces,
// and returns all faces found on the provided image, sorted from left to right. It returns a empty
// slice if there are no faces, error is returned if there was a error while decoding/processing image.
// This is thread-safe.
func (rec *Recognizer) RecognizeFile(imgPath string, jitter int) (faces []Face, err error) {
	if !rec.closed {
		return rec.recognizeFile(imgPath, 0, jitter)
	}
	err = closedError
	return
}

// RecognizeSingleFile takes the bytes of a JPEG and a int to specify the number of times to jitter the faces,
// and returns a face if it's the only face on the image otherwise it returns nil. This is thread-safe.
func (rec *Recognizer) RecognizeSingleFile(imgPath string, jitter int) (face *Face, err error) {
	var faces []Face
	if !rec.closed {
		faces, err = rec.recognizeFile(imgPath, 1, jitter)
		if err != nil || len(faces) != 1 {
			return
		}
		face = &faces[0]
		return
	}
	err = closedError
	return
}

// SetSamples takes a slice of Vectors and cats then sets known vectors so you can classify after training.
// This is thread-safe.
func (rec *Recognizer) SetSamples(samples []Vector, cats []int32) (err error) {
	if rec.closed {
		err = closedError
		return
	}
	if len(samples) == 0 || len(samples) != len(cats) {
		return
	}
	samples32 := make([]Vector32, len(samples))
	for i, vect := range samples {
		samples32[i] = castTo32(vect)
	}
	cSamples := (*C.float)(unsafe.Pointer(&samples32[0]))
	cCats := (*C.int32_t)(unsafe.Pointer(&cats[0]))
	cLen := C.int(len(samples))
	C.facerec_set_samples(rec.ptr, cSamples, cCats, cLen)
	return
}

// Classify takes a vector returns class ID for the given vector. A negative index is returned if there is no match.
// This is thread-safe.
func (rec *Recognizer) Classify(testSample Vector) (class int, err error) {
	if !rec.closed {
		testSample32 := castTo32(testSample)
		cTestSample := (*C.float)(unsafe.Pointer(&testSample32))
		class = int(C.facerec_classify(rec.ptr, cTestSample))
		return
	}
	err = closedError
	return
}

// Close frees the C++ resources used by the Recognizer. You cannot use the Recognizer after the close call.
func (rec *Recognizer) Close() (err error) {
	if !rec.closed {
		C.facerec_free(rec.ptr)
		rec.ptr = nil
		return
	}
	err = closedError
	return
}

// Probability calculates the the probability two faces are the same.
// Mathematically if the probability is 0.85 or greater it most likely the same person.
func (f *Face) Probability(f2 Face) float64 {
	dist := f.Euclidean(f2)
	return (1 - (dist / 4))

}

// Euclidean calculates the euclidean distance of the two face.
// Mathematically if the distance is 0.6 or less it most likely the same person.
func (f *Face) Euclidean(f2 Face) float64 {
	a := f.Vector
	b := f2.Vector
	var sum float64
	for i := 0; i < len(a); i++ {
		sum += math.Pow((a[i] - b[i]), 2.0)
	}
	return math.Sqrt(sum)
}
