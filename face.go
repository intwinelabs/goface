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
	"os"
	"unsafe"
)

const (
	rectLen = 4
	vectLen = 128
)

// A Recognizer creates face vectors for provided images and can also classify
// them into categories
type Recognizer struct {
	ptr    *_Ctype_struct_facerec
	closed bool
}

// Face holds coordinates and vector of a face
type Face struct {
	Rectangle image.Rectangle
	Vector    Vector
}

// Vector holds 128-dimensional feature vector
type Vector [128]float32

// New creates new face with the provided parameters
func NewFace(r image.Rectangle, d Vector) Face {
	return Face{r, d}
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
func (rec *Recognizer) recognize(imgData []byte, maxFaces int) (faces []Face, err error) {
	if len(imgData) == 0 {
		err = ImageLoadError("Empty image")
		return
	}
	cImgData := (*C.uint8_t)(&imgData[0])
	cLen := C.int(len(imgData))
	cMaxFaces := C.int(maxFaces)
	ret := C.facerec_recognize(rec.ptr, cImgData, cLen, cMaxFaces)
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
	defer C.free(unsafe.Pointer(ret.descriptors))

	rDataLen := numFaces * rectLen
	rDataPtr := unsafe.Pointer(ret.rectangles)
	rData := (*[1 << 30]C.long)(rDataPtr)[:rDataLen:rDataLen]

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
		copy(face.Vector[:], dData[i*vectLen:(i+1)*vectLen])
		faces = append(faces, face)
	}
	return
}

// recognizeFile is the private file recognize faces private function
func (rec *Recognizer) recognizeFile(imgPath string, maxFaces int) (face []Face, err error) {
	fd, err := os.Open(imgPath)
	if err != nil {
		return
	}
	imgData, err := ioutil.ReadAll(fd)
	if err != nil {
		return
	}
	return rec.recognize(imgData, maxFaces)
}

// Recognize takes the bytes of a JPEG and returns all faces found on the provided image, sorted from
// left to right. It returns a empty slice if there are no faces, error is returned if there was
// a error while decoding/processing image. This is thread-safe.
func (rec *Recognizer) Recognize(imgData []byte) (faces []Face, err error) {
	if !rec.closed {
		return rec.recognize(imgData, 0)
	}
	err = closedError
	return
}

// RecognizeSingle takes the bytes of a JPEG and returns a face if it's the only face on the image
// otherwise it returns nil. This is thread-safe.
func (rec *Recognizer) RecognizeSingle(imgData []byte) (face *Face, err error) {
	var faces []Face
	if !rec.closed {
		faces, err = rec.recognize(imgData, 1)
		if err != nil || len(faces) != 1 {
			return
		}
		face = &faces[0]
		return
	}
	err = closedError
	return
}

// RecognizeFile takes the path of a JPEG and returns all faces found on the provided image, sorted from
// left to right. It returns a empty slice if there are no faces, error is returned if there was
// a error while decoding/processing image. This is thread-safe.
func (rec *Recognizer) RecognizeFile(imgPath string) (faces []Face, err error) {
	if !rec.closed {
		return rec.recognizeFile(imgPath, 0)
	}
	err = closedError
	return
}

// RecognizeSingleFile takes the bytes of a JPEG and returns a face if it's the only face on the image
// otherwise it returns nil. This is thread-safe.
func (rec *Recognizer) RecognizeSingleFile(imgPath string) (face *Face, err error) {
	var faces []Face
	if !rec.closed {
		faces, err = rec.recognizeFile(imgPath, 1)
		if err != nil || len(faces) != 1 {
			return
		}
		face = &faces[0]
		return
	}
	err = closedError
	return
}

// SetSamples takes a slive of Vectors and cats then sets known vectors so you can classify after training.
// This is thread-safe.
func (rec *Recognizer) SetSamples(samples []Vector, cats []int32) (err error) {
	if rec.closed {
		err = closedError
		return
	}
	if len(samples) == 0 || len(samples) != len(cats) {
		return
	}
	cSamples := (*C.float)(unsafe.Pointer(&samples[0]))
	cCats := (*C.int32_t)(unsafe.Pointer(&cats[0]))
	cLen := C.int(len(samples))
	C.facerec_set_samples(rec.ptr, cSamples, cCats, cLen)
	return
}

// Classify takes a vector returns class ID for the given vector. A negative index is returned if there is no match.
// This is thread-safe.
func (rec *Recognizer) Classify(testSample Vector) (class int, err error) {
	if !rec.closed {
		cTestSample := (*C.float)(unsafe.Pointer(&testSample))
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
