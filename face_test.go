package goface

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"
)

const modelsDir = "models"
const testImgsDir = "tests"

func TestFaceRecognizer(t *testing.T) {
	assert := assert.New(t)

	// Init the recognizer
	rec, err := NewRecognizer(modelsDir)
	assert.Nil(err)
	defer rec.Close()

	// Test image with 7 faces
	testImageGroupOfYoungPeople := filepath.Join(testImgsDir, "group_of_young_people.jpg")
	// Recognize faces in that image
	faces, err := rec.RecognizeFile(testImageGroupOfYoungPeople, 5)
	assert.Nil(err)
	assert.Equal(7, len(faces))

	// Test image with colin powell group
	testImageGroupColinPowell := filepath.Join(testImgsDir, "group_colin_powell.jpg")
	// Recognize faces in that image
	cpFaces, err := rec.RecognizeFile(testImageGroupColinPowell, 5)
	assert.Nil(err)
	assert.Equal(5, len(cpFaces))

	// write out the img with numbers on the faces recognized
	imgIn, _ := os.Open(testImageGroupColinPowell)
	defer imgIn.Close()
	img, _, err := image.Decode(imgIn)
	assert.Nil(err)
	bounds := img.Bounds()
	imgRGB := image.NewRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	draw.Draw(imgRGB, imgRGB.Bounds(), img, bounds.Min, draw.Src)
	for i, face := range cpFaces {
		addImageLabel(imgRGB, face.Rectangle.Min.X+(face.Rectangle.Size().X/2), face.Rectangle.Min.Y+(face.Rectangle.Size().Y/2), fmt.Sprintf("%v", i))
	}
	testImageGroupOfYoungPeopleAnalysis := filepath.Join(testImgsDir, "group_of_young_people_analysis.jpg")
	imgOut, err := os.Create(testImageGroupOfYoungPeopleAnalysis)
	assert.Nil(err)
	defer imgOut.Close()
	jpeg.Encode(imgOut, imgRGB, nil)

	var samples []Vector
	var cats []int32
	for i, f := range cpFaces {
		samples = append(samples, f.Vector)
		// Each face is unique in that image so it goes into its own category
		cats = append(cats, int32(i))
	}
	// Name the categories, i.e. people in the image
	labels := []string{"Jim", "Judy", "Colin", "Maria", "Juan"}
	// Pass samples to the recognizer
	err = rec.SetSamples(samples, cats)
	assert.Nil(err)

	// Test the classification of some unknown colin image
	testImageColin := filepath.Join(testImgsDir, "Colin_Powell/Colin_Powell_0184.jpg")
	colinFace, err := rec.RecognizeSingleFile(testImageColin, 5)
	assert.Nil(err)
	assert.NotNil(colinFace)
	catID, err := rec.Classify(colinFace.Vector)
	assert.Nil(err)
	assert.False(catID < 0)
	assert.Equal("Colin", labels[catID])

	// Test calculations
	euc := colinFace.Euclidean(cpFaces[2])
	assert.True(euc < 0.6)
	prob := colinFace.Probability(cpFaces[2])
	assert.True(prob > 0.85)

}

func addImageLabel(img *image.RGBA, x, y int, label string) {
	col := color.RGBA{0, 0, 0, 255}
	point := fixed.Point26_6{fixed.Int26_6(x * 64), fixed.Int26_6(y * 64)}

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(col),
		Face: basicfont.Face7x13,
		Dot:  point,
	}
	d.DrawString(label)
}
func TestSerializationError(t *testing.T) {
	assert := assert.New(t)

	_, err := NewRecognizer("/notexist")
	assert.Equal(SerializationError("Unable to open /notexist/shape_predictor_5_face_landmarks.dat for reading."), err)
}

func TestInit(t *testing.T) {
	assert := assert.New(t)

	rec, err := NewRecognizer("models")
	assert.Nil(err)
	assert.NotNil(rec)
}

func TestImageLoadError(t *testing.T) {
	assert := assert.New(t)

	rec, err := NewRecognizer("models")
	assert.Nil(err)
	assert.NotNil(rec)

	_, err = rec.Recognize([]byte{1, 2, 3}, 1)
	assert.Equal(ImageLoadError("jpeg_mem_loader: decode error: Not a JPEG file: starts with 0x01 0x02"), err)
}

func TestClose(t *testing.T) {
	assert := assert.New(t)

	rec, err := NewRecognizer("models")
	assert.Nil(err)
	assert.NotNil(rec)
	err = rec.Close()
	assert.Nil(err)
}
