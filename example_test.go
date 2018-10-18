package goface

import (
	"path/filepath"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/stretchr/testify/assert"
)

const modelsDir = "models"
const testImgsDir = "tests"

func TestExampleFaceRecognizer(t *testing.T) {
	assert := assert.New(t)

	// Init the recognizer
	rec, err := NewRecognizer(modelsDir)
	assert.Nil(err)
	defer rec.Close()

	// Test image with 7 faces
	testImageGroupOfYoungPeople := filepath.Join(testImgsDir, "group_of_young_people.jpg")
	// Recognize faces in that image
	faces, err := rec.RecognizeFile(testImageGroupOfYoungPeople)
	assert.Nil(err)
	assert.Equal(7, len(faces))
	spew.Dump(faces)

	// Test image with colin powell group
	testImageGroupColinPowell := filepath.Join(testImgsDir, "group_colin_powell.jpg")
	// Recognize faces in that image
	cpFaces, err := rec.RecognizeFile(testImageGroupColinPowell)
	assert.Nil(err)
	assert.Equal(5, len(cpFaces))
	spew.Dump(cpFaces)

	var samples []Vector
	var cats []int32
	for i, f := range cpFaces {
		samples = append(samples, f.Vector)
		// Each face is unique in that image so it goes into its own category
		cats = append(cats, int32(i))
	}
	// Name the categories, i.e. people in the image
	labels := []string{"Jm", "Judy", "Colin", "Maria", "Juan"}
	// Pass samples to the recognizer
	err = rec.SetSamples(samples, cats)
	assert.Nil(err)

	// Test the classification of some unknown colin image
	testImageColin := filepath.Join(testImgsDir, "Colin_Powell/Colin_Powell_0001.jpg")
	colinFace, err := rec.RecognizeSingleFile(testImageColin)
	assert.Nil(err)
	assert.NotNil(colinFace)
	catID, err := rec.Classify(colinFace.Vector)
	assert.Nil(err)
	assert.False(catID < 0)
	assert.Equal("Colin", labels[catID])

}
