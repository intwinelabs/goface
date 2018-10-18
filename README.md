## goface

This package implements face recognition for Go using:

- [dlib](http://dlib.net)
- [FaceNet](https://arxiv.org/abs/1503.03832)

## Requirements

goface requires dlib v19.10 or greater and libjpeg dev packages to installed
goface also requires build-essential & cmake if we have to build dlib

### Ubuntu 16.04 or Ubuntu 18.04

We mus build the latest version of dlib:

```bash
make dlib
make install-deps
```

### Ubuntu 18.10+

Latest versions of Ubuntu and Debian provide dlib packages v19.10 or greater so we can just run:

```bash
sudo apt-get install libdlib-dev libopenblas-dev libjpeg-turbo8-dev
```

## Models

goface uses the dlib models `shape_predictor_5_face_landmarks.dat` and `dlib_face_recognition_resnet_model_v1.dat`

Run the make command to pull from our mirror:
```bash
make models
```

You can also download these from the [dlib-models](https://github.com/davisking/dlib-models) repo:

```bash
mkdir models && cd models
wget https://github.com/davisking/dlib-models/raw/master/shape_predictor_5_face_landmarks.dat.bz2
2
wget https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
```

## Usage

To use goface:

```go
import "github.com/intwineapp/goface"
```

To install:

```bash
go get github.com/intwineapp/goface
```

## Test

```bash
make test
```
