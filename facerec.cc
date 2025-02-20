#include <shared_mutex>
#include <dlib/dnn.h>
#include <dlib/image_loader/image_loader.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_transforms.h>
#include <stdio.h>
#include "facerec.h"
#include "jpeg_mem_loader.h"
#include "classify.h"

using namespace dlib;
using namespace std;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;

static const size_t RECT_LEN = 4;
static const size_t FEATURE_LEN = 2 * 68;
static const size_t DESCR_LEN = 128;
static const size_t RECT_SIZE = RECT_LEN * sizeof(long);
static const size_t FEATURE_SIZE = FEATURE_LEN * sizeof(long);
static const size_t DESCR_SIZE = DESCR_LEN * sizeof(float);

class FaceRec
{
  public:
	FaceRec(const char *model_dir)
	{
		detector_ = get_frontal_face_detector();

		string dir = model_dir;
		string shape_predictor_path = dir + "/shape_predictor_68_face_landmarks.dat";
		string resnet_path = dir + "/dlib_face_recognition_resnet_model_v1.dat";

		deserialize(shape_predictor_path) >> sp_;
		deserialize(resnet_path) >> net_;
	}

	tuple<std::vector<rectangle>, std::vector<full_object_detection>, std::vector<descriptor>>
	Recognize(const matrix<rgb_pixel> &img, int max_faces, int jitter)
	{
		std::vector<rectangle> rects;
		std::vector<full_object_detection> shapes;
		std::vector<descriptor> descrs;
		
		{
			lock_guard<std::mutex> lock(detector_mutex_);
			rects = detector_(img);
		}
		
		// Short circuit.
		if (rects.size() == 0 || (max_faces > 0 && rects.size() > (size_t)max_faces))
		{
			return {move(rects), move(shapes), move(descrs)};
		}

		sort(rects.begin(), rects.end());

		std::vector<matrix<rgb_pixel>> face_imgs;
		for (const auto &rect : rects)
		{
			auto shape = sp_(img, rect);
			shapes.push_back(shape);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			face_imgs.push_back(move(face_chip));
		}
		
		
		{
			// The face recognition accuracy is being improved by jittering the face descriptors.
			// In particular, to get 99.38% on the LFW benchmark you need to use the jitter_face()
			// routine to compute the descriptors
			for (size_t i = 0; i < face_imgs.size(); ++i)
			{
				// All this does is make jitterN copies of img, all slightly jittered by being zoomed,
				// rotated, and translated a little bit differently. They are also randomly mirrored.
				{
					lock_guard<std::mutex> lock(net_mutex_);
					descrs.push_back(mean(mat(net_(jitter_face(face_imgs[i], jitter)))));
				}

			}
		}
		return {move(rects), move(shapes), move(descrs)};
	}

	void SetSamples(std::vector<descriptor> &&samples, unordered_map<int, int> &&cats)
	{
		unique_lock<std::shared_mutex> lock(samples_mutex_);
		samples_ = move(samples);
		cats_ = move(cats);
	}

	int Classify(const descriptor &test_sample)
	{
		shared_lock<std::shared_mutex> lock(samples_mutex_);
		if (samples_.size() == 0)
			return -1;
		return classify(samples_, cats_, test_sample);
	}

  private:
	std::mutex detector_mutex_;
	std::mutex net_mutex_;
	std::shared_mutex samples_mutex_;
	frontal_face_detector detector_;
	shape_predictor sp_;
	anet_type net_;
	std::vector<descriptor> samples_;
	unordered_map<int, int> cats_;

	std::vector<matrix<rgb_pixel>> jitter_face(
		const matrix<rgb_pixel>& img, int jitter
	)
	{
		// All this function does is make 100 copies of img, all slightly jittered by being
		// zoomed, rotated, and translated a little bit differently. They are also randomly
		// mirrored left to right.
		thread_local dlib::rand rnd;

		std::vector<matrix<rgb_pixel>> crops; 
		for (int i = 0; i < jitter; ++i)
			crops.push_back(jitter_image(img,rnd));

		return crops;
	}
};

// Plain C interface for Go.

facerec *facerec_init(const char *model_dir)
{
	facerec *rec = (facerec *)calloc(1, sizeof(facerec));
	try
	{
		FaceRec *cls = new FaceRec(model_dir);
		rec->cls = (void *)cls;
	}
	catch (serialization_error &e)
	{
		rec->err_str = strdup(e.what());
		rec->err_code = SERIALIZATION_ERROR;
	}
	catch (exception &e)
	{
		rec->err_str = strdup(e.what());
		rec->err_code = UNKNOWN_ERROR;
	}
	return rec;
}

faceret *facerec_recognize(facerec *rec, const uint8_t *img_data, int len, int max_faces, int jitter)
{
	faceret *ret = (faceret *)calloc(1, sizeof(faceret));
	FaceRec *cls = (FaceRec *)(rec->cls);
	matrix<rgb_pixel> img;
	std::vector<rectangle> rects;
	std::vector<full_object_detection> shapes;
	std::vector<descriptor> descrs;
	int i, j;
	try
	{
		load_mem_jpeg(img, img_data, len);
		tie(rects, shapes, descrs) = cls->Recognize(img, max_faces, jitter);
	}
	catch (image_load_error &e)
	{
		ret->err_str = strdup(e.what());
		ret->err_code = IMAGE_LOAD_ERROR;
		return ret;
	}
	catch (exception &e)
	{
		ret->err_str = strdup(e.what());
		ret->err_code = UNKNOWN_ERROR;
		return ret;
	}
	ret->num_faces = descrs.size();
	if (ret->num_faces == 0)
		return ret;
	ret->rectangles = (long *)malloc(ret->num_faces * RECT_SIZE);
	for (i = 0; i < ret->num_faces; i++)
	{
		long *dst = ret->rectangles + i * 4;
		dst[0] = rects[i].left();
		dst[1] = rects[i].top();
		dst[2] = rects[i].right();
		dst[3] = rects[i].bottom();
	}
	ret->features = (long *)malloc(ret->num_faces * FEATURE_SIZE);
	for (i =0; i < ret->num_faces; i++ )
	{
		for (j=0; j < 68; j++)
		{
			//cout << "[" << i << "] pixel position of part:  " << shapes[i].part(j) << endl;
			long *dst = ret->features + ((i * 68) + j) * 2;
			dst[0] = shapes[i].part(j).x();
			dst[1] = shapes[i].part(j).y();
		}
	}
	ret->descriptors = (float *)malloc(ret->num_faces * DESCR_SIZE);
	for (i = 0; i < ret->num_faces; i++)
	{
		void *dst = (uint8_t *)(ret->descriptors) + i * DESCR_SIZE;
		void *src = (void *)&descrs[i](0, 0);
		memcpy(dst, src, DESCR_SIZE);
	}
	return ret;
}

void facerec_set_samples(
	facerec *rec,
	const float *c_samples,
	const int32_t *c_cats,
	int len)
{
	FaceRec *cls = (FaceRec *)(rec->cls);
	std::vector<descriptor> samples;
	samples.reserve(len);
	for (int i = 0; i < len; i++)
	{
		descriptor sample = mat(c_samples + i * DESCR_LEN, DESCR_LEN, 1);
		samples.push_back(move(sample));
	}
	unordered_map<int, int> cats;
	cats.reserve(len);
	for (int i = 0; i < len; i++)
	{
		cats[i] = c_cats[i];
	}
	cls->SetSamples(move(samples), move(cats));
}

int facerec_classify(facerec *rec, const float *c_test_sample)
{
	FaceRec *cls = (FaceRec *)(rec->cls);
	descriptor test_sample = mat(c_test_sample, DESCR_LEN, 1);
	return cls->Classify(test_sample);
}

void facerec_free(facerec *rec)
{
	if (rec)
	{
		if (rec->cls)
		{
			FaceRec *cls = (FaceRec *)(rec->cls);
			delete cls;
			rec->cls = NULL;
		}
		free(rec);
	}
}
