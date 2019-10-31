// ----------------------------------------------------------------------------
//
// Copyright (C) 2007-13 Alex Segal, Vladimir Monahov. All rights reserved.
// Use this source code "AS IS" with no warranty whatsoever
//
// $Author: asegal $
// $Rev: 1470 $
// $Date: 2013-06-21 12:03:10 -0700 (Fri, 21 Jun 2013) $
//
// ----------------------------------------------------------------------------

/*
This is the texture linearization utility. It is supposed to transform pixels
of an arbitrary input image into a different (linear) color space and save them
to a destination image of the same size.
Please note: the use of the "linear color space" term is a bit vague here.
In fact, we assume sRGB/Rec.709 color primaries, so the resulting RGB
is not a CIE RGB or XYZ or anything else but merely a linear sRGB space.
This is why for sRGB and Rec.709 linearization we only get rid of their gamma curves.
For DCI P3 the process is more involved and includes applying 2.6 gamma curve,
then inverse P3 to XYZ transform, then XYZ to sRGB transform.
If an input image has a premultiplied alpha, we unpremult color channels first,
then linearize them, then premult back. Alpha channel is not affected itself.

There is an undocumented -r flag (reverse). If used, it will force reversed
process: linear image can be converted ito one of the color space
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <boost/thread.hpp>

#include "OpenEXR/ImathMatrix.h"
#include "OpenEXR/ImathColor.h"

#include "OpenImageIO/imageio.h"

static
void usage()
{
	fprintf(stderr,
		"Usage:\n"
			"\tlinearize [-space srgb|rec709|p3]|[-gamma <float>] [-threads <int>] <input_file> <output_file>\n\n");

}

enum Space {
	kUnknown = -1,
	kSrgb,
	kRec709,
	kP3,
	kGamma
};

OIIO_NAMESPACE_USING

//
// Create a 4x4 matrix (with only 3x3 values used) for the given white point and R/G/B primaries.
// The algorithm is based on formulas in http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
// with some magic added to calculate white point.
// NOTE: all matrices in brucelindbloom.com (and everywhere in color-management literature or in the web)
// are TRANSPOSED (column-major).
// Since we stick to Imath::Matrix44 here, we have to use row-major order, so that matrix inversions and
// color transforms work as expected.
//
static
Imath::Matrix44 <float> makeColorMatrix(float wh_x, float wh_y,
					float prim_red_x, float prim_red_y,
					float prim_green_x, float prim_green_y,
					float prim_blue_x, float prim_blue_y)
{
	// Primaries:
	Imath::Color3 <float> X(prim_red_x / prim_red_y,
				prim_green_x / prim_green_y,
				prim_blue_x / prim_blue_y);
	Imath::Color3 <float> Y(1.0, 1.0, 1.0);
	Imath::Color3 <float> Z((1.0 - prim_red_x - prim_red_y) / prim_red_y,
				(1.0 - prim_green_x - prim_green_y) / prim_green_y,
				(1.0 - prim_blue_x - prim_blue_y) / prim_blue_y);

	// Transposed compared to brucelindbloom.com!
	Imath::Matrix44 <float> m1(X.x, Y.x, Z.x, 0,
				   X.y, Y.y, Z.y, 0,
				   X.z, Y.z, Z.z, 0,
				   0,   0,   0,   1);

	// brucelindbloom.com does not mention how to make Xw, Yw, Xw out of two white point values.
	// Here is how it's done:
	Imath::Color3 <float> W(wh_x / wh_y, 1.0, (1.0 - wh_x - wh_y) / wh_y);
	Imath::Color3 <float> S = W * m1.inverse();

	// Transposed compared to brucelindbloom.com!
 	return Imath::Matrix44 <float>(S.x * X.x, S.x * Y.x, S.x * Z.x, 0,
 					 S.y * X.y, S.y * Y.y, S.y * Z.y, 0,
 					 S.z * X.z, S.z * Y.z, S.z * Z.z, 0,
 					 0,         0,         0,         1);
}

//
// Using standard DCI P3 and Rec709 white point and R/G/B primaries,
// generate the p3->rec709 conversion matrix:
//
static
Imath::Matrix44 <float> makeP3ToSRGBMatrix()
{
	// Standard sRGB white point and primaries:
	Imath::Matrix44 <float> srgb = makeColorMatrix(0.3127, 0.329,	// white point: x, y
							0.640, 0.330,	// R.x, R.y
							0.300, 0.600,	// G.x, G.y
							0.150, 0.060);	// B.x, B.y

	// Standard DCI P3 white point and primaries:
	Imath::Matrix44 <float> p3 = makeColorMatrix(0.314, 0.351,	// white point: x, y
							0.675, 0.317,	// R.x, R.y
							0.261, 0.680,	// G.x, G.y
							0.150, 0.060);	// B.x, B.y

	return p3.inverse() * srgb;
}

//
// Transform an RGB color using a 3x3 part of a 4x4 matrix
//
static
void rgbTimesMatrix(float rgb[3], const Imath::Matrix44 <float>& colorMatrix, bool clampNegative=true)
{
	Imath::Color3 <float> col(rgb[0], rgb[1], rgb[2]);

	col *= colorMatrix;

	if (clampNegative) {
		col.x = std::max(0.0f, col.x);
		col.y = std::max(0.0f, col.y);
		col.z = std::max(0.0f, col.z);
	}

	rgb[0] = col.x;
	rgb[1] = col.y;
	rgb[2] = col.z;
}

//
// sRGB and Rec709 gamma curves have something in common:
// a linear segment around zero and a power curve outside that linear area:
//
#define CURVE_TO_LIN(__f, __thresh, __a, __b, __c) \
			((__f) > (__thresh) ? \
			pow(((__f) + (__a)) / (1.0 + (__a)), __c) : \
			((__f) / (__b)))

#define CURVE_FROM_LIN(__f, __thresh, __a, __b, __c) \
			((__f) > (__thresh) ? \
			(1 + (__a)) * pow((__f), 1.0 / (__c)) - (__a) : \
			((__f) * (__b)))


//
// The "worker" class
//
class ThreadedLinearizer {
	float* _pixels;
	int _width;
	int _height;
	int _nchannels;
	int _rgbChanIndex[3];		// indices of R, G, and B channels
	int _alphaChanIndex;		// index of alpha channel, or -1 if not known
	bool _unassociatedAlpha;	// colors have not been premultipled by alpha?
	bool _reverse;			// reverse conversion: do linear-to-space instead
	int _nthreads;	// # of threads to use
	int _nrunning;	// # of running threads - each thread increments that, startign from 0

	Space _space;
	float _gamma;
	Imath::Matrix44 <float> _p3ToRgbMatrix;

	boost::mutex _mutex;

	/*
	Linearization routines.
	Formulae from the following sources:
	sRGB: http://en.wikipedia.org/wiki/SRGB
	BT.709: http://www.itu.int/rec/R-REC-BT.709-5-200204-I/en
	*/

	static
	float _sRgbToLinear(float f, bool reverse=false) {
		const float thresh = reverse ? 0.0031308 : 0.04045;
		const float a = 0.055;
		const float b = 12.92;
		const float c = 2.4;
		return reverse ? CURVE_FROM_LIN(f, thresh, a, b, c) : CURVE_TO_LIN(f, thresh, a, b, c);
	}

	static
	float _rec709ToLinear(float f, bool reverse=false) {
		const float thresh = reverse ? 0.018 : 0.0801;
		const float a = 0.099;
		const float b = 4.5;
		const float c = 2.2222;
		return reverse ? CURVE_FROM_LIN(f, thresh, a, b, c) : CURVE_TO_LIN(f, thresh, a, b, c);
	}

	static
	float _gammaToLinear(float f, float gamma, bool reverse=false) {
		return f >= 0 ? pow(f, reverse ? 1.0 / gamma : gamma) : f;
	}

	static
	void _p3ToRgb(const float src[3], float dst[3],
			const Imath::Matrix44 <float> &colorMatrix, bool reverse=false) {
		const float GAMMA = 2.6;

		if (reverse) {
	 		dst[0] = src[0];
	 		dst[1] = src[1];
	 		dst[2] = src[2];

	 		rgbTimesMatrix(dst, colorMatrix);	// pre-inverted!

 			dst[0] = _gammaToLinear(dst[0], GAMMA, reverse);
 			dst[1] = _gammaToLinear(dst[1], GAMMA, reverse);
 			dst[2] = _gammaToLinear(dst[2], GAMMA, reverse);
		} else {
			dst[0] = _gammaToLinear(src[0], GAMMA, reverse);
			dst[1] = _gammaToLinear(src[1], GAMMA, reverse);
			dst[2] = _gammaToLinear(src[2], GAMMA, reverse);

	 		rgbTimesMatrix(dst, colorMatrix);
		}
	}

public:
	ThreadedLinearizer(float* pixels, int width, int height, int nchannels,
			int alphaChanIndex, bool unassociatedAlpha,
			Space space,
			float gamma,
			const Imath::Matrix44 <float> &p3ToRgbMatrix,
			bool reverse,
			int nthreads) {
		_pixels = pixels;
		_width = width;
		_height = height;
		_nchannels = nchannels;
		_alphaChanIndex = alphaChanIndex;
		_unassociatedAlpha = unassociatedAlpha;
		_nthreads = nthreads;
		_space = space;
		_gamma = gamma;
		_nrunning = 0;
		_reverse = reverse;
		_p3ToRgbMatrix = p3ToRgbMatrix;

		if (reverse)
			_p3ToRgbMatrix.invert();

// 		std::cerr << "m: " << _p3ToRgbMatrix << std::endl;

		// prepare indices to point R, G, and B values to corresponding image channels:
		_rgbChanIndex[0] = _rgbChanIndex[1] = _rgbChanIndex[2] = 0;

		int j = 0;

		for (int i = 0; j < std::min(3, _nchannels); ++i) {
			if (i != _alphaChanIndex)
				_rgbChanIndex[j++] = i;
		}
	};

	//
	// Linearize a color - a group of 1, 3, or 4 floats pointed to by ptr:
	//
	inline void linearizeColor(float* ptr) {
		float alpha = 1.0;

		if (_alphaChanIndex >= 0)
			alpha = ptr[_alphaChanIndex];

		bool useUnpremult = (alpha > 1e-6 && !_unassociatedAlpha);

		// Collect 3 float (RGB or XYZ color) values from the image:
		float inColor[3] = {0.0, 0.0, 0.0};

		for (int i = 0; i < 3; ++i) {
			inColor[i] = ptr[_rgbChanIndex[i]];

			// unpremult if necessary:
			if (useUnpremult)
				inColor[i] /= alpha;
		}

		float outColor[3];

		// linearize:
		// NOTE: p3 conversion involves color transformations, so its cannot be linearized
		// independently.
		// Instead, it needs all 3 RGB (actually, XYZ) values at once.
		if (_space == kP3) {
			_p3ToRgb(inColor, outColor, _p3ToRgbMatrix, _reverse);
		} else {
			for (int i = 0; i < 3; ++i) {
				if (_space == kSrgb)
					outColor[i] = _sRgbToLinear(inColor[i], _reverse);
				else if (_space == kRec709)
					outColor[i] = _rec709ToLinear(inColor[i], _reverse);
				else if (_space == kGamma)
					outColor[i] = _gammaToLinear(inColor[i], _gamma, _reverse);
			}
		}

		// premult back:
		if (useUnpremult) {
			for (int i = 0; i < 3; ++i)
				outColor[i] *= alpha;
		}

		// Store back to the image:
		for (int i = 0; i < 3; ++i)
			ptr[_rgbChanIndex[i]] = outColor[i];
	}

	//
	// Method to be run in multiple threads:
	//
	void threadFunc() {
		_mutex.lock();
		int thisThreadCount = _nrunning++;
		_mutex.unlock();

		int npixels = _width * _height;

		// Detect the range of pixels this thread is to operate on:
		int pixel0 = npixels * thisThreadCount / _nthreads;
		int pixel1 = npixels * (thisThreadCount + 1) / _nthreads;

		float *curPtr = _pixels + pixel0 * _nchannels;
		float *endPtr = _pixels + pixel1 * _nchannels;

		// Loop over all pixels in the range... curPtr points to each pixel's group of _nchannels floats:
		for (; curPtr < endPtr; curPtr += _nchannels)
			linearizeColor(curPtr);
	}

	//
	// Create the threads, run them and wait until they are done.
	//
	void start() {
		std::vector <boost::thread*> threads;
		threads.resize(_nthreads);

		for (int i = 0; i < _nthreads; ++i)
			threads[i] = new boost::thread(boost::bind(&ThreadedLinearizer::threadFunc, this));

		for (int i = 0; i < _nthreads; ++i)
			threads[i]->join();
	}
};

static
int doIt(const char* infile, const char* outfile, Space space, float gamma, bool reverse, int nthreads)
{
	// read input image:
	ImageInput *in = ImageInput::create(infile);

	if (!in) {
		fprintf(stderr, "linearize: input image %s cannot be read\n", infile);
		return 20;
	}

	if (reverse)
		fprintf(stderr, "linearize: reverse (debugging) mode in use!\n");

	ImageSpec spec;
	in->open(infile, spec);
	int npixels = spec.width * spec.height * spec.nchannels;
	float *pixels = new float[npixels];

	if (!pixels) {
		fprintf(stderr, "linearize: cannot allocate pixel buffer (%ld bytes)\n",
				(long)npixels*sizeof(float));
		delete in;
		return 21;
	}

	in->read_image(TypeDesc::FLOAT, pixels);
	in->close();

	int alphaChanIndex = spec.alpha_channel;
	bool unassociatedAlpha = false;

	if (alphaChanIndex == -1) {
		// OpenImageIO would not find unassociated alpha channel. Let's guess it by name then:
		for (int ch = 0; ch < spec.nchannels; ++ch) {
			if (spec.channelnames[ch].find("channel3") != std::string::npos) {
				alphaChanIndex = ch;
				unassociatedAlpha = true;
				break;
			}
		}
	}

	if (alphaChanIndex == -1) {
		fprintf(stderr, "linearize: no alpha channel\n");
	} else {
		fprintf(stderr, "linearize: %s alpha channel %d ('%s') found\n",
			unassociatedAlpha ? "unassociated" : "associated",
			alphaChanIndex, spec.channelnames[alphaChanIndex].c_str());
	}

	// Instantiate the multitheader class and run it:
	ThreadedLinearizer theDude(pixels, spec.width, spec.height, spec.nchannels,
				   alphaChanIndex, unassociatedAlpha, space, gamma,
				   makeP3ToSRGBMatrix(), reverse,
				   nthreads);
	theDude.start();

	// write output image:
	ImageOutput *out = ImageOutput::create(outfile);

	if (!out) {
		fprintf(stderr, "linearize: output image %s cannot be created\n", outfile);
		delete in;
		delete [] pixels;
		return 22;
	}

	// Promote 8-bit image to 16 bit int to avoid banding artifacts.
	// If the output format doesn't supports it, OIIO will just switch to something more appropriate.
	if (spec.format.basetype == TypeDesc::CHAR || spec.format.basetype == TypeDesc::UCHAR) {
		spec.format.basetype = TypeDesc::USHORT;

		/*
		// OIIO 1.0+ has deprecated quantization!
		// Quantization should be set according to the new spec.format:
		QuantizationSpec quantSpec(spec.format);
		spec.quant_black = quantSpec.quant_black;
		spec.quant_white = quantSpec.quant_white;
		spec.quant_min = quantSpec.quant_min;
		spec.quant_max = quantSpec.quant_max;
		*/
	}

	out->open(outfile, spec);
	out->write_image(TypeDesc::FLOAT, pixels);
	out->close();

	delete in;
	delete out;
	delete [] pixels;
	return 0;
}

int main(int argc, char *argv[])
{
	if (argc < 3) {
		fprintf(stderr, "%s:\n\n", argv[0]);
		usage();
		exit(1);
	}

	Space space = kUnknown;
	float gamma = 0;
	bool reverse = false;
	int nthreads = 4;

	const char* infile = NULL;
	const char* outfile = NULL;

	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-threads")) {
			if (++i >= argc) {
				fprintf(stderr, "linearize: not enough arguments in command line\n");
				exit(2);
			}

			nthreads = atoi(argv[i]);

			if (nthreads < 1) {
				fprintf(stderr, "linearize: -threads must be followed by a positive integer\n");
				exit(3);
			}
		} else if (!strcmp(argv[i], "-space")) {
			if (++i >= argc) {
				fprintf(stderr, "linearize: not enough arguments in command line\n");
				exit(2);
			}

			if (gamma != 0) {
				fprintf(stderr, "linearize: -space and -gamma cannot be used the same time\n");
				exit(4);
			}

			if (!strcmp(argv[i], "srgb"))
				space = kSrgb;
			else if (!strcmp(argv[i], "rec709"))
				space = kRec709;
			else if (!strcmp(argv[i], "p3"))
				space = kP3;
			else {
				fprintf(stderr, "linearize: unknown -space argument: %s\n", argv[i]);
				exit(5);
			}
		} else if (!strcmp(argv[i], "-gamma")) {
			if (++i >= argc) {
				fprintf(stderr, "linearize: not enough arguments in command line\n");
				exit(2);
			}

			if (space != kUnknown) {
				fprintf(stderr, "linearize: -space and -gamma cannot be used the same time\n");
				exit(4);
			}

			gamma = atof(argv[i]);

			if (gamma == 0) {
				fprintf(stderr, "linearize: -gamma should be followed by a non-zero float, got %s\n", argv[i]);
				exit(6);
			}

			space = kGamma;
		} else if (!strcmp(argv[i], "-r")) {
			reverse = true;
		} else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help") || !strcmp(argv[i], "--help")) {
			usage();
			exit(0);
		} else if (*argv[i] == '-') {
			fprintf(stderr, "linearize: unknown flag: %s\n", argv[i]);
			exit(7);
		} else {
			if (!infile)
				infile = argv[i];
			else if (!outfile)
				outfile = argv[i];
			else {
				fprintf(stderr, "linearize: extra argument in command line: %s\n", argv[i]);
				exit(8);
			}
		}
	}

	if (space == kUnknown)
		space = kSrgb;

	if (!infile) {
		fprintf(stderr, "linearize: input image file not specified\n");
		exit(9);
	}

	if (!outfile) {
		fprintf(stderr, "linearize: ouput image file not specified\n");
		exit(10);
	}

	if (!strcmp(infile, outfile)) {
		fprintf(stderr, "linearize: input and ouput image files cannot be the same\n");
		exit(11);
	}

	if (access(infile, R_OK)) {
		fprintf(stderr, "linearize: %s: input file cannot be accessed\n", infile);
		exit(12);
	}

	int result = doIt(infile, outfile, space, gamma, reverse, nthreads);

	if (result)
		fprintf(stderr, "linearize: output image %s can be corrupt\n", outfile);

	exit(result);
}
