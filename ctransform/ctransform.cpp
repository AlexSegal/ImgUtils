// ----------------------------------------------------------------------------
//
// Copyright (C) 2007-13 Alex Segal, Vladimir Monahov. All rights reserved.
// Use this source code "AS IS" with no warranty whatsoever
//
// $Author: asegal $
// $Rev: 1479 $
// $Date: 2013-07-13 17:21:32 -0700 (Sat, 13 Jul 2013) $
//
// ----------------------------------------------------------------------------

/*
This is the color space transformation utility. What does it do?
1. Read an an arbitrary input image,
2. Convert its pixel values to a linear domain by removing a gamma (or a log) curve,
3. Transform linear color values to a different (but linear) color space,
4. Convert pixel values to non-linear domain by applying an output gamma (or log) curve,
5. Save resulting pixels to a new image file.

Steps 2 through 4 are optional.
Supported image formats are the ones OpenImageIO can read and write.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <string>
#include <boost/thread.hpp>

#include "OpenEXR/ImathMatrix.h"
#include "OpenEXR/ImathMatrixAlgo.h"
#include "OpenEXR/ImathColor.h"

#include "OpenImageIO/imageio.h"

// Uncomment the #define below to use HP DreamColor primaries vs standard DCI P3 ones:
//#define USE_HP_P3_PRIMARIES


static
void usage()
{
	fprintf(stderr,
		"Usage:\n"
			"\tctransform [options] <input_file> <output_file>\n\n"
		"options are:\n"
		"\t-fromspace srgb|p3|xyz|<wx wy rx ry gx gy bx bz>\n"
		"\t-fromcurve srgb|rec709|p3|log|linear|<float_gamma>\n"
		"\t-tospace srgb|p3|xyz|<wx wy rx ry gx gy bx bz>\n"
		"\t-tocurve srgb|rec709|p3|log|linear|<float_gamma>\n"
		"\t-threads <int>\n\n"
		"\tdefault options are: -fromspace srgb -fromcurve srgb "
			"-tospace srgb -tocurve 1.0 -threads <# of cpu cores>\n\n");

}

enum Space {
	kUnknownSpace = -1,
	kSrgbSpace,
	kP3Space,
	kXYZSpace,
	kCustomSpace
};

enum XferFunc {
	kUnknownXferFunc = -1,
	kLinearXferFunc,
	kSrgbXferFunc,
	kRec709XferFunc,
	kP3XferFunc,
	kLogXferFunc,
	kCustomXferFunc
};

// Standard sRGB white point and primaries:
const float SRGB_WP_PRIMS[8] = {0.3127, 0.329,	// white point: x, y
				0.640,  0.330,	// R.x, R.y
				0.300,  0.600,	// G.x, G.y
				0.150,  0.060};	// B.x, B.y

#ifdef USE_HP_P3_PRIMARIES
// HP DreamColor's P3 white point and primaries:
const float P3_WP_PRIMS[8] =   {0.314, 0.351,
				0.675, 0.317,
				0.261, 0.680,
				0.150, 0.060};
#else
// Standard DCI P3 white point and primaries:
const float P3_WP_PRIMS[8] =   {0.314, 0.351,
				0.680, 0.320,
				0.265, 0.690,
				0.150, 0.060};
#endif

OIIO_NAMESPACE_USING

//
// Make an XYZ reference value given its x and y.
// NOTE: brucelindbloom.com does not mention how to make it, however he uses
// the same method to calculate Zr, Zg, Zb for RGB->XYZ matrices.
//
inline
Imath::Color3f primToXYZ(float x, float y)
{
	return Imath::Color3f(x / y, 1.0, (1.0 - x - y) / y);
}

//
// Create a 4x4 matrix (with only 3x3 values actually used) for the given white point and R/G/B primaries.
// Multiplying an RGB color by it will transform it to XYZ color space.
// The algorithm is based on formulas in http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
// with somewhat cleaner and more logical way of building the matrices.
// NOTE: all matrices in brucelindbloom.com (and everywhere in color-management literature or on the web)
// are TRANSPOSED (column-major).
// Since we stick to Imath::M44f here, we have to use row-major order, so that matrix inversions and
// color transforms work as expected.
//
static
Imath::M44f makeColorMatrix(float wh_x, float wh_y,
			float prim_red_x, float prim_red_y,
			float prim_green_x, float prim_green_y,
			float prim_blue_x, float prim_blue_y)
{
	// Primaries:
	Imath::Color3f Rp = primToXYZ(prim_red_x, prim_red_y);
	Imath::Color3f Gp = primToXYZ(prim_green_x, prim_green_y);
	Imath::Color3f Bp = primToXYZ(prim_blue_x, prim_blue_y);

	Imath::M44f m1( Rp.x, Rp.y, Rp.z, 0,
			Gp.x, Gp.y, Gp.z, 0,
			Bp.x, Bp.y, Bp.z, 0,
			0,    0,    0,    1);

	// White point:
	Imath::Color3f W = primToXYZ(wh_x, wh_y);
	Imath::Color3f S = W * m1.inverse();

	return Imath::M44f(S.x, 0, 0, 0,
			   0, S.y, 0, 0,
			   0, 0, S.z, 0,
			   0, 0,   0, 1) * m1;
}

//
// Same as above but with 8 floats (white point and primaries) passed via an array:
//
static
Imath::M44f makeColorMatrix(const float wpr[8])
{
	return makeColorMatrix(wpr[0], wpr[1], wpr[2], wpr[3],
			       wpr[4], wpr[5], wpr[6], wpr[7]);
}

//
// Generate Bradford's chromatic adaptation matrix for the given
// 'source' and 'destination' white points.
// See the explanantion at: http://www.brucelindbloom.com/Eqn_ChromAdapt.html
//
static
Imath::M44f makeChromAdaptMatrix(float wh_x_s, float wh_y_s, float wh_x_d, float wh_y_d)
{
	// Bradford matrix:
	const Imath::M44f Ma = Imath::M44f( 0.8951, 0.2664, -0.1614, 0,
					   -0.7502, 1.7135,  0.0367, 0,
					    0.0389, -0.0685, 1.0296, 0,
					    0,	    0,	     0,	     1).transposed();

	// Reference whites: source and destination
	Imath::Color3f pyb_s = primToXYZ(wh_x_s, wh_y_s) * Ma;
	Imath::Color3f pyb_d = primToXYZ(wh_x_d, wh_y_d) * Ma;

	Imath::M44f scale(pyb_d.x/pyb_s.x, 0, 0, 0,
			  0, pyb_d.y/pyb_s.y, 0, 0,
			  0, 0, pyb_d.z/pyb_s.z, 0,
			  0, 0, 0,               1);

	// I spent a few hours on this one. According to brucelindbloom.com
	// the multiplication order should be reverse!
	// Hell I don't know why but it doesn't work that way :(
	return Ma * scale * Ma.inverse();
}

//
// Transform an RGB color using a 3x3 part of a 4x4 matrix
//
static
Imath::Color3f colorTimesMatrix(const Imath::Color3f& rgb, const Imath::M44f& colorMatrix, bool clampNegative=false)
{
	Imath::Color3f result = rgb * colorMatrix;

	if (clampNegative) {
		result.x = std::max(0.0f, result.x);
		result.y = std::max(0.0f, result.y);
		result.z = std::max(0.0f, result.z);
	}

	return result;
}

//
// Same as above, but using a 3-element float array as color
//
static
void rgbTimesMatrix(float rgb[3], const Imath::M44f& colorMatrix, bool clampNegative=false)
{
	Imath::Color3f col(rgb[0], rgb[1], rgb[2]);

	col = colorTimesMatrix(col, colorMatrix, clampNegative);

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
class ThreadedTransformer {
	float* _pixels;
	int _width;
	int _height;
	int _nchannels;
	int _rgbChanIndex[3];		// indices of R, G, and B channels
	int _alphaChanIndex;		// index of alpha channel, or -1 if not known
	bool _unassociatedAlpha;	// colors have not been premultipled by alpha?
	int _nthreads;			// # of threads to use
	int _nrunning;			// # of running threads - each thread increments that, starting from 0

	XferFunc _fromXferFunc;
	float _fromCustomGamma;

	Imath::M44f _colorXformMatrix;
	bool  _matrixIsIdentity;	// to speed up calculations if there is no transform to be made:

	XferFunc _toXferFunc;
	float _toCustomGamma;

	boost::mutex _mutex;

	/*
	Linearization routines.
	Formulae from the following sources:
	sRGB: http://en.wikipedia.org/wiki/SRGB
	BT.709: http://www.itu.int/rec/R-REC-BT.709-5-200204-I/en
	*/
	static
	void _sRgbXferFuncToLinear(float *f, int count=1, bool reverse=false) {
		static const float thresh = reverse ? 0.0031308 : 0.04045;
		static const float a = 0.055;
		static const float b = 12.92;
		static const float c = 2.4;

		for (int i = 0; i < count; ++i, ++f)
			*f = reverse ? CURVE_FROM_LIN(*f, thresh, a, b, c) : CURVE_TO_LIN(*f, thresh, a, b, c);
	}

	static
	void _rec709XferFuncToLinear(float *f, int count=1, bool reverse=false) {
		static const float thresh = reverse ? 0.018 : 0.0801;
		static const float a = 0.099;
		static const float b = 4.5;
		static const float c = 2.2222;

		for (int i = 0; i < count; ++i, ++f)
			*f = reverse ? CURVE_FROM_LIN(*f, thresh, a, b, c) : CURVE_TO_LIN(*f, thresh, a, b, c);
	}

	static
	void _customGammaToLinear(float *f, float gamma, int count=1, bool reverse=false) {
		if (reverse && gamma > 1e-6)
			gamma = 1.0 / gamma;

		for (int i = 0; i < count; ++i, ++f)
			if (*f >= 0)
				*f = pow(*f, gamma);
	}

	static
	void _p3XferFuncToLinear(float *f, int count=1, bool reverse=false) {
		float P3_GAMMA = 2.6;
		_customGammaToLinear(f, P3_GAMMA, count, reverse);
	}

	static
	void _logXferFuncToLinear(float *f, int count=1, bool reverse=false) {
		static const int MAXVALUE = 1 << 10;
		static const int REF_WHITE = 685;
		static const int REF_BLACK = 95;
		static const float COEF = 0.002 / 0.6;
		static const float DISP_GAMMA = 1.7;
		static const float gamma = DISP_GAMMA / 1.7;
		static const float gain = 1.0 / pow((1 - pow(10, (REF_BLACK - REF_WHITE) * COEF)), gamma);
		static const float offset = gain - 1.0;

		for (int i = 0; i < count; ++i, ++f) {
			if (reverse) {
			    *f = REF_WHITE + log10(pow((*f + offset) / gain, 1 / gamma)) / COEF;
			    *f /= MAXVALUE;
			} else {
			    float in = *f * MAXVALUE;
			    *f = pow(10, pow((in - REF_WHITE) * COEF, gamma)) * gain - offset;
			}
		}
	}

	//
	// Choose the right curve and apply:
	//
	static
	void _applyXferFunc(float *f, int count, XferFunc type, float customGamma=1.0, bool inverse=false) {
		switch (type) {
			default:
			case kLinearXferFunc:
				break;

			case kSrgbXferFunc:
				_sRgbXferFuncToLinear(f, count, inverse);
				break;

			case kRec709XferFunc:
				_rec709XferFuncToLinear(f, count, inverse);
				break;

			case kP3XferFunc:
				_p3XferFuncToLinear(f, count, inverse);
				break;

			case kLogXferFunc:
				_logXferFuncToLinear(f, count, inverse);
				break;

			case kCustomXferFunc:
				_customGammaToLinear(f, customGamma, count, inverse);
				break;
		}
	}

public:
	ThreadedTransformer(float* pixels, int width, int height, int nchannels,
			int alphaChanIndex, bool unassociatedAlpha,
			XferFunc fromXferFunc,
			float fromCustomGamma,
			const Imath::M44f &colorXformMatrix,
			XferFunc toXferFunc,
			float toCustomGamma,
			int nthreads) {
		_pixels = pixels;
		_width = width;
		_height = height;
		_nchannels = nchannels;
		_alphaChanIndex = alphaChanIndex;
		_unassociatedAlpha = unassociatedAlpha;
		_nthreads = nthreads;
		_nrunning = 0;

		_fromXferFunc = fromXferFunc;
		_fromCustomGamma = fromCustomGamma;
		_colorXformMatrix = colorXformMatrix;
		_matrixIsIdentity = (_colorXformMatrix.equalWithAbsError(Imath::identity44f, 1e-5));

		_toXferFunc = toXferFunc;
		_toCustomGamma = toCustomGamma;

		// prepare indices to point R, G, and B values to corresponding image channels:
		_rgbChanIndex[0] = _rgbChanIndex[1] = _rgbChanIndex[2] = 0;

		int j = 0;

		for (int i = 0; j < std::min(3, _nchannels); ++i) {
			if (i != _alphaChanIndex)
				_rgbChanIndex[j++] = i;
		}
	};

	//
	// Transform a color - a group of 1, 3, or 4 floats pointed to by ptr:
	//
	inline void xformColor(float* ptr, int count=3) {
		float alpha = 1.0;

		if (_alphaChanIndex >= 0)
			alpha = ptr[_alphaChanIndex];

		bool useUnpremult = (alpha > 1e-6 && !_unassociatedAlpha);

		// Collect 3 float (RGB or XYZ color) values from the image:
		float color[count];

		for (int i = 0; i < count; ++i)
			color[i] = ptr[_rgbChanIndex[i]];

		// step 1: unpremult by alpha if necessary:
		if (useUnpremult) {
			for (int i = 0; i < count; ++i)
				color[i] /= alpha;
		}

		// step 2: apply fromXferFunc:
		_applyXferFunc(color, count, _fromXferFunc, _fromCustomGamma, false);

		// step 3: apply matrix transform, if it's not an identity matrix:
		if (!_matrixIsIdentity)
			rgbTimesMatrix(color, _colorXformMatrix, true);

		// step 4: apply toXferFunc:
		_applyXferFunc(color, count, _toXferFunc, _toCustomGamma, true);

		// step 5: premult back (if necessary):
		if (useUnpremult) {
			for (int i = 0; i < 3; ++i)
				color[i] *= alpha;
		}

		// Store back to the image:
		for (int i = 0; i < 3; ++i)
			ptr[_rgbChanIndex[i]] = color[i];
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

		int nRgbChannels = std::min(3, _alphaChanIndex < 0 ? _nchannels : _nchannels - 1);

		// Loop over all pixels in the range... curPtr points to each pixel's group of _nchannels floats:
		for (; curPtr < endPtr; curPtr += _nchannels)
			xformColor(curPtr, nRgbChannels);
	}

	//
	// Create threads, run them and wait until they are done.
	//
	void start() {
		std::vector <boost::thread*> threads;
		threads.resize(_nthreads);

		for (int i = 0; i < _nthreads; ++i)
			threads[i] = new boost::thread(boost::bind(&ThreadedTransformer::threadFunc, this));

		for (int i = 0; i < _nthreads; ++i)
			threads[i]->join();
	}
};

static
int doIt(const char* infile, const char* outfile,
	XferFunc fromXferFunc, float fromCustomGamma,
	const Imath::M44f &colorXformMatrix,
	XferFunc toXferFunc, float toCustomGamma,
	int nthreads)
{
	// read input image:
	ImageInput *in = ImageInput::create(infile);

	if (!in) {
		fprintf(stderr, "ctransform: input image %s cannot be read\n", infile);
		return 20;
	}

	ImageSpec spec;
	in->open(infile, spec);
	int npixels = spec.width * spec.height * spec.nchannels;
	float *pixels = new float[npixels];

	if (!pixels) {
		fprintf(stderr, "ctransform: cannot allocate pixel buffer (%ld bytes)\n",
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
		fprintf(stderr, "ctransform: no alpha channel\n");
	} else {
		fprintf(stderr, "ctransform: %s alpha channel %d ('%s') found\n",
			unassociatedAlpha ? "unassociated" : "associated",
			alphaChanIndex, spec.channelnames[alphaChanIndex].c_str());
	}

	// Instantiate the multitheader class and run it:
	ThreadedTransformer theDude(pixels, spec.width, spec.height, spec.nchannels,
				    alphaChanIndex, unassociatedAlpha,
				    fromXferFunc, fromCustomGamma,
				    colorXformMatrix,
				    toXferFunc, toCustomGamma,
				    nthreads);
	theDude.start();

	// write output image:
	ImageOutput *out = ImageOutput::create(outfile);

	if (!out) {
		fprintf(stderr, "ctransform: output image %s cannot be created\n", outfile);
		delete in;
		delete [] pixels;
		return 22;
	}

	std::string outfilestr(outfile);

	if (outfilestr.rfind(".dpx") == outfilestr.size() - 4) {
		spec.format = TypeDesc::UNKNOWN; // let OIIO select the most appropriate format
	} else if (spec.format.basetype == TypeDesc::CHAR || spec.format.basetype == TypeDesc::UCHAR) {
		// Promote 8-bit image to 16 bit int to avoid banding artifacts.
		// If the output format doesn't supports it, OIIO will just switch to something more appropriate.
		spec.format.basetype = TypeDesc::SHORT;

		// Quantization should be set according to the new spec.format:
		// OIIO > 1.0 has deprecated QuantizationSpec altogether...
		/*
		QuantizationSpec quantSpec(spec.format);
		spec.quant_black = quantSpec.quant_black;
		spec.quant_white = quantSpec.quant_white;
		spec.quant_min = quantSpec.quant_min;
		spec.quant_max = quantSpec.quant_max;
		*/
		// DPX:
		//spec.attribute("oiio:BitsPerSample", 10);
		//spec.attribute("oiio:Endian", "big");
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

	Space fromSpace = kUnknownSpace;
	Space toSpace = kUnknownSpace;
	XferFunc fromXferFunc = kLinearXferFunc;
	XferFunc toXferFunc = kLinearXferFunc;

	float fromCustomGamma = 1.0;
	float toCustomGamma = 1.0;
	float fromCustomWpPrims[6];
	float toCustomWpPrims[6];

	int nthreads = std::max(1, static_cast <int>(boost::thread::hardware_concurrency()));

	const char* infile = NULL;
	const char* outfile = NULL;

	for (int i = 1; i < argc; ++i) {
		if (!strcmp(argv[i], "-threads")) {
			if (++i >= argc) {
				fprintf(stderr, "ctransform: not enough arguments in command line\n");
				exit(2);
			}

			nthreads = atoi(argv[i]);

			if (nthreads < 1) {
				fprintf(stderr, "ctransform: -threads must be followed by a positive integer\n");
				exit(3);
			}
		} else if (!strcmp(argv[i], "-fromspace") || !strcmp(argv[i], "-tospace")) {
			if (++i >= argc) {
				fprintf(stderr, "ctransform: not enough arguments in command line after %s\n", argv[i-1]);
				exit(2);
			}

			Space space = kUnknownSpace;
			const int NFLOATS = 8;

			float floats[NFLOATS];

			if (!strcmp(argv[i], "srgb"))
				space = kSrgbSpace;
			else if (!strcmp(argv[i], "p3"))
				space = kP3Space;
			else if (!strcmp(argv[i], "xyz"))
				space = kXYZSpace;
			else {
				space = kCustomSpace;

				for (int j = 0; j < NFLOATS; ++j) {
					if (i + j >= argc || !sscanf(argv[i+j], "%f", floats+j)) {
						fprintf(stderr, "ctransform: a known color space name or %d floats expected in command line after %s\n",
								NFLOATS, argv[i-1]);
						exit(5);
					}
				}
			}

			if (!strcmp(argv[i-1], "-fromspace")) {
				fromSpace = space;
				for (int j = 0; j < NFLOATS; ++j)
					fromCustomWpPrims[j] = floats[j];
			} else {
				toSpace = space;
				for (int j = 0; j < NFLOATS; ++j)
					toCustomWpPrims[j] = floats[j];
			}
		} else if (!strcmp(argv[i], "-fromcurve") || !strcmp(argv[i], "-tocurve")) {
			if (++i >= argc) {
				fprintf(stderr, "ctransform: not enough arguments in command line after %s\n", argv[i-1]);
				exit(2);
			}

			XferFunc xfunc = kUnknownXferFunc;
			float customGamma = 0;

			if (!strcmp(argv[i], "srgb"))
				xfunc = kSrgbXferFunc;
			else if (!strcmp(argv[i], "rec709"))
				xfunc = kRec709XferFunc;
			else if (!strcmp(argv[i], "p3"))
				xfunc = kP3XferFunc;
			else if (!strcmp(argv[i], "log")) {
				xfunc = kLogXferFunc;
				customGamma = 1.0;
			} else if (!strcmp(argv[i], "linear")) {
				xfunc = kCustomXferFunc;
				customGamma = 1.0;
			} else {
				xfunc = kCustomXferFunc;
				customGamma = atof(argv[i]);

				if (customGamma == 0) {
					fprintf(stderr, "ctransform: a known curve name or a float > 0 expected after %s\n", argv[i-1]);
					exit(6);
				}
			}

			if (!strcmp(argv[i-1], "-fromcurve")) {
				fromXferFunc = xfunc;
				fromCustomGamma = customGamma;
			} else {
				toXferFunc = xfunc;
				toCustomGamma = customGamma;
			}
		} else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "-help") || !strcmp(argv[i], "--help")) {
			usage();
			exit(0);
		} else if (*argv[i] == '-') {
			fprintf(stderr, "ctransform: unknown flag: %s\n", argv[i]);
			exit(7);
		} else {
			if (!infile)
				infile = argv[i];
			else if (!outfile)
				outfile = argv[i];
			else {
				fprintf(stderr, "ctransform: extra argument in command line: %s\n", argv[i]);
				exit(8);
			}
		}
	}

	Imath::M44f colorXformMatrix;

	// Unspecified color space = sRGB!
	if (fromSpace == kUnknownSpace)
		fromSpace = kSrgbSpace;

	if (toSpace == kUnknownSpace)
		toSpace = kSrgbSpace;

	if (fromSpace != toSpace || fromSpace == kCustomSpace || toSpace == kCustomSpace) {
		// from one of RGB spaces to XYZ:
		Imath::M44f fromMatrix, toMatrix, chrAdaptMatrix;
		const float* fromWpPrims = NULL;
		const float* toWpPrims = NULL;

		if (fromSpace == kSrgbSpace)
			fromWpPrims = SRGB_WP_PRIMS;
		else if (fromSpace == kP3Space)
			fromWpPrims = P3_WP_PRIMS;
		else if (fromSpace == kCustomSpace)
			fromWpPrims = fromCustomWpPrims;

		// from XYZ to one of RGB spaces:
		if (toSpace == kSrgbSpace)
			toWpPrims = SRGB_WP_PRIMS;
		else if (toSpace == kP3Space)
			toWpPrims = P3_WP_PRIMS;
		else if (toSpace == kCustomSpace)
			toWpPrims = toCustomWpPrims;

		if (fromWpPrims)
			fromMatrix = makeColorMatrix(fromWpPrims);

		if (toWpPrims)
			toMatrix = makeColorMatrix(toWpPrims);

 		if (fromWpPrims && toWpPrims)
 			chrAdaptMatrix = makeChromAdaptMatrix(fromWpPrims[0], fromWpPrims[1],
 								toWpPrims[0], toWpPrims[1]);

		colorXformMatrix = fromMatrix * chrAdaptMatrix * toMatrix.inverse();
	}

	if (!infile) {
		fprintf(stderr, "ctransform: input image file not specified\n");
		exit(9);
	}

	if (!outfile) {
		fprintf(stderr, "ctransform: ouput image file not specified\n");
		exit(10);
	}

	if (!strcmp(infile, outfile)) {
		fprintf(stderr, "ctransform: input and ouput image files cannot be the same\n");
		exit(11);
	}

	if (access(infile, R_OK)) {
		fprintf(stderr, "ctransform: %s: input file cannot be accessed\n", infile);
		exit(12);
	}

	int result = doIt(infile, outfile,
			fromXferFunc, fromCustomGamma,
			colorXformMatrix,
			toXferFunc, toCustomGamma,
			nthreads);

	if (result)
		fprintf(stderr, "ctransform: output image %s can be corrupt\n", outfile);

	exit(result);
}
