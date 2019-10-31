// ----------------------------------------------------------------------------
//
// Copyright (C) 2007-13 Alex Segal, Vladimir Monahov. All rights reserved.
// Use this source code "AS IS" with no warranty whatsoever
//
// $Author: asegal $
// $Rev: 1442 $
// $Date: 2013-01-24 18:55:45 -0800 (Thu, 24 Jan 2013) $
//
// ----------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>
#include <stdint.h>

#include <string>
#include <vector>

#include <boost/regex.hpp>

#include <half.h>
#include <ImfOutputFile.h>
#include <ImfInputFile.h>
#include <ImfChannelList.h>

typedef std::vector<std::string> StringVector;
typedef std::pair <boost::regex, std::string> RenamePair;
typedef std::vector <RenamePair> RenameList;

void printUsage(const char* exe)
{
	fprintf(stderr, "%s: Process OpenEXR files \"in-place\" to improve Nuke performance\n"
			"by minimizing dataWindow size and setting individual scanline zip compression.\n\n"
			"Usage: %s [-r|-rename <search_regex> <new_name>] <exrfiles>\n\n"
			"-r|-rename: a POSIX regex search pattern and a replacement string can be used to rename channels.\n"
	                "\tNOTE: -rename can be used multiple times, but be aware of cascading rename effects it can cause.\n",
			exe, exe);
}

#if defined(_WINDOWS) || defined (_WIN32) || defined (_WIN64)
	static const char SLASH[] = "\\";
#else
	static const char SLASH[] = "/";
#endif

//
// A smple struct to be used in std::vector to keep track of input channels and their pixel data:
//
struct ChannelRecord {
	char* pixels;
	Imf::PixelType channelType;
	int pixelSize;
	std::string channelName;

	ChannelRecord() : pixels(NULL), pixelSize(0), channelType(Imf::FLOAT) {}
};


//
// Given the regexs->strings map, rename the channels:
//
static inline std::string renameChannel(const std::string &name, const RenameList& renameList)
{
	std::string result = name;

	for (RenameList::const_iterator iter = renameList.begin();
	     iter != renameList.end();
	     ++iter) {
		result = boost::regex_replace(result, iter->first, iter->second);
	}

	return result;
}

//
// Detect data bounding box and copy data to the new file
//
static void copyData(const char* inFileName, const char* outFileName, const RenameList& renameList)
{
	Imf::InputFile in(inFileName);

	const Imf::Header inHeader = in.header();
	Imath::Box2i inDataWindow = inHeader.dataWindow();
	int inWidth = inDataWindow.max.x - inDataWindow.min.x + 1;
	int inHeight = inDataWindow.max.y - inDataWindow.min.y + 1;

	const Imf::ChannelList &channels = inHeader.channels();

	std::vector<ChannelRecord> channelRecordList;

	Imf::FrameBuffer inFrameBuffer;

	//
	// Populate inFrameBuffer with all channels and pixel pointers,
	// and add all this information to channelRecordList:
	//
	for (Imf::ChannelList::ConstIterator iter = channels.begin(); iter != channels.end(); ++iter) {
		Imf::PixelType chanType = iter.channel().type;
		const char* chanName = iter.name();

		int pixelSize = sizeof (float); // also good for UINT

		if (chanType == Imf::HALF)
			pixelSize = sizeof (half);

		ChannelRecord record;

		record.pixels = new char[inWidth * inHeight * pixelSize];
		record.channelType = chanType;
		record.pixelSize = pixelSize;
		record.channelName = std::string(chanName);
		channelRecordList.push_back(record);

		inFrameBuffer.insert(chanName, Imf::Slice(chanType,
							record.pixels
								- pixelSize * inDataWindow.min.x
								- pixelSize * inDataWindow.min.y * inWidth,
							pixelSize * 1,
							pixelSize * inWidth));

	}

	// Read all channels at once:
	in.setFrameBuffer(inFrameBuffer);
	in.readPixels(inDataWindow.min.y, inDataWindow.max.y);

	int minX = inDataWindow.max.x;
	int maxX = inDataWindow.min.x;
	int minY = inDataWindow.max.y;
	int maxY = inDataWindow.min.y;

	// Now analyze pixels:
	for (std::vector<ChannelRecord>::const_iterator iter = channelRecordList.begin();
	     iter != channelRecordList.end();
	     ++iter) {
		const char* pixels = iter->pixels;
		Imf::PixelType chanType  = iter->channelType;

		float *floatPtr = (float*)pixels;
		half *halfPtr = (half*)pixels;
		uint32_t *uintPtr = (uint32_t*)pixels;

		for (int y = inDataWindow.min.y; y <= inDataWindow.max.y; ++y) {
			bool emptyScanline = true;

			for (int x = inDataWindow.min.x; x <= inDataWindow.max.x; ++x, ++floatPtr, ++halfPtr, ++uintPtr) {
				if (chanType == Imf::FLOAT && *floatPtr == 0)
					continue;
				else if (chanType == Imf::HALF && halfPtr->isZero())
					continue;
				else if (chanType == Imf::UINT && *uintPtr == 0)
					continue;

				emptyScanline = false;

				if (x < minX)
					minX = x;
				if (x > maxX)
					maxX = x;
			}

			if (!emptyScanline) {
				if (y < minY)
					minY = y;
				if (y > maxY)
					maxY = y;
			}
		}
	}

	Imath::Box2i outDataWindow = inDataWindow;

	if (minX == outDataWindow.max.x)
		outDataWindow.max.x = outDataWindow.min.x;
	else {
		outDataWindow.min.x = minX;
		outDataWindow.max.x = maxX;
	}

	if (minY == outDataWindow.max.y)
		outDataWindow.max.y = outDataWindow.min.y;
	else {
		outDataWindow.min.y = minY;
		outDataWindow.max.y = maxY;
	}

	if (outDataWindow == inDataWindow) {
		fprintf(stderr, "dataWindow left unchanged: %d %d - %d %d\n",
				inDataWindow.min.x, inDataWindow.min.y,
				inDataWindow.max.x, inDataWindow.max.y);
	} else {
		fprintf(stderr, "dataWindow modified: "
				"old: %d %d - %d %d; "
				"new: %d %d - %d %d\n",
				inDataWindow.min.x, inDataWindow.min.y,
				inDataWindow.max.x, inDataWindow.max.y,
				outDataWindow.min.x, outDataWindow.min.y,
				outDataWindow.max.x, outDataWindow.max.y);
	}

	//
	// Now at last write output:
	//
	int outHeight = outDataWindow.max.y - outDataWindow.min.y + 1;

	// Create output image inHeader, identical to the input one, except for
	// the outDataWindow and compression type
	Imf::Header outHeader(inHeader.displayWindow(),
				outDataWindow,
				inHeader.pixelAspectRatio(),
				inHeader.screenWindowCenter(),
				inHeader.screenWindowWidth(),
				inHeader.lineOrder(),
				Imf::ZIPS_COMPRESSION);

	const char *skipAttrs[] = {"channels", "dataWindow", "compression", "tiles", NULL};

	// Copy all attributes from inHeader except for those in skipAttrs:
	for (Imf::Header::ConstIterator iter = inHeader.begin(); iter != inHeader.end(); ++iter) {
		bool skip = false;

		for (const char **ptr = skipAttrs; *ptr; ++ptr) {
			if (!strcmp(iter.name(), *ptr)) {
				skip = true;
				break;
			}
		}

		if (!skip)
			outHeader.insert(iter.name(), iter.attribute());
	}

	Imf::FrameBuffer outFrameBuffer;

	// For every channel in input file: add it to outFrameBuffer:
	for (std::vector<ChannelRecord>::const_iterator iter = channelRecordList.begin();
	     iter != channelRecordList.end();
	     ++iter) {
		char* pixels = iter->pixels;
		Imf::PixelType chanType  = iter->channelType;
		int pixelSize = iter->pixelSize;

		// Rename the channel if necessary:
		std::string chanName = renameChannel(iter->channelName, renameList);

		outHeader.channels().insert(chanName.c_str(), Imf::Channel(chanType));

		outFrameBuffer.insert(chanName.c_str(),
				      Imf::Slice(chanType,
						pixels
							- pixelSize * inDataWindow.min.x
							- pixelSize * inDataWindow.min.y * inWidth,
						pixelSize * 1,
						pixelSize * inWidth));
	}

	// Write the pixels to output, but only those within outDataWindow in outFrameBuffer
	// will be magically used
	Imf::OutputFile out(outFileName, outHeader);
	out.setFrameBuffer(outFrameBuffer);
	out.writePixels(outHeight);

	// pixel data cleanup:
	for (std::vector<ChannelRecord>::const_iterator iter = channelRecordList.begin();
	     iter != channelRecordList.end();
	     ++iter) {
		if (iter->pixels)
			delete [] iter->pixels;
	}
}

int main(int argc, char *argv[])
{
	const char *exe = strrchr(argv[0], *SLASH);

	if (exe)
		++exe;
	else
		exe = argv[0];

	if (argc < 2) {
		printUsage(exe);
		return 0;
	}

	StringVector inFileNames;
	RenameList renameList;

	// Scan command line args:
	for (int i = 1; i < argc; ++i){
		// parse flags:
		if (*argv[i] == '-') {
			if (!strcmp(argv[i], "-h")) {
				printUsage(exe);
				return 0;
			}

			if (!strcmp(argv[i], "-r") || !strcmp(argv[i], "-rename")) {
				if (i >= argc - 2) {
					fprintf(stderr, "%s: not enough arguments in command line after %s\n", exe, argv[i]);
					return 1;
				}
				boost::regex re(argv[++i]);
				renameList.push_back(RenamePair(re, argv[++i]));
			} else {
				fprintf(stderr, "%s: unknown flag %s\n", exe, argv[i]);
				return 2;
			}
		} else {
			if (!access(argv[i], W_OK))
				inFileNames.push_back(argv[i]);
			else
				fprintf(stderr, "%s: %s does not exist or is not writable. Skipped\n", exe, argv[i]);
		}
	}

	if (inFileNames.empty()) {
		fprintf(stderr, "%s: no image files to process\n", exe);
		return 3;
	}

	for (int i = 0; i < inFileNames.size(); ++i) {
		const char* inFileName = inFileNames[i].c_str();

		std::string tempFileNameStr = inFileNames[i] + ".TMP";
		const char* tempFileName = tempFileNameStr.c_str();

		copyData(inFileName, tempFileName, renameList);

		// rename tempFileName:
		unlink(inFileName);

		if (rename(tempFileName, inFileName) == -1 || access(inFileName, F_OK))
			fprintf(stderr, "%s: renaming failed (%s -> %s)\n", exe, tempFileName, inFileName);
		else
			fprintf(stderr, "%s: %s processed.\n", exe, inFileName);
	}

	return 0;
}
