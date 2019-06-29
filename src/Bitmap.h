#ifndef BITMAP_GUARDIAN_H
#define BITMAP_GUARDIAN_H

void SaveBMP(const char* fname, const unsigned int* pixels, int w, int h);

/**
\brief load 24 bit RGB bitmap images.
\param fname - file name
\param pW    - file name
\param pH    - file name
\return RGBA data, 4 bytes per pixel

  Note that this function in this sample works correctly _ONLY_ for 24 bit RGB ".bmp" images.
  If you want to support gray-scale images or images with palette, please upgrade its implementation.
*/
std::vector<unsigned int> LoadBMP(const char* fname, int* pW, int* pH);

#endif
