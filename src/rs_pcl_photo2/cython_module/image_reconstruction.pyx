from numpy cimport ndarray, complex128_t, float64_t
from numpy import zeros, sqrt, add, asarray
from numpy.fft import ifftshift, fft2, ifft2 ,fftshift
from libcpp.map cimport map
from libcpp.utility cimport pair
from cython cimport boundscheck, wraparound
ctypedef unsigned char uchar

cdef class Image_Reconstruction:
    cdef:

        int dim1, dim2, dim3
        double[:,:] rec_image
        complex128_t[:,:] individual_grating

    def __cinit__(self, ndarray dimg, int[:,:] coords, complex128_t[:,:] ft, int centre):
        if(dimg.ndim==2):
            self.dim1 = dimg.shape[0]
            self.dim2 = dimg.shape[1]
        elif (dimg.ndim==3):
            self.dim1 = dimg.shape[0]
            self.dim2 = dimg.shape[1]
            self.dim3 = dimg.shape[2]
        else:
            print(" dimensions!")
        self.rec_image = zeros((self.dim1, self.dim2))
        print("shape -> ", self.dim1, self.dim2)
        self.individual_grating = zeros((self.dim1, self.dim2), dtype="complex")

        self.mainLoop(coords, ft, centre)

    @boundscheck(False)
    @wraparound(False)
    cdef double[:,:] getRecImage(self):
        return self.rec_image

    @boundscheck(False)
    @wraparound(False)
    cdef complex128_t[:,:] getIndividualGrating(self):
        return self.individual_grating

    @boundscheck(False)
    @wraparound(False)
    cdef inline pair[int, int] find_symmetric_coordinates(self, int[:] coords, int centre):
        cdef:
            pair[int, int] coordinates
        coordinates.first = centre + (centre - coords[0])
        coordinates.second =  centre + (centre - coords[1])
        return coordinates

    @boundscheck(False)
    @wraparound(False)
    cdef inline complex128_t[:,:] calculate_2dft(self, input):
        cdef complex128_t[:,:] ft = fft2(ifftshift(input))
        return fftshift(ft)

    @boundscheck(False)
    @wraparound(False)
    cdef inline float64_t[:,:] calculate_2dift(self, input):
        cdef complex128_t[:,:] ift = ifft2(ifftshift(input))
        return fftshift(ift).real

    @boundscheck(False)
    @wraparound(False)
    cdef void mainLoop(self, int[:,:] coords, complex128_t[:,:] ft, int centre):
        cdef:
            size_t length, idx
            pair[int, int] symm_coords
            float64_t[:,:] rec_grating
        length = len(coords)
        idx = 0
        # Step 2
        for i in range(length): # coords_left_half:
            # Central column: only include if points in top half of the central column
            if not (coords[i][1] == centre and coords[i][0] > centre):
                idx += 1
                symm_coords = self.find_symmetric_coordinates(coords[i], centre)

                # Step 3
                # Copy values from Fourier transform into individual_grating for the pair of points in
                # current iteration
                self.individual_grating[(coords[i][0], coords[i][1])] = ft[(coords[i][0], coords[i][1])]
                self.individual_grating[(symm_coords.first, symm_coords.second)] = ft[(symm_coords.first, symm_coords.second)]

                # Step 4
                # Calculate inverse Fourier transform to give the reconstructed grating. Add this reconstructed
                # grating to the reconstructed image
                rec_grating = self.calculate_2dift(self.individual_grating)
                self.rec_image = add(self.rec_image,rec_grating)

                # Clear individual_grating array, ready for next iteration
                self.individual_grating[(coords[i][0], coords[i][1])] = 0
                self.individual_grating[(symm_coords.first, symm_coords.second)] = 0
                # print(" ind -> ",self.individual_grating[(symm_coords.first, symm_coords.second)], "index -> ", idx)

    def recImage(self):
        return  asarray(self.getRecImage())

    def individualGrating(self):
        return asarray(self.getIndividualGrating())