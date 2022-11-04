# Fast Fourier Transformation for Image Reconstruction 


## Dependencies 

the following libraries should be installed before running the source code:

    imageio==2.21.2
    matplotlib==3.5.3
    numpy==1.23.2
    opencv-python==4.6.0.66
    Pillow==9.2.0
    scikit-image==0.19.3
    scipy==1.9.1

The easy way is directly to install these libraries using pip:

      python3.? -m pip install imageio, matplotlib, numpy, opencv-python, Pillow, scikit-image, scipy

or they could also be installed from the **requirements.txt** file. Navigate into the directory _rs-pcl-photo_. 
There run the following line of code:
    
    python3.? -m pip install -r ./requirements.txt

it is also possible to install the rs-pcl-photo modul as a library using the .toml file, navigating into the rs-pcl-photo:

      python3.? -m pip install -e .

using the last way will enable to test or run the rs-pcl-photo modul using the already written main file in the ./src/rs_pcl_photo2/ directory:


## Run

After the dependencies have been installed, you can navigate to the directory ./src/rs_pcl_photo2/ and there run the file main:

    python3.? main.py python3.8 main.py --fname=depth_images/rs-photo-00-depth.png --pfilter=high


In order to use the different parser arguments, set them as follow:

    --fpath ->  path to a filter
    --out ->  outputs a png file for the: -filter -mask -original -reco  
    --pfilter -> low: low pass filter; high: high pass filter
    --method -> 1: fft2 -> fftshift 2: iffshift -> fft2 -> ffshift
    --freq -> 




