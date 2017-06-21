iceview is a Python module for mosaicing sea ice images built on top of skimage
and distributed under the included LICENSE.txt.


Official source code repo: https://github.com/johannah/iceview

# INSTALLATION
iceview has been tested under Python 2.7 and 3.4. 

    python setup.py install

# DEVELOPMENT
You can check out the latest code with this command:

    git clone https://github.com/johannah/iceview

# TODO
- Documentation
 * add params/desc for each function

- Testing
 * identify where
 * tests for patchmaker
 * measuring accuracy, propagation of error
 * make image graph output (see piz.)

- Knowledge Gaps
 * Which type of transform where (affine/projective?)
 * Measuring error

- Speed
 * Isolate neighbors based on gps/imu data
 * what size image should be used?
 * give user options for method of matching (incremental links vs stabilize)
 * optimize addition of features

- Features
 * add zernike descriptors in style of skimage
 * add water filter - color or texture based
 * add conv net for feature extraction

- Architecture
 * add configuration file??? -> num of iterations
 * exit cleanly with ctrl+c (how to deal with unmatched)?, save as you go
 * keep track of parameters used for features in .json file... how, what iteration
 

