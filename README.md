# UnAxSeg
My own version of [AxonDeepSeg](https://axondeepseg.readthedocs.io).

In this repository, I have reimplemented AxonDeepSeg from Zaimi, A., Wabartha, M., Herman, V., Antonsanti, P.-L., Perone, C. S., & Cohen-Adad, J. (2018). 
AxonDeepSeg: automatic axon and myelin segmentation from microscopy data using convolutional neural networks. Scientific Reports, 8(1), 3816. 
Link to paper: https://doi.org/10.1038/s41598-018-22181-4.

The implementation is in keras and the model is training for Light microscopy data (the input data are RGb although we convert
the images in grayscale, very early in the pipeline).

To start, use the jupyter notebooks.

For the install use the anaconda and the .yml file... Use the README.txt file for complete instruction for the install
