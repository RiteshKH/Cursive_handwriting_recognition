## [Cursive_handwriting_recognition](https://github.com/RiteshKH/Cursive_handwriting_recognition)

Python scripts for segmentation of cursive handwritten image, and recognizing the characters using a CNN based model

Handwriting recognition is one of the challenging and difficult problems. A number of papers have been published with research detailing new techniques for the classification of handwritten numerals and words. In most of the research, the text recognition is carried out by passing each small part of segmented text directly to a neural network based model. This technique relies on the accuracy of model and confidence values for verification of correctness which is very slow & difficult to implement.
The technique proposed here is a simple and unique set of algorithms for text recognition which primarily focuses on the character segmentation. The algorithm involves the preprocessing, segmentation, model building and recognition steps.

### Instructions 

All the python scripts can be independently executed. The following order needs to be maintained.

* [`image-straighten.py`](https://github.com/RiteshKH/Cursive_handwriting_recognition/blob/master/image-straighten.py) (optional) This module is used for slant-correction of the images. If the image is already straightened, no need to execute this.
![slant_correction](https://user-images.githubusercontent.com/38212000/61037250-4d87db00-a3e8-11e9-9552-71c26e789b7a.JPG)

* [`segmentation.py`](https://github.com/RiteshKH/Cursive_handwriting_recognition/blob/master/segmentation.py) : This takes in the desired image from the folder [`sample_image`](https://github.com/RiteshKH/Cursive_handwriting_recognition/tree/master/sample_images), and performs segmentation operation. This includes word and character segmentation, stored in the [`result/characters`](https://github.com/RiteshKH/Cursive_handwriting_recognition/tree/master/result/new_result) folder.

![word_segment](https://user-images.githubusercontent.com/38212000/61037258-51b3f880-a3e8-11e9-88ed-f519d9e4e4a0.JPG)
![top_contour](https://user-images.githubusercontent.com/38212000/61037267-55e01600-a3e8-11e9-9c4a-b63a7b22eef8.JPG)
![char_segment](https://user-images.githubusercontent.com/38212000/61037274-58db0680-a3e8-11e9-8c01-1e73908c295c.JPG)

* [`center-align.py`](https://github.com/RiteshKH/Cursive_handwriting_recognition/blob/master/center-align.py) : This takes in the segmented character images, adds padding, resizes and center aligns the images, ready to be fed to the model for prediction. The standardized images are stored in [`result/image`](https://github.com/RiteshKH/Cursive_handwriting_recognition/tree/master/result/image) folder.

![image_standardization](https://user-images.githubusercontent.com/38212000/61037283-5d072400-a3e8-11e9-8a8c-84625ce56812.JPG)

* [`model_build.py`](https://github.com/RiteshKH/Cursive_handwriting_recognition/blob/master/model_build.py) : This module builds the CNN model using Keras. The training and testing data is taken from [`gzip`](https://github.com/RiteshKH/Cursive_handwriting_recognition/tree/master/gzip). The model is saved in the same folder. If the model exists, it is used for future recognition. If not, a new model is built.

![model_architecture](https://user-images.githubusercontent.com/38212000/61037287-5f697e00-a3e8-11e9-9769-f61a02072960.JPG)

* [`recognition.py`](https://github.com/RiteshKH/Cursive_handwriting_recognition/blob/master/recognition.py) : This uses the model built to recognise the characters in the images. The images and their predicted output from the model is displayed together.

### Future work:

* Few characters missing in segmentation, needs fixes.
* The accuracy of the model is low as of now. Needs more training.
* Use of transfer learning using models like Resnet, or ensemble methods, may give better results.
