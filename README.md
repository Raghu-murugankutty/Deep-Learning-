# Deep-Learning
This repo contains deep learning projects
![](https://d2r55xnwy6nx47.cloudfront.net/uploads/2022/02/SCALING_NETS_2880x1620_Lede.svg)
<hr>


 - ## Training Generative Adversarial Networks (GANs) | PyTorch | Generative Modeling ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.org/github/Raghu-murugankutty/Deep-Learning-/blob/main/Training%20Generative%20Adversarial%20Networks%20%28GANs%29%20in%20PyTorch.ipynb)):
    - `Problem Statement`: The dataset contains over 63,000 cropped anime faces, we are generating fake images from the existing real images using generative adversarial networks (GANs).
    - `Dataset`: Anime Face Dataset, which consists of over 63,000 cropped anime faces.
    - `Discriminator Network:`
    The discriminator takes an image as input, and tries to classify it as "real" or "generated". 
In this sense, it's like any other neural network. We'll use a convolutional neural networks (CNN) which outputs a single number output for every image. 
used stride of 2 to progressively reduce the size of the output feature map. 
    - `Activation function:` Used Leaky ReLU activation for the discriminator.
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290483-2e2bb625-afc7-4ec0-af85-e41882c7997a.png">

    - `Generator Network:`
The input to the generator is typically a vector or a matrix of random numbers (referred to as a latent tensor) which is used as a seed for generating an image. 
The generator will convert a latent tensor of shape (128, 1, 1) into an image tensor of shape 3 x 28 x 28. 
To achive this,I used the ConvTranspose2d layer from PyTorch, which is performs to as a transposed convolution (deconvolution)
    - `Activation Function:` The ReLU activation is used in the generator with the exception of the output layer which uses the Tanh function.
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290517-8b8d4bc0-5e18-4be8-b692-b36371cfe90d.png">
    - I have to build a GAN with CNN architecture using PyTorch to achieve `accuracy of ~93.5%`.
    - `Real Images:`
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290297-45c481cf-1170-4152-8e55-c9e8f496abe7.png">
    - `Fake generated images using (GAN):`
    - <img width="464" alt="image" src="https://user-images.githubusercontent.com/41443395/208290414-89ad07c8-2937-482c-a7b6-de8b92d4bd1a.png">

<hr>

 - ## Image classification using ResNets | Regularization | Data Augmentation in PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://nbviewer.org/github/Raghu-murugankutty/Deep-Learning-/blob/main/Image%20classification%20using%20ResNets%2C%20Regularization%20and%20Data%20Augmentation%20in%20PyTorch.ipynb)):
    - `Problem Statement`: The dataset contains over 60,000 images belonging to 10 classes,Image classification using ResNets.
    - `Dataset`: The dataset contains over 60,000 images belonging to 10 classes. 
    - `Residual Block:`
    - <img width="510" alt="image" src="https://user-images.githubusercontent.com/41443395/208294435-d610a010-18ee-46ba-ab05-71d674b7f625.png">
    - `Convolution Block with ResNet9:`
    - <img width="510" alt="image" src="https://user-images.githubusercontent.com/41443395/208294492-bc60c300-2986-4262-bba5-b4796fa13712.png">
    - `One Cycle Learning Rate Policy:`
    - <img width="510" alt="image" src="https://user-images.githubusercontent.com/41443395/208294535-b3faebb9-45d1-4ebe-8528-2433ba5019a3.png">
    - `Built Feed Forward neural network(ANN)` and achievied an accurcy of 48%.
    - `Built Convolutional Neural Network(CNN)` and improved the accuracy till 75%.
    - <img width="230" alt="image" src="https://user-images.githubusercontent.com/41443395/208294842-4c51b6c7-449d-4b1c-b207-0f9e16bc370a.png">
    - <img width="268" alt="image" src="https://user-images.githubusercontent.com/41443395/208294918-d307e51a-d6e9-48c2-b3dd-6251c9e0a97f.png">
    - Applied technique like `Data normalization`, `Data augmentation`, `Batch normalization`, `Learning rate scheduling`, `Weight Decay`, `Gradient clipping`...etc
    - Using ResNet architecture, I achieved the `accuracy of 90.45%`.
<hr>

 - ## Transfer Learning for Image Classification | ResNets |  PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/Raghu-murugankutty/Deep-Learning-/blob/main/Transfer%20Learning%20for%20Image%20Classification%20PyTorch.ipynb)):
    - `Problem Statement`: The dataset contains 37 category (breeds) pet dataset with roughly 200 images for each class, 
Performing image classification using tranfser learning models.
    - `Dataset`: We'll use the Oxford-IIIT Pets dataset from "https://course.fast.ai/datasets". It is 37 category (breeds) pet dataset with roughly 200 images for each class. The images have a large variations in scale, pose and lighting.
    - `Sample Images from Oxford-IIIT Pets dataset:`
    - <img width="438" alt="image" src="https://user-images.githubusercontent.com/41443395/208370719-f082c2e4-f319-4884-a42f-2312b8798e3f.png">
    - `Using Pre-trained weights:`
    - <img width="530" alt="image" src="https://user-images.githubusercontent.com/41443395/208370341-e1731788-74d3-4f8d-a8af-084fc8f3a0f5.png">
    - `Parameter:`
    - <img width="405" alt="image" src="https://user-images.githubusercontent.com/41443395/208370924-f70659a6-2990-440c-a3f7-42c42f13e84b.png">
    - Using pre-trained transfer learning model weights, I achieved the `accuracy of 80.01%`.(with minimal epochs)
    
 <hr>

 - ## Image Classification using Convolutional Neural Networks (CNN) in PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/Raghu-murugankutty/Deep-Learning-/blob/main/Image_Classification_using_Convolutional_Neural_Networks_in_PyTorch%20v2.ipynb)):
    - `Problem Statement`: The dataset contains over 60,000 images belonging to 10 classes,Image Classification using Convolutional Neural Networks using PyTorch.
    - `Dataset`:  Dataset contains 2 folders train and test,The training set contains (50000 images) and test set (10000 images) respectively. The images belonging to 10 classes. 
    - `Sample Image Grid:`
    - <img width="491" alt="image" src="https://user-images.githubusercontent.com/41443395/208302244-cedfac65-f7c7-4d52-8f95-4f2d59ba10b5.png">
    - `Convolution Example:`
    - <img src="https://miro.medium.com/max/1070/1*Zx-ZMLKab7VOCQTxdZ1OAw.gif" style="max-width:400px;">
    - `CNN Block:`
    - <img width="523" alt="image" src="https://user-images.githubusercontent.com/41443395/208302290-0172c6c0-0de9-433b-a856-541e04e45118.png">
    - `Result:`Using Convolutional Neural Networks(CNN), I achieved the accuracy of 78.19% 
    - `Sample prediction Results:`
    - <img width="215" alt="image" src="https://user-images.githubusercontent.com/41443395/208302380-3263f178-59d1-4168-b5cb-5ef4d5874fbe.png">
    - <img width="232" alt="image" src="https://user-images.githubusercontent.com/41443395/208302395-c5ccfdb3-72c1-43b7-8e1f-9a85f03eb879.png">

<hr>


 - ## Regularization of CNN models | Image classification model | PyTorch ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://github.com/Raghu-murugankutty/Deep-Learning-/blob/main/Image_Classification_using_Convolutional_Neural_Networks_in_PyTorch%20v2.ipynb)):
    - `Problem Statement`: The dataset contains over 90483 images of fruits or vegitables, we are using CNN to build a classifcation model to predict the class of fruit from a set of 131 classes using PyTorch.
    - `Dataset`:  The daaset contains 90483 images of fruits or vegitables. 
    `Training set size`: 67692 images (one fruit or vegetable per image).
    `Test set size`: 22688 images (one fruit or vegetable per image).
    `Number of classes`: 131 (fruits and vegetables). Image size: 100x100 pixels.
    - Model is built on PyTorch along with the implementation of techniques like `Data augmentation`, `Batch normalization`, `learning rate schedule`, `Weight Decay`, `Gradient clipping`, `adam optimizer`, `layer dropouts`, `Minmax pooling` to achieve the best results.
    - The model is trained and evaluated on GPU using PyTorch built-in `CUDA library`.
    - <img width="580" alt="image" src="https://user-images.githubusercontent.com/41443395/209466663-b5d2c208-edb4-41a8-862b-5b9dcf1a8828.png">
    - <img width="580" alt="image" src="https://user-images.githubusercontent.com/41443395/209466675-855b715e-1e41-43de-8419-175f30353f06.png">




