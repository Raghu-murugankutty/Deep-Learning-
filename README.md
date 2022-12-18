# Deep-Learning
This repo contains deep learning projects

<hr>

 - ## Training Generative Adversarial Networks (GANs) | PyTorch | Generative Modeling ([<img src="https://img.icons8.com/fluency/48/000000/code.png"/>](https://jovian.ai/raghu-rayirath/walmart-store-sales-forecasting-v4)):
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
