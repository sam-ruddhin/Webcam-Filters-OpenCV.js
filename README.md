__OpenCV.js Webcam Filters__

A lightweight, browser-based playground for real-time webcam effects with OpenCV.js. It captures your webcam stream, applies selectable image filters (grayscale, noise, colorize, cartoon, posterize), and includes a DNN-based face blur using a Caffe model ,all on the client side

✨ Features

Live webcam capture via getUserMedia
Real-time processing with OpenCV.js (WASM)
Adjustable filter intensity (0–100)

Filters included:

Black & White (grayscale with intensity scaling)

Noisy (random noise blend)

Colorized (false color via colormap + blend)

Cartoon (edge mask + color smoothing)

Posterize (LUT-based color quantization)

Face Blur (DNN, Caffe SSD face detector)

On-screen overlay showing the active filter

Runs fully in the browser 
