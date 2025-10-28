let video = document.getElementById('video');
let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
let net = null;

console.log("Script loaded");

// Start webcam
navigator.mediaDevices.getUserMedia({ video: true })
  .then(stream => {
    video.srcObject = stream;
    console.log("🎥 Webcam stream started");
  })
  .catch(err => console.error(" Webcam error:", err));


// Wait for OpenCV.js to load
cv['onRuntimeInitialized'] = async () => {
  console.log("OpenCV.js ready");
  await loadDNN();
  startProcessing();
};


async function loadDNN() {
  try {
    console.log("Loading DNN files");

    // Load deploy.prototxt as text
    const protoResponse = await fetch('deploy.prototxt');
    const proto = await protoResponse.text();

    // Load .caffemodel as binary
    const modelResponse = await fetch('face_detector.caffemodel');
    const modelData = new Uint8Array(await modelResponse.arrayBuffer());

    // Write both to OpenCV’s in-memory filesystem
    cv.FS_createDataFile('/', 'deploy.prototxt', proto, true, false, false);
    cv.FS_createDataFile('/', 'face_detector.caffemodel', modelData, true, false, false);

    // Read the model from the virtual FS
    net = cv.readNetFromCaffe('deploy.prototxt', 'face_detector.caffemodel');

    console.log(" DNN model loaded successfully!");
  } catch (err) {
    console.error(" Error loading DNN model:", err);
  }
}



// Main loop
function startProcessing() {
//   const FPS = 30;

    const FPS = 15;

    function processFrame() {
        if (video.readyState !== video.HAVE_ENOUGH_DATA) {
        requestAnimationFrame(processFrame);
        const intensity = parseInt(document.getElementById('intensity').value);
        return;
    }

    // Draw video frame to canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    let src = cv.imread(canvas);
    if (src.empty()) {
      console.warn("Empty frame, skipping...");
      requestAnimationFrame(processFrame);
      return;
    }
    let dst = new cv.Mat();

    const filter = document.getElementById('filter').value;

    try {
      switch (filter) {
        
        case 'gray':
          cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
          cv.cvtColor(dst, dst, cv.COLOR_GRAY2RGBA);
          console.log("Applied: Black & White");
          break;

        case 'noisy':
          addNoise(src, dst);
          console.log("Applied: Noisy filter");
          break;

        case 'colorize':
          colorize(src, dst);
          console.log("Applied: Colorized");
          break;

        case 'faceblur_dnn':
          applyDnnFaceBlur(src, dst);
          console.log("Applied: Face Blur (DNN)");
          break;

        case 'cartoon':
          cartoon(src, dst);
          console.log(" Applied: Cartoon");
          break;

        case 'posterize':
          posterize(src, dst);
          console.log("Applied: Posterize");
          break;

        default:
          src.copyTo(dst);
      }

      // Overlay filter name on screen
      cv.putText(
        dst,
        `Filter: ${filter}`,
        new cv.Point(10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        new cv.Scalar(255, 255, 255, 255),
        2
      );

      cv.imshow('canvas', dst);
    } catch (e) {
      console.error("Filter error:", e);
    }

    src.delete();
    dst.delete();
    setTimeout(processFrame, 1000 / FPS);
  }

  processFrame();
}

// Filter Functions 

// NOISE FILTER
function addNoise(src, dst) {
  let noise = new cv.Mat(src.rows, src.cols, src.type());
  let randomArray = new Uint8ClampedArray(src.rows * src.cols * src.channels());

  const intensity = parseInt(document.getElementById('intensity').value); 
  const noiseAmount = Math.min(255, intensity * 2.5);
  
  // Fill with random noise
    for (let i = 0; i < randomArray.length; i++) {
    randomArray[i] = Math.floor(Math.random() * noiseAmount);
  }

  noise.data.set(randomArray);
  cv.addWeighted(src, 1.0, noise, 0.5, 0, dst);
  //cv.add(src, noise, dst);
  noise.delete();
}

// COLORIZE FILTER
function colorize(src, dst) {
  try {
    // Convert to grayscale 8-bit 1 channel
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // Create a 3-channel color output
    let color = new cv.Mat();
    cv.applyColorMap(gray, color, cv.COLORMAP_JET);

    // Copy the result to dst so rest of pipeline works
    color.copyTo(dst);

    gray.delete();
    color.delete();
  } 
  catch (err) {
    console.error("Colorize error:", err);
    src.copyTo(dst);
  }
}

// DNN FACE BLUR 
function applyDnnFaceBlur(src, dst) {
  try {
    if (!net) return src.copyTo(dst);

    // Convert RGBA to BGR 
    let bgr = new cv.Mat();
    cv.cvtColor(src, bgr, cv.COLOR_RGBA2BGR);

    // Prepare blob
    let blob = cv.blobFromImage(bgr, 1.0, new cv.Size(300, 300),
      new cv.Scalar(104.0, 177.0, 123.0), false, false);

    net.setInput(blob);

    let out = net.forward();

    // Process detections
    let faces = [];
    for (let i = 0; i < out.total(); i += 7) {
      let confidence = out.data32F[i + 2];
      if (confidence > 0.5) {
        let x1 = out.data32F[i + 3] * src.cols;
        let y1 = out.data32F[i + 4] * src.rows;
        let x2 = out.data32F[i + 5] * src.cols;
        let y2 = out.data32F[i + 6] * src.rows;

        let rect = new cv.Rect(x1, y1, x2 - x1, y2 - y1);
        let roi = src.roi(rect);
        cv.GaussianBlur(roi, roi, new cv.Size(55, 55), 30, 30);
        roi.copyTo(src.roi(rect));
        roi.delete();
      }
    }

    src.copyTo(dst);

    bgr.delete();
    blob.delete();
    out.delete();

  }
  catch (err) {
    console.error("Face blur error (caught safely):", err.toString());
    src.copyTo(dst);
  }
}

// CARTOON FILTER
function cartoon(src, dst) {
  try {
    
    // Step 1: Convert to gray
    let gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    // Step 2: Median blur
    cv.medianBlur(gray, gray, 7);

    // Step 3: Edge detection (lighter)
    let edges = new cv.Mat();
    cv.adaptiveThreshold(gray, edges, 255,
      cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9);

    // Step 4: Convert edges to 3-channel
    cv.cvtColor(edges, edges, cv.COLOR_GRAY2RGBA);

    // Step 5: Light color smoothing instead of bilateralFilter
    let color = new cv.Mat();
    cv.GaussianBlur(src, color, new cv.Size(5, 5), 0, 0);

    cv.bitwise_and(color, edges, dst);
    gray.delete(); edges.delete(); color.delete();
  } 
  catch (err) {
    console.error("Cartoon filter failed:", err);
    src.copyTo(dst);
  }
}

// POSTERIZE FILTER
function posterize(src, dst) {
  try {
    // Convert to float for math
    let tmp = new cv.Mat();
    src.convertTo(tmp, cv.CV_32F);

    let levels = 4; 
    let step = 255.0 / (levels - 1);

    //  lookup table (LUT)
    let lut = new cv.Mat(1, 256, cv.CV_8U);
    for (let i = 0; i < 256; i++) {
      let quantized = Math.round(i / step) * step;
      lut.ucharPtr(0, i)[0] = Math.min(255, quantized);
    }

    // Apply LUT to each channel
    cv.LUT(src, lut, dst);

    tmp.delete();
    lut.delete();
  } 
  catch (err) {
    console.error("Posterize error:", err);
    src.copyTo(dst);
  }
}

