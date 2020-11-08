//import * as tf from '@tensorflow/tfjs';
let model;
const IMAGE_SIZE = 784;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;
const TRAIN_TEST_RATIO = 5 / 6;
const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;
const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';
export class MnistData {
    constructor() {
        this.shuffledTrainIndex = 0;
        this.shuffledTestIndex = 0;
    }
    async load() {
        const img = new Image();
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const imgRequest = new Promise((resolve) => {
            img.crossOrigin = '';
            img.onload = () => {
                img.width = img.naturalWidth;
                img.height = img.naturalHeight;
                const datasetBytesBuffer =
                    new ArrayBuffer(NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4);
                const chunkSize = 5000;
                canvas.width = img.width;
                canvas.height = chunkSize;
                for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
                    const datasetBytesView = new Float32Array(
                        datasetBytesBuffer, i * IMAGE_SIZE * chunkSize * 4,
                        IMAGE_SIZE * chunkSize);
                    ctx.drawImage(
                        img, 0, i * chunkSize, img.width, chunkSize, 0, 0, img.width,
                        chunkSize);
                    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                    for (let j = 0; j < imageData.data.length / 4; j++) {
                        datasetBytesView[j] = imageData.data[j * 4] / 255;
                    }
                }
                this.datasetImages = new Float32Array(datasetBytesBuffer);
                resolve();
            };
            img.src = MNIST_IMAGES_SPRITE_PATH;
        });
        const labelsRequest = fetch(MNIST_LABELS_PATH);
        const [imgResponse, labelsResponse] =
            await Promise.all([imgRequest, labelsRequest]);
        this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer());
        this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
        this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);
        this.trainImages =
            this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
        this.trainLabels =
            this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
        this.testLabels =
            this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    }
    nextTrainBatch(batchSize) {
        return this.nextBatch(
            batchSize, [this.trainImages, this.trainLabels], () => {
                this.shuffledTrainIndex =
                    (this.shuffledTrainIndex + 1) % this.trainIndices.length;
                return this.trainIndices[this.shuffledTrainIndex];
            });
    }
    nextTestBatch(batchSize) {
        return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
            this.shuffledTestIndex =
                (this.shuffledTestIndex + 1) % this.testIndices.length;
            return this.testIndices[this.shuffledTestIndex];
        });
    }
    nextBatch(batchSize, data, index) {
        const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
        const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

        for (let i = 0; i < batchSize; i++) {
            const idx = index();

            const image =
                data[0].slice(idx * IMAGE_SIZE, idx * IMAGE_SIZE + IMAGE_SIZE);
            batchImagesArray.set(image, i * IMAGE_SIZE);

            const label =
                data[1].slice(idx * NUM_CLASSES, idx * NUM_CLASSES + NUM_CLASSES);
            batchLabelsArray.set(label, i * NUM_CLASSES);
        }

        const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
        const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

        return { xs, labels };
    }
}

async function showExamples(data) {
    const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});  
    const examples = data.nextTestBatch(20);
    const numExamples = examples.xs.shape[0];
    for (let i = 0; i < numExamples; i++) {
      const imageTensor = tf.tidy(() => {
        return examples.xs
          .slice([i, 0], [1, examples.xs.shape[1]])
          .reshape([28, 28, 1]);
      });
      const canvas = document.createElement('canvas');
      canvas.width = 28;
      canvas.height = 28;
      canvas.style = 'margin: 4px;';
      await tf.browser.toPixels(imageTensor, canvas);
      surface.drawArea.appendChild(canvas);
      imageTensor.dispose();
    }
  }

async function run() {  
    const data = new MnistData();
    await data.load();
    await showExamples(data);
    model = getModel();
    tfvis.show.modelSummary({name: 'Model Architecture'}, model);
    await train(model, data);
    await showAccuracy(model,data);
    await showConfusion(model,data);
  }
  document.addEventListener('DOMContentLoaded', run);
  function getModel() {
    const model = tf.sequential();
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;
    model.add(tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], kernelSize: 5, filters: 8,
      strides: 1, activation: 'relu', kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.conv2d({kernelSize: 5, filters: 16,
      strides: 1, activation: 'relu', kernelInitializer: 'varianceScaling'
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten());
    const NUM_OUTPUT_CLASSES = 10;
    model.add(tf.layers.dense({units: NUM_OUTPUT_CLASSES, kernelInitializer: 'varianceScaling', activation: 'softmax'}));
    const optimizer = tf.train.adam();
    model.compile({
      optimizer: optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
    return model;
  }
  async function train(model, data) {
    const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
    const container = { name: 'Model Training', styles: { height: '1000px' } };
    const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 55000;
    const TEST_DATA_SIZE = 10000;
    const [trainXs, trainYs] = tf.tidy(() => {
      const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
      return [d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]), d.labels];
    });
    const [testXs, testYs] = tf.tidy(() => {
      const d = data.nextTestBatch(TEST_DATA_SIZE);
      return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
    });
    return model.fit(trainXs, trainYs, {
      batchSize: BATCH_SIZE,
      validationData: [testXs, testYs],
      epochs: 10,
      shuffle: true,
      callbacks: fitCallbacks
    });
  }

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

function doPrediction(model, data, testDataSize = 500) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const testData = data.nextTestBatch(testDataSize);
  const testxs = testData.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
}


async function showAccuracy(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  const container = {name: 'Accuracy', tab: 'Evaluation'};
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);

  labels.dispose();
}

async function showConfusion(model, data) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  tfvis.render.confusionMatrix(container, {values: confusionMatrix, tickLabels: classNames});

  labels.dispose();
}

var canvasWidth = 280;
var canvasHeight = 280;
var canvasStrokeStyle = "white";
var canvasLineJoin = "round";
var canvasLineWidth = 10;
var canvasBackgroundColor = "black";
var canvasId = "canvas";

var clickX = new Array();
var clickY = new Array();
var clickD = new Array();
var drawing;

var canvasBox = document.getElementById('canvas_box');
var canvas = document.createElement("canvas");

canvas.setAttribute("width", canvasWidth);
canvas.setAttribute("height", canvasHeight);
canvas.setAttribute("id", canvasId);
canvas.style.backgroundColor = canvasBackgroundColor;
canvasBox.appendChild(canvas);
if (typeof G_vmlCanvasManager != 'undefined') {
    canvas = G_vmlCanvasManager.initElement(canvas);
}

var context = canvas.getContext("2d");
$("#canvas").mousedown(function (e) {
	var rect = canvas.getBoundingClientRect();
	var mouseX = e.clientX - rect.left;;
	var mouseY = e.clientY - rect.top;
	drawing = true;
	addUserGesture(mouseX, mouseY);
	drawOnCanvas();
});
$("#canvas").mousemove(function (e) {
	if (drawing) {
		var rect = canvas.getBoundingClientRect();
		var mouseX = e.clientX - rect.left;;
		var mouseY = e.clientY - rect.top;
		addUserGesture(mouseX, mouseY, true);
		drawOnCanvas();
	}
});
$("#canvas").mouseup(function (e) {
	drawing = false;
});
$("#canvas").mouseleave(function (e) {
	drawing = false;
});
function addUserGesture(x, y, dragging) {
	clickX.push(x);
	clickY.push(y);
	clickD.push(dragging);
}
function drawOnCanvas() {
	context.clearRect(0, 0, context.canvas.width, context.canvas.height);

	context.strokeStyle = canvasStrokeStyle;
	context.lineJoin = canvasLineJoin;
	context.lineWidth = canvasLineWidth;

	for (var i = 0; i < clickX.length; i++) {
		context.beginPath();
		if (clickD[i] && i) {
			context.moveTo(clickX[i - 1], clickY[i - 1]);
		} else {
			context.moveTo(clickX[i] - 1, clickY[i]);
		}
		context.lineTo(clickX[i], clickY[i]);
		context.closePath();
		context.stroke();
	}
}
$("#clear-button").click(async function () {
	context.clearRect(0, 0, canvasWidth, canvasHeight);
	clickX = new Array();
	clickY = new Array();
    clickD = new Array();
    $("#prediect")[0].textContent = "";
});
function preprocessCanvas(image) {
	// resize the input image to target size of (1, 28, 28)
	let tensor = tf.browser.fromPixels(image)
		.resizeNearestNeighbor([28, 28])
		.mean(2)
		.expandDims(2)
		.expandDims()
		.toFloat();
	console.log(tensor.shape);
	return tensor.div(255.0);
}
$("#predict-button").click(async function () {
	var imageData = canvas.toDataURL();
	let tensor = preprocessCanvas(canvas);
	let predictions = await model.predict(tensor).data();
    let results = Array.from(predictions);
    console.log(results);
    let result = 0, index = 0;
    for (var i = 0; i < results.length; i++) {
        if (result < results[i]) {
            result = results[i];
            index = i;
        }
    }
    console.log(index);
    $("#prediect")[0].textContent = "The maximum predicted value is " + index;
});
