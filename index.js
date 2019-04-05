const loadCSV = require('./load-csv');
const tf = require('@tensorflow/tfjs');

// shuffle -  randonize the data
// splitTest - get number of data
// dataColumns - get data base on the columns
let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long'],
    labelColumns: ['price']
});

//console.log(testFeatures);
//console.log(testLabels);

const feature = tf.tensor([
    [-121, 47],
    [-121.2, 46.5],
    [-122, 46.4],
    [-120.9, 46.7]
]);

const label = tf.tensor([
    [200],
    [250],
    [215],
    [240]
]);

const predictionPoint = tf.tensor([-121, 47]);

console.log(predictionPoint);