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

// using the distance formula
feature
    // find the difference bewteen each row and the predictionPoint
    .sub(predictionPoint)
    // raise it to the power of two
    .pow(2)
    // find the sum for each row
    .sum(1)
    // square root each row
    .pow(.5)

console.log(feature);