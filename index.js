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

// features is the dataset of the dataColumns
// testFeatures is the splitTest
// labels is the labelColumns
// testLabels is the labelColumns for each splitTest

console.log(testFeatures);
console.log(testLabels);

function knn(features, labels, predictionPoint, k){
    // using the distance formula
    return features
        // find the difference bewteen each row and the predictionPoint
        .sub(predictionPoint)
        // raise it to the power of two
        .pow(2)
        // find the sum for each row
        .sum(1)
        // square root each row
        .pow(.5)
        // make this tensor 2D
        .expandDims(1)
        // combine tensor
        .concat(labels, 1)
        // break up this tensor into group of tensors
        .unstack()
        // this is js method
        .sort((a, b) => a > b ? 1 : -1)
        // this is js method
        .slice(0, k)
        // add up the total and then divide the total by k
        .reduce((acc, pair) => acc + pair, 0) / k;
}