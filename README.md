Autologistic model
===================
## About
This is an implementation of Autologistic model in Python for the experiments as shown in the Section 5 of the paper "Contrasting Vertical and Horizontal Transmission of Typological Features" in COLING 2016.

This software is released under the MIT License, see LICENSE.

## Requirements
* Python 3.5
* NetworkX
* Numpy
* [WALS data (CSV)](http://wals.info/download) should be downloaded to `data` directory

## Usage overview
### Print WALS features to process
* You can see the indices of WALS features to process by:
```
$ python ./scripts/experiment.py print-features
Index	Feature name
----------------------------------------------
0	101A Expression of Pronominal Subjects
1	112A Negative Morphemes
2	116A Polar Questions
...
```

### Run experiments
* To run the experiments, use `scripts/experiment.py` with following arguments.
```
$ python ./scripts/experiment.py exp [first index of WALS feature to process] [last index of WALS feature to process] [experiment type]
```
* You can specify the path of input files and the directory to output. See more information by `help`:
```
$ python ./scripts/experiment.py exp --help
```

## Missing value imputation
### Run
* To run the missing value imputation experiment like:
```
$ python ./scripts/experiment.py exp 0 10 mvi
```
* It outputs the results to the directory `(output directory)/mvi/(number of the split)/(feature name).json`.

### Output
Output JSON file is like:
```json
{
    "feature": "101A Expression of Pronominal Subjects",
    "lambda": 0.002049842660738542,
    "theta": 0.009065555215474861,
    "beta": [
        -0.09558277923882781,
        0.9942113364129587,
        -0.5985722330582819,
        -0.15532445575421136,
        -0.24775781040167438,
        -0.5600623232969136
    ],
    "accuracy": {
        "baseline": 0.4788732394366197,
        "proposed": 0.5915492957746479
    },
    "estimate_results": [ ... ]
}
```
`estimate_results` field contains each estimated or original feature value of each language like:
```json
{
   "value": "2 Subject affixes on verb",
   "is_hidden_estimated": true,
   "language": "Aja",
   "original": "2 Subject affixes on verb"
},
```
if `is_hidden_estimated` is `true`, it denotes the original feature value (`original`) of the language ("Aja" in example) is hidden to the missing value imputation test and estimated as `value` by the model.

if `is_original` is `true`, it denotes the feature value (`value`) of the language is the original one, otherwise, is estimated by the model.

## Parameter estimation
### Run
* To run the parameter estimation experiment like:
```
$ python ./scripts/experiment.py exp 0 10 param
```
* It outputs the results to the directory `(output directory)/param/(feature name).json`.

### Output
Output JSON file is like:
```json
{
    "feature": "116A Polar Questions",
    "lambda": 0.010144166778914547,
    "theta": 0.013618914159751434,
    "beta": [
        0.5496708416159658,
        0.2262238700953768,
        -0.07882853251432281,
        -0.07955149753492666,
        -0.09035882308183195,
        0.2525423664878199,
        -0.09236543248194536
    ],
    "estimate_results": [ ... ]
}
```
`estimate_results` field contains each estimated or original feature value of each language like:
```json
{
    "is_hidden_estimated": false,
    "language": "Ajagbe",
    "is_original": false,
    "value": "1 Question particle"
},
```
if `is_original` is `true`, it denotes the feature value (`value`) of the language ("Ajagbe" in example) is the original, otherwise, is estimated by the model.
