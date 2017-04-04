Capstone Project - Machine Learning Engineer Nanodegree - Forest Cover Type Prediction (https://www.kaggle.com/c/forest-cover-type-prediction)

Brandon Ma

Feb., 2017

Definition
Project Overview
People depend on forests to live. They filter the water we drink and the air we breathe. Worldwide, 1.6 billion people rely on forests for their livelihoods, including food, clothing, or shelter. Forests are home to nearly half of the world's species, including some of the most endangered birds and mammals, such as orangutans, gorillas, pandas, Northern Spotted Owls and Marbled Murrelets. Deforestation and forest destruction is the second leading cause of carbon pollution, causing 20% of total greenhouse gas emissions.

In the US, much of our forestland is private. If landowners can’t earn a living from these forests, they will inevitably cut them down for farms, ranches or real estate development. While total acreage of forests in the US remains relatively stable, certain parts of the country are seeing declining forest coverage. For example, the US Forest Service estimates 12 million acres of forest in the Southeast will be lost to suburban real estate development between 1992 - 2020. Many states in the US, have designed 'Forest classification' programs to to help keep the private forests intact. It allows landowners with certain acres of forest to set it aside and to remain as forest. In return for meeting program guidelines landowners receive property tax breaks, forestry literature and periodic free inspections by a professional forester while the forest is enrolled in the program.

The goal of this project is to create a model that can be used to predict and classify forests. Given the strictly cartographic variables (as opposed to remotely sensed data), this project tries to evaluate the accuracy of forest type prediction. To evaluate, the mean accuracy on the given test data and labels was used. Kaggle has launched a machine learning competition on this topic.

The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:
1 - Spruce/Fir
2 - Lodgepole Pine
3 - Ponderosa Pine
4 - Cottonwood/Willow
5 - Aspen
6 - Douglas-fir
7 - Krummholz

The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).


Problem Statement
This project is looking to solve the problem of predicting forest cover type. As mentioned, our training set has 15120 data points. Each data point falls into one of seven categories. These data points make up the training dataset used for fitting our machine learning models. The models is then used on an unlabeled test dataset. A predicted score of 1-7 will be assigned to each test data point. The aim is to accurately score the test data and submit to kaggle competition. While the competition has passed its deadline and is inactive, test data predictions can still be submitted for benchmarking on the leaderboard, where ranked the performance of the algorithms.

After exploring the dataset and understanding its structure, it became apparent that this is a classification problem. Numerous attempts were made on this Kaggle competition. I decided to base this project on the knowledge I learned from Udacity. The solution may be far from perfect, but one I can speak to comfortably.

The approach taken in this project is as follow:
- pre-process the data to remove outliers
- attempt feature selection
- fit various models
- tune the models if needed with grid search cross validation techniques
- perform predictions on the test dataset
- ensemble the models predictions to get a final score
- submit the test dataset to the Kaggle competition for benchmarking and evaluation.


Analysis
Data Exploration
The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form (not scaled) and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.

This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:

1 - Spruce/Fir
2 - Lodgepole Pine
3 - Ponderosa Pine
4 - Cottonwood/Willow
5 - Aspen
6 - Douglas-fir
7 - Krummholz

The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).

Data Fields

Elevation - Elevation in meters
Aspect - Aspect in degrees azimuth
Slope - Slope in degrees
Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

The wilderness areas are:

1 - Rawah Wilderness Area
2 - Neota Wilderness Area
3 - Comanche Peak Wilderness Area
4 - Cache la Poudre Wilderness Area

The soil types are:

1 Cathedral family - Rock outcrop complex, extremely stony.
2 Vanet - Ratake families complex, very stony.
3 Haploborolis - Rock outcrop complex, rubbly.
4 Ratake family - Rock outcrop complex, rubbly.
5 Vanet family - Rock outcrop complex complex, rubbly.
6 Vanet - Wetmore families - Rock outcrop complex, stony.
7 Gothic family.
8 Supervisor - Limber families complex.
9 Troutville family, very stony.
10 Bullwark - Catamount families - Rock outcrop complex, rubbly.
11 Bullwark - Catamount families - Rock land complex, rubbly.
12 Legault family - Rock land complex, stony.
13 Catamount family - Rock land - Bullwark family complex, rubbly.
14 Pachic Argiborolis - Aquolis complex.
15 unspecified in the USFS Soil and ELU Survey.
16 Cryaquolis - Cryoborolis complex.
17 Gateview family - Cryaquolis complex.
18 Rogert family, very stony.
19 Typic Cryaquolis - Borohemists complex.
20 Typic Cryaquepts - Typic Cryaquolls complex.
21 Typic Cryaquolls - Leighcan family, till substratum complex.
22 Leighcan family, till substratum, extremely bouldery.
23 Leighcan family, till substratum - Typic Cryaquolls complex.
24 Leighcan family, extremely stony.
25 Leighcan family, warm, extremely stony.
26 Granile - Catamount families complex, very stony.
27 Leighcan family, warm - Rock outcrop complex, extremely stony.
28 Leighcan family - Rock outcrop complex, extremely stony.
29 Como - Legault families complex, extremely stony.
30 Como family - Rock land - Legault family complex, extremely stony.
31 Leighcan - Catamount families complex, extremely stony.
32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.
34 Cryorthents - Rock land complex, extremely stony.
35 Cryumbrepts - Rock outcrop - Cryaquepts complex.
36 Bross family - Rock land - Cryumbrepts complex, extremely stony.
37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
38 Leighcan - Moran families - Cryaquolls complex, extremely stony.
39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
40 Moran family - Cryorthents - Rock land complex, extremely stony.

Diving into the dataset some more, we learned that there are 54 features. No attribute is missing as the count in each attribute equals the size of the dataset. Wilderness_Area and Soil_Type are one hot encoded. Soil_Type7 and Soil_Type15 can be removed from the dataset because they are constant.

We also learned that the dataset is not scaled the same. Several features in Soil_Type show a large skew. Hence, rescaling and normalization may be necessary.

The number of instances belonging to each class is equally presented. Hence, no re-balancing is necessary.


Exploration Visualization
algorithms and Techniques
Benchmark

Methodology
Data Preprocessing
Implementation
Refinement

Results
Model Evaluation and Validation
Justification

Conclusion
Reflection
Improvement
