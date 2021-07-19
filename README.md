# people-detection

This script was built to complement another private project I'm working on that requires filtering out photos where a person is a central point of focus.

To achieve this, the requirements set out in this script to delete files in the "data" directory are:
-People == 0
-People > 3
-Dimensions of largest "people" bounding box vs Dimension of image < 0.1
