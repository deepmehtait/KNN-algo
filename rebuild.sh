#!/bin/bash

dirPath="/user/user01/LAB3_SUBMISSION/E2"
mkdir $dirPath/classes
javac -d $dirPath/classes $dirPath/KnnAlgo.java
jar -cvf knn.jar -C $dirPath/classes/ .
mv knn.jar $dirPath/

