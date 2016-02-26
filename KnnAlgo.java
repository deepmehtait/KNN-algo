package cs286;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;

public class KnnAlgo {

	private static int holdoutPercentageIndex;
	private static int k;
	private static String similarity;

	public static void main(String[] args) {

		int holdoutPercent = 0;
		try {
			holdoutPercent = Integer.parseInt(args[0]);
		} catch (NumberFormatException nfe) {
			System.err.println("Holdout percentage must be an integer.");
			System.exit(1);
		}

		if (holdoutPercent < 5 || holdoutPercent > 20) {
			System.err.println("Holdout percentage must between 5 and 20.");
			System.exit(1);
		}

		try {
			k = Integer.parseInt(args[1]);
			if (k == 0)
				throw new NumberFormatException();
		} catch (NumberFormatException nfe) {
			System.err.println("Number of neighbours (k) must be a positive integer.");
			System.exit(1);
		}

		similarity = args[2];

		if (!("euclidean".equals(similarity) || "cosine".equals(similarity))) {
			System.err.println("Please specify similarity as either euclidean or cosine.");
			System.exit(1);
		}

		Path dataFilePath = Paths.get(args[3]);
		if (Files.notExists(dataFilePath)) {
			System.err.println("Input file does not exist on given path: " + dataFilePath.toString());
			System.exit(1);
		}

		Path outputFilePath = Paths.get(args[4]);
		if (Files.exists(outputFilePath)) {
			System.err.println("Output file already exist on path: " + outputFilePath.toString());
			System.exit(1);
		}

		double[][] data = readIrisDataFile(dataFilePath);

		int totalDatapointsPerClass = data.length / 3;
		holdoutPercentageIndex = (int) (totalDatapointsPerClass * (1 - (holdoutPercent / 100.0))) - 1;

		KnnAlgo knn = new KnnAlgo();

		double[][] trainingData = knn.holdoutTrainingData(data);
		double[][] testData = knn.holdoutTestData(data);

		KnnResult[] knnResults = knn.train(trainingData, testData);
		double[] accuracy = knn.calculateAccuracy(testData, knnResults);
		writeResultToOutputFile(accuracy, outputFilePath);
	}

	private static double[][] readIrisDataFile(Path dataFilePath) {

		try (BufferedReader reader = Files.newBufferedReader(dataFilePath, Charset.defaultCharset())) {
			double[][] data = new double[150][5];
			String line;
			int datapointCount = 0;
			while ((line = reader.readLine()) != null) {
				String[] features = line.split("\\t");
				int featureCount = 0;
				for (String feature : features) {
					data[datapointCount][featureCount] = Double.parseDouble(feature);
					featureCount++;
				}
				datapointCount++;
			}
			return data;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	private static void writeResultToOutputFile(double[] accuracy, Path outputFilePath) {

		try (BufferedWriter writer = Files.newBufferedWriter(outputFilePath, StandardCharsets.UTF_8,
				StandardOpenOption.CREATE_NEW)) {
			writer.write(String.format("k = %s%ndistance = %s%n", k, similarity));
			for (int i = 0; i < accuracy.length; i++) {
				writer.write(String.format("average accuracy for species %d = %s%n", (i + 1), accuracy[i]));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	public double[][] holdoutTrainingData(double[][] data) {

		double[][] trainData1 = copyMatrix(data, 0, holdoutPercentageIndex);
		double[][] trainData2 = copyMatrix(data, 50, 50 + holdoutPercentageIndex);
		double[][] trainData12 = appendMatrix(trainData1, trainData2);
		double[][] trainData3 = copyMatrix(data, 100, 100 + holdoutPercentageIndex);
		double[][] trainData = appendMatrix(trainData12, trainData3);

		return trainData;
	}

	public double[][] holdoutTestData(double[][] data) {

		double[][] testData1 = copyMatrix(data, 1 + holdoutPercentageIndex, 49);
		double[][] testData2 = copyMatrix(data, 51 + holdoutPercentageIndex, 99);
		double[][] testData12 = appendMatrix(testData1, testData2);
		double[][] testData3 = copyMatrix(data, 101 + holdoutPercentageIndex, 149);
		double[][] testData = appendMatrix(testData12, testData3);
		return testData;
	}

	public KnnResult[] train(double[][] trainingData, double[][] testData) {
		int testDatapointIndex = 0;
		KnnResult[] knnResults = new KnnResult[testData.length];
		for (double[] testDatapoints : testData) {
			int trainingDatapointIndex = 0;
			Distance[] distances = new Distance[trainingData.length];
			for (double[] trainingDatapoints : trainingData) {
				if ("euclidean".equals(similarity))
					distances[trainingDatapointIndex] = new Distance(trainingDatapointIndex++,
							calEuclideanDistance(trainingDatapoints, testDatapoints));
				else
					distances[trainingDatapointIndex] = new Distance(trainingDatapointIndex++,
							calCosineDistance(trainingDatapoints, testDatapoints));
			}
			knnResults[testDatapointIndex] = new KnnResult(testDatapointIndex++, distances);
		}

		for (KnnResult knnResult : knnResults) {
			knnResult.calculateKnn(k, similarity);
			int[] knnIndexes = knnResult.getKnnIndexes();
			double[] speciesIds = new double[knnIndexes.length];
			int speciesCount = 0;
			for (int knnIndex : knnIndexes) {
				speciesIds[speciesCount++] = trainingData[knnIndex][4];
			}
			knnResult.setKnnSpeciesIds(speciesIds);
			knnResult.identifySpeciesFromKnn();
		}
		return knnResults;
	}

	
	private double calCosineDistance(double[] vectorA, double[] vectorB) {
		double dotProduct = 0.0;
		double normA = 0.0;
		double normB = 0.0;
		for (int i = 0; i < vectorA.length - 1; i++) {
			dotProduct += vectorA[i] * vectorB[i];
			normA += Math.pow(vectorA[i], 2);
			normB += Math.pow(vectorB[i], 2);
		}
		return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}
	
	private double calEuclideanDistance(double[] vectorA, double[] vectorB) {
		double total = 0.0;
		for (int i = 0; i < vectorA.length - 1; i++) {
			total += Math.pow(vectorA[i] - vectorB[i], 2);
		}
		return Math.sqrt(total);
	}

	public double[] calculateAccuracy(double[][] testData, KnnResult[] knnResults) {
		int testDatapointIndex = 0;
		double[] accuracy = new double[3];
		int[] trueCount = new int[3];
		int[] total = new int[3];
		for (double[] testDatapoints : testData) {
			double actualSpeciesId = testDatapoints[4];
			total[(int) actualSpeciesId]++;
			if (Double.compare(actualSpeciesId, knnResults[testDatapointIndex].getPredictedSpeciesId()) == 0) {
				trueCount[(int) actualSpeciesId]++;
			}

			testDatapointIndex++;
		}

		for (int i = 0; i < accuracy.length; i++) {
			accuracy[i] = trueCount[i] / total[i];

		}

		return accuracy;
	}


	private double[][] copyMatrix(double[][] src, int start, int end) {
		double[][] target = new double[(end - start) + 1][src[0].length];
		for (int i = start; i <= end; i++) {
			System.arraycopy(src[i], 0, target[i - start], 0, src[i].length);
		}
		return target;
	}

	private double[][] appendMatrix(double[][] a, double[][] b) {
		double[][] result = new double[a.length + b.length][];
		System.arraycopy(a, 0, result, 0, a.length);
		System.arraycopy(b, 0, result, a.length, b.length);
		return result;
	}

	private void printMatrix(double[][] matrix) {
		System.out.println("-------------------------");
		int i = 1;
		for (double[] datapoints : matrix) {
			System.out.print((i++) + "-> ");
			for (double feature : datapoints)
				System.out.print(feature + " ");
			System.out.println();
		}
		System.out.println("-------------------------");
	}
}

/*
 * KnnResults
 */
class KnnResult {

	private int datapointIndex;
	private Distance[] distances;
	private int[] knnIndexes;
	private double[] knnSpeciesIds;
	private double predictedSpeciesId;

	public KnnResult(int datapointIndex, Distance[] distances) {
		this.datapointIndex = datapointIndex;
		this.distances = distances;
	}

	public double[] getKnnSpeciesIds() {
		return knnSpeciesIds;
	}

	public void setKnnSpeciesIds(double[] knnSpeciesIds) {
		this.knnSpeciesIds = knnSpeciesIds;
	}

	public int getDatapointIndex() {
		return datapointIndex;
	}

	public void setDatapointIndex(int datapointIndex) {
		this.datapointIndex = datapointIndex;
	}

	public Distance[] getDistances() {
		return distances;
	}

	public void setDistances(Distance[] distances) {
		this.distances = distances;
	}

	public int[] getKnnIndexes() {
		return knnIndexes;
	}

	public void setKnnIndexes(int[] knnIndexes) {
		this.knnIndexes = knnIndexes;
	}

	public double getPredictedSpeciesId() {
		return predictedSpeciesId;
	}

	public void setPredictedSpeciesId(double predictedSpeciesId) {
		this.predictedSpeciesId = predictedSpeciesId;
	}

	public void calculateKnn(int k, String similarity) {
		Arrays.sort(this.distances);
		int lastDistanceIndex = this.distances.length - 1;
		this.knnIndexes = new int[k];
		for (int i = 0; i < k; i++) {
			if ("euclidean".equals(similarity))
				this.knnIndexes[i] = this.distances[i].getIndex();
			else
				this.knnIndexes[i] = this.distances[lastDistanceIndex - i].getIndex();
		}
	}

	public void identifySpeciesFromKnn() {
		int count_0 = 0;
		int count_1 = 0;
		int count_2 = 0;
		for (double speciesId : this.knnSpeciesIds) {
			switch (speciesId + "") {
			case "0.0":
				count_0++;
				break;

			case "1.0":
				count_1++;
				break;

			case "2.0":
				count_2++;
				break;
			}
		}

		this.predictedSpeciesId = count_0 > count_1 ? (count_0 > count_2 ? 0.0 : 2.0) : (count_1 > count_2 ? 1.0 : 2.0);
	}
}

/*
 * 
 * 
 */

class Distance implements Comparable<Distance> {

	private int index;
	private double distance;

	public Distance(int index, double distance) {
		this.index = index;
		this.distance = distance;
	}

	public int getIndex() {
		return index;
	}

	public void setIndex(int index) {
		this.index = index;
	}

	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	@Override
	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof Distance))
			return false;

		Distance distance1 = (Distance) o;

		return Double.compare(distance1.distance, distance) == 0;
	}

	@Override
	public int hashCode() {
		long temp = Double.doubleToLongBits(distance);
		return (int) (temp ^ (temp >>> 32));
	}

	@Override
	public int compareTo(Distance o) {
		return Double.compare(this.distance, o.getDistance());
	}
}
