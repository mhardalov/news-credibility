package com.credibility;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import scala.Tuple2;

public class TwittterMain {

	/**
	 * Returns a list with all links contained in the input
	 */
	public static boolean extractUrls(String text) {
		// List<String> containedUrls = new ArrayList<String>();
		String urlRegex = "\\b(?:(?:(?:https?|ftp|file)://|www\\.|ftp\\.)[-A-Z0-9+&@#/%?=~_|$!:,.;]*[-A-Z0-9+&@#/%=~_|$]| ((?:mailto:)?[A-Z0-9._%+-]+@[A-Z0-9._%-]+\\.[A-Z]{2,4})\\b)|\"(?:(?:https?|ftp|file)://|www\\.|ftp\\.)[^\"\\r\\n]+\"?|'(?:(?:https?|ftp|file)://|www\\.|ftp\\.)[^'\\r\\n]+'?";
		Pattern pattern = Pattern.compile(urlRegex, Pattern.CASE_INSENSITIVE);
		Matcher urlMatcher = pattern.matcher(text);

		return urlMatcher.matches();

		// while (urlMatcher.find())
		// {
		// containedUrls.add(text.substring(urlMatcher.start(0),
		// urlMatcher.end(0)));
		// }
		//
		// return containedUrls;
	}

	public static void main(String[] args) {

		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Twitter credibility");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContext = new SQLContext(sc);

			DataFrame semevalDF = sqlContext.read().format("com.databricks.spark.xml")
					.option("rowTag", "OrgQuestion").load("/home/momchil/Desktop/master-thesis/datasets/semeval/SemEval2016-Task3-CQA-QL-train-part1.xml,"
							+ "/home/momchil/Desktop/master-thesis/datasets/semeval/SemEval2016-Task3-CQA-QL-train-part2.xml");
			semevalDF.printSchema();
			semevalDF.registerTempTable("semeval");
			semevalDF = semevalDF.select("Thread.RelComment.@RELC_ID", "Thread.RelComment.RelCText");

			semevalDF.printSchema();
			semevalDF.show();

			JavaPairRDD<String, Vector> semevalMap = semevalDF.javaRDD().map(row -> {
				List<String> ids = new LinkedList<>(row.getList(0));
				List<String> texts = new LinkedList<>(row.getList(1));
				List<Tuple2<String, Vector>> rows = new LinkedList<>();
				// String text = row.getString(1);

				for (int i = 0; i < ids.size(); i++) {
					String text = texts.get(i);
					String id = ids.get(i);
					Vector vector = textToVector(text);

					rows.add(new Tuple2<>(id, vector));
				}

				return rows;
			}).flatMapToPair(row -> row);

			DataFrame df = sqlContext.read().format("com.databricks.spark.csv").option("header", "true")
					.option("inferSchema", "true")
					.load("/home/momchil/Desktop/twitter_download-master/downloaded-2016-01-24-19-44.csv");

			DataFrame labels = sqlContext.read().format("com.databricks.spark.csv").option("header", "true")
					.option("inferSchema", "true")
					.load("/home/momchil/Desktop/master-thesis/datasets/credibility/labels-trendid-credible.csv");

			df = df.where("TEXT <> 'Not Available'");
			df.printSchema();
			labels.printSchema();

			JavaPairRDD<String, Integer> tweetsAndLabels = df.javaRDD().mapToPair(row -> {
				return new Tuple2<>(row.getString(0), row.getString(2));
			}).join(labels.javaRDD().mapToPair(row -> {
				return new Tuple2<>(row.getString(0), row.getString(1));
			})).mapToPair(row -> {
				String labelStr = row._2._2;
				String text = row._2._1;

				int label = labelStr.equals("CREDIBLE") ? 1 : 0;
				return new Tuple2<>(text, label);

			});

			JavaRDD<LabeledPoint> data = tweetsAndLabels.map(row -> {
				String text = row._1;
				Vector vector = textToVector(text);
				return new LabeledPoint((double) row._2, vector);
			});

			// Split initial RDD into two... [60% training data, 40% testing
			// data].
			JavaRDD<LabeledPoint> training = data.sample(false, 0.9, 11L);
			training.cache();
			StandardScalerModel scaler = new StandardScaler().fit(training.map(row -> row.features()).rdd());

			JavaRDD<LabeledPoint> test = data.subtract(training);

			training = training.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));
			test = test.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));

			// Run training algorithm to build the model.
			int numIterations = 100;
			final SVMModel model = SVMWithSGD.train(training.rdd(), numIterations);
			final SVMModel model_prob = SVMWithSGD.train(training.rdd(), numIterations);
			model_prob.clearThreshold();

			// Run training algorithm to build the model.
			// LogisticRegressionModel model = new
			// LogisticRegressionWithLBFGS().setNumClasses(2).run(training.rdd());

			// Clear the prediction threshold so the model will return
			// probabilities
			// model.clearThreshold();

			// Compute raw scores on the test set.
			JavaRDD<Tuple2<Object, Object>> predictionAndLabels = test.map(p -> {
				Double prediction = model.predict(p.features());
				return new Tuple2<Object, Object>(prediction, p.label());
			});

			semevalMap.map(row -> {
				String id = row._1;
				Vector vec = scaler.transform(row._2);
				Double prediction = model.predict(vec);
				Double prediction_prob = model_prob.predict(vec);

				String output = id + "\t" + prediction + "\t" + prediction_prob + "\t"
						+ row._2().toString().replace(",", "\t").replace("[", "").replace("]", "");
				return output;
			}).repartition(1).saveAsTextFile("/home/momchil/Desktop/master-thesis/datasets/semeval/features.tsv");

			// Get evaluation metrics.
			MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
			double precision = metrics.precision();
			double f1 = metrics.fMeasure();

			long correctZeros = predictionAndLabels
					.filter(row -> (double) row._1 == (double) row._2 && (double) row._1 == 0.0).count();
			long predictedZeros = predictionAndLabels.filter(row -> (double) row._1 == 0.0).count();
			long testCount = test.count();
			long countZeros = test.filter(row -> row.label() == 0.0).count();

			System.out.println("Precision = " + precision);
			System.out.println("F1 = " + f1);

			System.out.println("Testing set size: " + testCount);
			System.out.println("Count zeros: " + countZeros);
			System.out.println("Correct zeros: " + correctZeros);
			System.out.println("Predicted zeros: " + predictedZeros);

		}

	}

	private static Vector textToVector(String text) {
		Set<String> hashSet = new HashSet<String>();
		hashSet.addAll(Arrays.asList(text.toLowerCase().split("\\s+")));
		double hasQuestionMark = text.contains("?") || text.contains("!") ? 1.0 : 0.0;
		double userMetion = text.contains("@") ? 1.0 : 0.0;

		double hasEmoticon = hashSet.contains(":)") || hashSet.contains(":(") ? 1.0 : 0.0;

		double hasFirstPersion = hashSet.contains("me") || hashSet.contains("i") || hashSet.contains("my")
				|| hashSet.contains("mine") || hashSet.contains("we") || hashSet.contains("our")
				|| hashSet.contains("ours") || hashSet.contains("us") ? 1.0 : 0.0;
		double hasThirdPersion = hashSet.contains("he") || hashSet.contains("she") || hashSet.contains("it")
				|| hashSet.contains("his") || hashSet.contains("hers") || hashSet.contains("him")
				|| hashSet.contains("her") || hashSet.contains("they") || hashSet.contains("them")
				|| hashSet.contains("their") ? 1.0 : 0.0;

		double hasUrl = TwittterMain.extractUrls(text) ? 1.0 : 0.0;

		Vector vector = Vectors.dense(7, text.length(), hasQuestionMark, hasEmoticon, hasFirstPersion, hasThirdPersion,
				userMetion, hasUrl);
		return vector;
	}

}
