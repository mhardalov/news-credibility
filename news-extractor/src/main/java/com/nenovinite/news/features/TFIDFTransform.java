package com.nenovinite.news.features;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.spark.HashPartitioner;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.StatCounter;

import com.google.common.collect.Multiset;

import scala.Tuple2;

public class TFIDFTransform implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8608677816826282768L;

	private final long newsCount;

	private long featuresCount;

	private Map<String, Tuple2<Integer, Long>> idf;

	private boolean verbose;

	public TFIDFTransform(long allNewsCount, boolean verbose) {
		this.newsCount = allNewsCount;
		this.setVerbose(verbose);
	}

	public TFIDFTransform(long allNewsCount) {
		this(allNewsCount, false);
	}

	public static double calcIDF(final long allNewsCount, long wordCount) {
		return Math.log(1 + (double) (allNewsCount) / (wordCount));
	}

	public void printIDFInfo(JavaPairRDD<String, Integer> idf) {
		StatCounter idfStats = idf.mapToDouble(row -> row._2).stats();
		System.out.println(idfStats);
	}

	private void saveIDFToFile(String path, JavaPairRDD<String, Integer> idf) {
		idf.map(word -> word._1() + "," + word._2()).repartition(1).saveAsTextFile(path);
	}

	public void extract(JavaPairRDD<Double, Multiset<String>> trainingDocs) {
		// .filter(word -> word._2 > 6 && word._2 < 1000);
		JavaPairRDD<Double, Multiset<String>> wordsClass0 = trainingDocs.filter(row -> row._1.equals(0.0)).cache();
		JavaPairRDD<Double, Multiset<String>> wordsClass1 = trainingDocs.filter(row -> row._1.equals(1.0)).cache();

		JavaPairRDD<String, Integer> idfClass0 = extractWordCount(wordsClass0).cache();
		JavaPairRDD<String, Integer> idfClass1 = extractWordCount(wordsClass1).cache();

		JavaPairRDD<String, Integer> idfWords = idfClass0.fullOuterJoin(idfClass1).mapToPair(joined -> {
			String word = joined._1;
			Integer left = joined._2()._1().orNull();
			Integer right = joined._2()._2().orNull();
			int count;

			if (left != null && right != null) {
				count = left.intValue() + right.intValue();
			} else {
				count = left != null ? left.intValue() : right.intValue();
			}

			return new Tuple2<>(word, count);
		}).partitionBy(new HashPartitioner(100)).filter(word -> word._2 > 10).cache();

		if (isVerbose()) {
			this.saveIDFToFile("/home/momchil/Desktop/master-thesis/datasets/stats/idf-final.csv", idfWords);
			this.saveIDFToFile("/home/momchil/Desktop/master-thesis/datasets/stats/idfClass0.csv", idfClass0);
			this.saveIDFToFile("/home/momchil/Desktop/master-thesis/datasets/stats/idfClass1.csv", idfClass1);
			this.printIDFInfo(idfWords);
		}

		wordsClass0.unpersist();
		wordsClass1.unpersist();
		idfClass0.unpersist();
		idfClass1.unpersist();

		JavaPairRDD<String, Tuple2<Integer, Long>> idfRdd = idfWords.zipWithIndex()
				.mapToPair(row -> new Tuple2<>(row._1._1, new Tuple2<>(row._1._2, row._2)));
		idf = idfRdd.collectAsMap();
		idfWords.unpersist();

		featuresCount = idf.size();
	}

	private JavaPairRDD<String, Integer> extractWordCount(JavaPairRDD<Double, Multiset<String>> wordsClass0) {
		return wordsClass0.flatMap(row -> row._2()).mapToPair(word -> new Tuple2<>(word, 1))
				.partitionBy(new HashPartitioner(10)).reduceByKey((a, b) -> a + b);
	}

	public LabeledPoint transform(Tuple2<Double, Multiset<String>> doc) {
		double label = doc._1();
		List<Tuple2<Integer, Double>> vector = new ArrayList<>();
		for (Multiset.Entry<String> entry : doc._2().entrySet()) {
			String word = entry.getElement();
			int tf = entry.getCount();

			Tuple2<Integer, Long> wordInfo = idf.get(word);
			if (wordInfo != null) {
				int index = wordInfo._2().intValue();
				double tfidf = calcTFIDF(tf, wordInfo);

				vector.add(new Tuple2<>(index, tfidf));
			}
		}
		Vector features = Vectors.sparse((int) featuresCount, vector);

		return new LabeledPoint(label, features);
	}

	private double calcTFIDF(int tf, Tuple2<Integer, Long> wordInfo) {
		double idfScore = calcIDF(this.newsCount, (wordInfo._2()));
		double tfidf = tf * idfScore;
		return tfidf;
	}

	public long getFeaturesCount() {
		return featuresCount;
	}

	public Map<String, Tuple2<Integer, Long>> getIdf() {
		return idf;
	}

	public boolean isVerbose() {
		return verbose;
	}

	public void setVerbose(boolean verbose) {
		this.verbose = verbose;
	}

}
