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

	public TFIDFTransform(long allNewsCount) {
		this.newsCount = allNewsCount;
	}

	public static double calcIDF(final long allNewsCount, long wordCount) {
		return Math.log(1 + (double) (allNewsCount) / (wordCount));
	}

	// public void idfInfo(final long allNewsCount, JavaPairRDD<String,
	// Tuple2<Integer, Long>> idfRdd) {
	//
	// idfRdd.map(word -> word._2._1 + "," + word._2._2()).repartition(1)
	// .saveAsTextFile("/home/momchil/Desktop/master-thesis/datasets/stats/idf.csv");
	//
	// StatCounter idfStats = idfRdd.mapToDouble(row ->
	// TFIDFTransform.calcIDF(allNewsCount, row._2._2)).stats();
	//
	// System.out.println(idfStats);
	// for (Tuple2<String, Tuple2<Integer, Long>> word :
	// idfRdd.takeOrdered(1000, new WordComparator())) {
	// System.out.println(word._1() + "," + word._2._1);
	// }
	// }

	public void extract(JavaPairRDD<Double, Multiset<String>> trainingDocs) {
		// .filter(word -> word._2 > 6 && word._2 < 1000)
		JavaPairRDD<String, Integer> idfWords = trainingDocs.flatMap(row -> row._2())
				.mapToPair(word -> new Tuple2<>(word, 1)).partitionBy(new HashPartitioner(10))
				.reduceByKey((a, b) -> a + b).cache();

		JavaPairRDD<String, Tuple2<Integer, Long>> idfRdd = idfWords.zipWithIndex()
				.mapToPair(row -> new Tuple2<>(row._1._1, new Tuple2<>(row._1._2, row._2))).cache();

		// idfInfo(allNewsCount, idfRdd);

		idf = idfRdd.collectAsMap();

		featuresCount = idfRdd.count();
		idfRdd.unpersist();
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

}
