package com.nenovinite.news.features.extractors;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.StopWordsRemover;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.storage.StorageLevel;

import com.nenovinite.news.features.TokenTransform;

import scala.Tuple2;
import scala.collection.immutable.List;

public class Word2VecExtractor {

	public static double similarity(double[] v1, double[] v2) {
		double score = 0;
		double dotProd = 0;
		double normV1 = 0;
		double normV2 = 0;
		for (int i = 0; i < v1.length; i++) {
			dotProd += v1[i] * v2[i];
			normV1 += Math.pow(v1[i], 2);
			normV2 += Math.pow(v2[i], 2);
		}
		score = dotProd / (Math.sqrt(normV1) * Math.sqrt(normV2));

		return score;
	}

	public static double[] addition(double[] v1, double[] v2) {
		double[] addition = Arrays.copyOf(v1, v1.length);

		for (int i = 0; i < v1.length; i++) {
			addition[i] += v2[i];
		}

		return addition;
	}

	public static double compare(Word2VecModel fit, String word1, String word2) {
		Vector vector1 = fit.transform(word1);
		Vector vector2 = fit.transform(word2);

		double[] v1 = vector1.toArray();
		double[] v2 = vector2.toArray();
		return similarity(v1, v2);
	}

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Parser news");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);
			DataFrame df = sqlContxt.read().json("/home/momchil/Desktop/master-thesis/w2v/long-abstracts_bg.json");
			df.registerTempTable("news");
			df.printSchema();
			df = df.select("abstract").where("abstract IS NOT NULL").limit(10000);
			df.show(50);

			// RegexTokenizer regexTokenizer = new
			// RegexTokenizer().setInputCol("abstract").setOutputCol("words")
			// .setPattern("\\w+").setGaps(false);
			// df = regexTokenizer.transform(df);
			//
			// StopWordsRemover remover = new StopWordsRemover()
			// .setStopWords(TokenTransform.STOP_WORDS.toArray(new
			// String[TokenTransform.STOP_WORDS.size()]))
			// .setInputCol("words").setOutputCol("filtered");
			// df = remover.transform(df);
			//
			// NGram ngramTransformer = new
			// NGram().setN(2).setInputCol("filtered").setOutputCol("ngrams");
			// df = ngramTransformer.transform(df);
			//
			// JavaRDD<List<String>> words = df.select("ngrams",
			// "filtered").javaRDD().map(row -> {
			// java.util.List<String> ngrams = new LinkedList<>(row.getList(0));
			// java.util.List<String> filtered = row.getList(1);
			// ngrams.addAll(filtered);
			//
			// return
			// scala.collection.JavaConversions.asScalaBuffer(ngrams).toList();
			// }).cache();
			//
			// System.out.println("Vocabilary: " + words.count());

			JavaRDD<List<String>> words = df.javaRDD().map(row -> {
				String[] split = row.getString(0).split(" ");
				ArrayList<String> wordsList = new ArrayList<String>();

				for (int i = 0; i < split.length - 1; i++) {
					wordsList.add(split[i] + " " + split[i + 1]);

				}
				List<String> wl = scala.collection.JavaConversions.asScalaBuffer(wordsList).toList();

				return wl;
			}).persist(StorageLevel.MEMORY_AND_DISK_SER());

			Word2Vec w2v = new Word2Vec().setMinCount(3);
			Word2VecModel fit = w2v.fit(words.rdd());
			for (Tuple2<String, Object> word : fit.findSynonyms("3 май", 100)) {
				System.out.println(word);
			}
			// Vector vector1 = fit.transform("Бойко");
			// Vector vector2 = fit.transform("Борисов");
			// Vector vector3 = fit.transform("МВР");

			// double[] v1 = vector1.toArray();
			// double[] v2 = vector2.toArray();
			// double[] v3 = vector3.toArray();
			// System.out.println(similarity(v1, v2));
			// System.out.println(similarity(v2, v3));

			// fit.save(sc.sc(),
			// "/home/momchil/Desktop/master-thesis/w2v/model/wikipedia/bigrams/bg");
		}

	}

}
