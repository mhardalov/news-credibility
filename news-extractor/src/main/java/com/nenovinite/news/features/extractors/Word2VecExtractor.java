package com.nenovinite.news.features.extractors;

import java.util.Arrays;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.NGram;
import org.apache.spark.ml.feature.RegexTokenizer;
import org.apache.spark.ml.feature.Word2Vec;
import org.apache.spark.ml.feature.Word2VecModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

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

//	public static double compare(Word2VecModel fit, String word1, String word2) {
//		Vector vector1 = fit.transform(word1);
//		Vector vector2 = fit.transform(word2);
//
//		double[] v1 = vector1.toArray();
//		double[] v2 = vector2.toArray();
//		return similarity(v1, v2);
//	}
	
	public static DataFrame getTrainingDataset(SQLContext sqlContxt) {
		DataFrame df = sqlContxt.read().json("/home/momchil/Documents/MasterThesis/dataset/w2v/long-abstracts_bg.json");
		df.registerTempTable("dbpedia");
		df.printSchema();
		
		String sqlText = 
				"SELECT abstract as content\n"
				+ "FROM dbpedia\n"
				+ "WHERE abstract IS NOT NULL\n"
				+ "LIMIT 171444";
		df = sqlContxt.sql(sqlText);
		
		return df;
	}
	
	public static Word2VecModel trainw2v(DataFrame df, String outputColumn) {
		RegexTokenizer tokenizer = new RegexTokenizer()
				  .setInputCol("content")
				  .setOutputCol("tokens")
				  .setPattern("\\w+").setGaps(false);
		df = tokenizer.transform(df);
		
		NGram ngramTransformer = new NGram()
				.setInputCol("tokens")
				.setOutputCol("ngrams");
		
		df = ngramTransformer.transform(df);

		// Learn a mapping from words to Vectors.
		Word2Vec word2Vec = new Word2Vec()
		  .setInputCol("tokens")
		  .setOutputCol(outputColumn)
		  .setVectorSize(1)
		  .setMinCount(3)
		  .setNumPartitions(100)
		  .setMaxIter(1);
		Word2VecModel model = word2Vec.fit(df);
		
		return model;
	}
	

	public static void main(String[] args) {
		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Parser news");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);
			DataFrame df = getTrainingDataset(sqlContxt);
			df.show(50);

			trainw2v(df, "w2v");

//			for (Tuple2<String, Object> word : fit.findSynonyms("3 май", 100)) {
//				System.out.println(word);
//			}

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
