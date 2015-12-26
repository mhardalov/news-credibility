package com.nenovinite.news;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import com.google.common.collect.Multiset;
import com.nenovinite.news.features.TFIDFTransform;
import com.nenovinite.news.features.TokenTransform;
import com.nenovinite.news.models.ModelBase;
import com.nenovinite.news.models.ModelNaiveBayes;

public class Main {

	private static DataFrame getBodyContent(SQLContext sqlContxt, String jsonPath, String bodyColumn,
			String whereClause, double label) {
		DataFrame df = sqlContxt.read().json(jsonPath);
		df.registerTempTable("news");
		df.printSchema();

		String sql = "SELECT " + bodyColumn + " as content, CAST(" + label + " as DOUBLE) as label FROM news "
				+ whereClause;
		DataFrame newsData = sqlContxt.sql(sql);
		return newsData;
	}

	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Parser news");
		JavaSparkContext sc = new JavaSparkContext(conf);
		SQLContext sqlContxt = new SQLContext(sc);

		DataFrame newsData = getBodyContent(sqlContxt,
				"/home/momchil/Desktop/master-thesis/datasets/ne!novinite-data-extended.json", "content",
				" WHERE category = \"Политика\" ", 0.0).cache();
		final long neNoviniteCount = newsData.count();

		// " LIMIT 15000"
		newsData = newsData.unionAll(getBodyContent(sqlContxt,
				"/home/momchil/Desktop/master-thesis/datasets/Publications-all-2013-01-01-2015-04-01.json", "BodyText",
				" ", 1.0));

		// Random shuffle
		newsData = newsData.sort("content").cache();

		final long allNewsCount = newsData.count();

		JavaPairRDD<Double, Multiset<String>> docs = newsData.javaRDD().mapToPair(TokenTransform::transform).cache();

		// Split initial RDD into two... [60% training data, 40% testing data].
		JavaPairRDD<Double, Multiset<String>> trainingDocs = docs.sample(false, 0.6, 11L);
		trainingDocs.cache();
		JavaPairRDD<Double, Multiset<String>> testDocs = docs.subtract(trainingDocs);

		TFIDFTransform tfIdf = new TFIDFTransform(allNewsCount);
		tfIdf.extract(trainingDocs);

		JavaRDD<LabeledPoint> training = trainingDocs.map(tfIdf::transform);
		JavaRDD<LabeledPoint> test = testDocs.map(tfIdf::transform);

		ModelBase model = new ModelNaiveBayes(training);
		model.evaluate(test);

		System.out.println("Ne!Novite news:" + neNoviniteCount);
		System.out.println("Dnevnik news:" + (allNewsCount - neNoviniteCount));

		sc.close();

	}

}
