package com.nenovinite.news;

import org.apache.commons.cli.ParseException;
import org.apache.spark.Accumulator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.feature.Normalizer;
import org.apache.spark.mllib.feature.StandardScaler;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import com.google.common.collect.Multiset;
import com.nenovinite.news.configuration.NewsConfiguration;
import com.nenovinite.news.features.TFIDFTransform;
import com.nenovinite.news.features.TokenTransform;
import com.nenovinite.news.models.ModelBase;

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

	public static void main(String[] args) throws ParseException {
		NewsConfiguration conf = new NewsConfiguration(args);

		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("Parser news");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);

			long seed = 11l;
			double[] weights = new double[] { 0.6, 0.4 };

			// Split initial RDD into two... [60% training data, 40% testing
			// data].
			DataFrame[] unreliableData = getBodyContent(sqlContxt, conf.getUnreliableDataset(), "content",
					" WHERE category = \"Политика\" ", 0.0).randomSplit(weights, seed);

			// " LIMIT 15000"
			DataFrame[] credibleData = getBodyContent(sqlContxt, conf.getCredibleDataset(), "BodyText", " LIMIT 35000",
					1.0).randomSplit(weights, seed);

			TokenTransform tokenizer = new TokenTransform(conf.isVerbose());

			// Random shuffle by sort by content
			JavaPairRDD<Double, Multiset<String>> trainingDocs = unreliableData[0].unionAll(credibleData[0])
					.sort("content").javaRDD().mapToPair(tokenizer::transform).cache();
			trainingDocs.cache();

			JavaPairRDD<Double, Multiset<String>> testDocs = unreliableData[1].unionAll(credibleData[1]).sort("content")
					.javaRDD().mapToPair(tokenizer::transform);
			testDocs.cache();

			long trainingCount = trainingDocs.count();

			TFIDFTransform tfIdf = new TFIDFTransform(trainingCount, conf.isVerbose());
			tfIdf.extract(trainingDocs);

			JavaRDD<LabeledPoint> training = trainingDocs.map(tfIdf::transform);
			training.cache();
			trainingDocs.unpersist();
			StandardScalerModel scaler = new StandardScaler().fit(training.map(row -> row.features()).rdd());
			training = training.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));

			JavaRDD<LabeledPoint> test = testDocs.map(tfIdf::transform);
			test = test.map(row -> new LabeledPoint(row.label(), scaler.transform(row.features())));
			test.cache();
			testDocs.unpersist();

			ModelBase model = conf.getModel(training);
			training.unpersist();

			test.cache();

			long unreliableCount = testDocs.filter(row -> row._1 == 0).count();
			long credibleCount = testDocs.filter(row -> row._1 == 1).count();

			Accumulator<Integer> counterFor0 = sc.accumulator(0);
			Accumulator<Integer> corrcetFor0 = sc.accumulator(0);
			String output = model.evaluate(test, counterFor0, corrcetFor0);

			test.unpersist();

			System.out.println(output);
			System.out.println("Classified as Ne!Novinite: " + counterFor0.value());
			System.out.println("Correct Ne!Novinite: " + corrcetFor0.value());
			System.out.println("Features count:" + tfIdf.getFeaturesCount());

			System.out.println("Ne!Novite news:" + unreliableCount);
			System.out.println("Dnevnik news:" + credibleCount);
		}
	}

}
