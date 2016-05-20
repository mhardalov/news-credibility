package com.nenovinite.news.features.export;

import org.apache.commons.cli.ParseException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import com.nenovinite.news.NewsCredibilityMain;
import com.nenovinite.news.configuration.NewsConfiguration;
import com.nenovinite.news.dataset.DatasetLoader;

public class ExportCommonFeatures {

	public static void main(String[] args) throws ParseException {
		final NewsConfiguration conf = new NewsConfiguration(args);

		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("News Classificator");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);
			
			DatasetLoader data = new DatasetLoader(sqlContxt, new double[] {0.7, 0.}, conf);
			
			DataFrame allData = data.getBazikiLeaks().unionAll(data.getCredibleData()).unionAll(data.getUnreliableData()).unionAll(data.getValidationSet());
			NewsCredibilityMain
				.getCommonFeatures(sqlContxt, allData, "tokens")
				.select("label", "commonfeatures")
				.repartition(1)
				.write()
			    .format("com.databricks.spark.csv")
			    .option("header", "true")
			    .save("/home/momchil/Documents/MasterThesis/features/commonfeatures.csv");
			
		}
	}
}
