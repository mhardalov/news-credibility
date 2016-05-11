package com.nenovinite.news.dataset;

import org.apache.commons.cli.ParseException;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

import com.nenovinite.news.configuration.NewsConfiguration;

public class FindSimilarNews {

	public static void main(String[] args) throws ParseException {
		final NewsConfiguration conf = new NewsConfiguration(args);
		
		SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("News Classificator");
		try (JavaSparkContext sc = new JavaSparkContext(sparkConf)) {
			SQLContext sqlContxt = new SQLContext(sc);
			double[] weights = new double[] { 0.7, 0.3 };

			DatasetLoader dataset = new DatasetLoader(sqlContxt, weights, conf);
			dataset.getUnreliableData().selectExpr("content as c_content").registerTempTable("unreliable");
			dataset.getCredibleData().selectExpr("content as u_content").registerTempTable("credible");
			
			String sqlText = "SELECT c_content, u_content\n" +
							 "FROM unreliable\n"+
							 "CROSS JOIN credible";
			
			DataFrame df = sqlContxt.sql(sqlText);
			df.show();
			System.out.println(df.count());
		}
	}

}
