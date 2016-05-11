package com.nenovinite.news.dataset;

import java.security.SecureRandom;
import java.util.Random;

import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;

import com.nenovinite.news.configuration.NewsConfiguration;

public class DatasetLoader {
	
	private static final long SEED = 11l;
	private final DataFrame credibleData;
	private final DataFrame unreliableData;
	private final DataFrame validationData;
	private final DataFrame train;
	private final DataFrame test;
	private final DataFrame validation;
	
	private void registerUDFs(final Random rand, SQLContext sqlContxt) {
		sqlContxt.udf().register("generateId", (String s) -> rand.nextInt(1000000), DataTypes.IntegerType);
		sqlContxt.udf().register("getFeatures", (String s) -> {
			int size = s.length();
			
			return size;
		}, DataTypes.IntegerType);
	}

	public DatasetLoader(SQLContext sqlContxt, double[] weights, NewsConfiguration conf) {
		final Random rand = new SecureRandom();
		this.registerUDFs(rand, sqlContxt);
		
		this.unreliableData = this.getBodyContent(sqlContxt, conf.getUnreliableDataset(), "content",
				"\nAND (category = \"Политика\")", 
				0.0);
		
		this.credibleData = this.getBodyContent(sqlContxt, conf.getCredibleDataset(), "BodyText", 
				"",
				1.0);
		
		this.validationData = this.getBodyContent(sqlContxt, "/home/momchil/Documents/MasterThesis/dataset/bazikileaks-data-extended.json", "content",
				"", 0.0);
		
		DataFrame[] unreliableSplits = this.getSplitsFromDF(this.getUnreliableData(), weights);
		DataFrame[] credibleSplits = this.getSplitsFromDF(this.getCredibleData(), weights);
		
		this.train = unreliableSplits[0].unionAll(credibleSplits[0]).orderBy("content").cache();
		this.test = unreliableSplits[1].unionAll(credibleSplits[1]).orderBy("content").cache();
		this.validation = validationData.unionAll(credibleSplits[1]).orderBy("content").cache();
	}
	
	private DataFrame[] getSplitsFromDF(DataFrame df, double[] weights) {
		return df.randomSplit(weights, SEED);
	}
	
	private DataFrame getBodyContent(SQLContext sqlContxt, String jsonPath, String bodyColumn,
			String whereClause, double label) {
		DataFrame df = sqlContxt.read().json(jsonPath);
		df.registerTempTable("news");
		df.printSchema();
		
		String sql = "SELECT\n"
				   + "  generateId('') AS id,\n"
				   + "	" + bodyColumn + " AS content,\n"
				   + "	CAST(" + label + " AS DOUBLE) AS label\n"
				   + "FROM news\n"
				   + "WHERE (nvl(" + bodyColumn + " , '') != '')\n"
				   + whereClause;
		DataFrame newsData = sqlContxt.sql(sql);
		
		return newsData;
	}
	
	public DataFrame getTrainingSet() {
		return train;
	}

	public DataFrame getTestingSet() {
		return test;
	}

	public DataFrame getValidationSet() {
		return validation;
	}

	public DataFrame getCredibleData() {
		return credibleData;
	}

	public DataFrame getUnreliableData() {
		return unreliableData;
	}
}
