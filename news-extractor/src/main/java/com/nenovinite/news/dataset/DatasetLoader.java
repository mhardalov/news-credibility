package com.nenovinite.news.dataset;

import java.security.SecureRandom;
import java.util.Random;

import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.DataTypes;

import com.nenovinite.news.configuration.NewsConfiguration;

import scala.collection.mutable.WrappedArray;

public class DatasetLoader {
	
	private static final long SEED = 11l;
	private static final String CREDIBLE_LABEL = "1.0";
	private static final String UNRELIABLE_LABEL = "0.0";
	
	private final DataFrame credibleData;
	private final DataFrame unreliableData;
	private final DataFrame validationData;
	private final DataFrame train;
	private final DataFrame test;
	private final DataFrame validation;
	private final DataFrame bazikiLeaks;
	
	private void registerUDFs(final Random rand, SQLContext sqlContxt) {
		sqlContxt.udf().register("generateId", (String s) -> rand.nextInt(1000000), DataTypes.IntegerType);
		sqlContxt.udf().register("getVectorLength", (WrappedArray<String> tokens) -> {
			Double size = (double) tokens.length();
			
			return Math.max(Double.valueOf(1.0), size);
		}, DataTypes.DoubleType);
		
		sqlContxt.udf().register("categoryToLabel", (String cat) -> {
			return (cat.equals("Лайфстайл")) ? Double.parseDouble(CREDIBLE_LABEL) : Double.parseDouble(UNRELIABLE_LABEL);
		}, DataTypes.DoubleType);
	}
	
	private DataFrame getBodyContent(SQLContext sqlContxt, String jsonPath, String bodyColumn,
			String whereClause, String label) {
		DataFrame df = sqlContxt.read().json(jsonPath);
		df.registerTempTable("news");
		df.printSchema();
		
		String sql = "SELECT\n"
				   + "  generateId('') AS id,\n"
				   + "	" + bodyColumn + " AS content,\n"
				   + "	CAST(" + label + " AS Double) AS label\n"
				   + "FROM news\n"
				   + "WHERE (trim(nvl(" + bodyColumn + " , '')) != '')\n"
				   + whereClause;
		DataFrame newsData = sqlContxt.sql(sql);
		
		return newsData;
	}

	public DatasetLoader(SQLContext sqlContxt, double[] weights, NewsConfiguration conf) {
		final Random rand = new SecureRandom();
		this.registerUDFs(rand, sqlContxt);
		
		this.unreliableData = this.getBodyContent(sqlContxt, conf.getUnreliableDataset(), "content",
//				\nAND (category = \"Политика\")
				"", 
				UNRELIABLE_LABEL);
		
		this.credibleData = this.getBodyContent(sqlContxt, conf.getCredibleDataset(), "BodyText", 
				"\nORDER BY DatePublished DESC\n"
				+ "LIMIT 6061",
				CREDIBLE_LABEL);
		
		this.validationData = this.getBodyContent(sqlContxt, conf.getValidationDataset(), "content",
				"", "categoryToLabel(category)");
		
		
		DataFrame[] unreliableSplits = this.getSplitsFromDF(this.getUnreliableData(), weights);
		DataFrame[] credibleSplits = this.getSplitsFromDF(this.getCredibleData(), weights);
		
		DataFrame uTrainingSplit = unreliableSplits[0].cache();
		DataFrame cTrainingSplit = credibleSplits[0].cache();
		
		DataFrame uTestingSplit = unreliableSplits[1].cache();
		DataFrame cTestingSplit = credibleSplits[1].cache();

		this.train = uTrainingSplit.unionAll(cTrainingSplit).orderBy("content").repartition(10).cache();
		this.test = uTestingSplit.unionAll(cTestingSplit).orderBy("content").repartition(10).cache();
		this.validation = validationData.orderBy("content").repartition(10).cache();
		this.bazikiLeaks = this.getBodyContent(sqlContxt, "/home/momchil/Documents/MasterThesis/dataset/bazikileaks-data-extended.json", "content",
				"", UNRELIABLE_LABEL).unionAll(cTestingSplit).orderBy("content").repartition(10).cache();
		
		uTrainingSplit.unpersist();
		cTrainingSplit.unpersist();
		cTestingSplit.unpersist();
		uTestingSplit.unpersist();
		
	}
	
	private DataFrame[] getSplitsFromDF(DataFrame df, double[] weights) {
		return df.randomSplit(weights, SEED);
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

	public DataFrame getBazikiLeaks() {
		return bazikiLeaks;
	}
}
