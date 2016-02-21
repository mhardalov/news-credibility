package com.nenovinite.news;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.ml.util.Identifiable$;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Function1;
import scala.collection.mutable.WrappedArray;

public class TokenFeaturesExtractor extends Transformer implements HasInputCol, HasOutputCol, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1073376553624344597L;
	
	private static final String HAS_UPPER_CASE = "^.*?[А-ЯA-Z].*?$";
	private static final String ALL_UPPER_CASE = "^[А-ЯA-Z]+$";
	private static final String FIRST_UPPER_CASE =  "^[А-ЯA-Z].*$";
	private static final String HAS_EXCL_MARK = "^.*[!?].*$";
	private static final String HAS_QUOTES = "^.*\".*$";
	private static final String HAS_COMMA = "^.*[,;].*$";

	private String uid_ = Identifiable$.MODULE$.randomUID(this.getClass().toString().toLowerCase());

	private Param<String> inputCol = new Param<String>(this, "content", "input column name");
	private Param<String> outputCol = new Param<String>(this, "tokens", "output column name");

	
	public TokenFeaturesExtractor() {}
	
	public TokenFeaturesExtractor(String uid) {
		this.uid_ = uid;
	}

	@Override
	public String uid() {
		return uid_;
	}

	public Function1<WrappedArray<String>, scala.collection.immutable.List<Double>> createTransformFunc() {
		return new JavaFunction1<WrappedArray<String>, scala.collection.immutable.List<Double>>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;

			@Override
			public scala.collection.immutable.List<Double> apply(WrappedArray<String> tokens) {
				
				List<String> tokensList = new LinkedList<>(scala.collection.JavaConversions.asJavaList(tokens));
				double tokensCount = (double) tokensList.size();
				double upperCaseCount = 0;
				for (String token : tokensList) {
					if (token.matches(HAS_UPPER_CASE)) {
						upperCaseCount++;
					};
				}
				
				List<Double> features = new LinkedList<>();
				features.add(tokensCount);
				features.add(upperCaseCount/(tokensCount+1));				

				return scala.collection.JavaConversions.asScalaBuffer(features).toList();
			}
		};
	}

	public DataType outputDataType() {
		return DataTypes.createArrayType(DataTypes.StringType, true);
	}

	@Override
	public String getOutputCol() {
		return this.outputCol().name();
	}
	
	public TokenFeaturesExtractor setOutputCol(String value) {
		this.outputCol = new Param<>(this, value, "output column name");
		return this;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasOutputCol$_setter_$outputCol_$eq(Param arg0) {

	}

	@Override
	public Param<String> outputCol() {
		return this.outputCol;
	}

	@Override
	public String getInputCol() {
		return this.inputCol.name();
	}

	@Override
	public Param<String> inputCol() {
		return inputCol;
	}
	
	public TokenFeaturesExtractor setInputCol(String value) {
		this.inputCol = new Param<>(this, value, "input column name");
		return this;
	}

	@Override
	public void org$apache$spark$ml$param$shared$HasInputCol$_setter_$inputCol_$eq(Param arg0) {

	}	

	public DataFrame transform(DataFrame dataset) {
		transformSchema(dataset.schema());
		
		dataset = dataset.withColumn(this.getOutputCol(), 
				org.apache.spark.sql.functions.callUDF(this.createTransformFunc(), this.outputDataType(),
				dataset.col(this.getInputCol())));

		return dataset;
	}

	@Override
	public StructType transformSchema(StructType schema) {
		StructType outputFields = schema
				.add(new StructField(this.getOutputCol(), DataTypes.createArrayType(DataTypes.StringType), true,  Metadata.empty()));
		return outputFields;
	}
	
	@Override
	public Transformer copy(ParamMap extra) {
		// TODO Auto-generated method stub
		return defaultCopy(extra);
	}

}
