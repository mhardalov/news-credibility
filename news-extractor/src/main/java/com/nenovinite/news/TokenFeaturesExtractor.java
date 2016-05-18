package com.nenovinite.news;

import java.io.IOException;
import java.io.Serializable;
import java.io.StringReader;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.core.LetterTokenizer;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.miscellaneous.LengthFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Attribute;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.shared.HasInputCol;
import org.apache.spark.ml.param.shared.HasOutputCol;
import org.apache.spark.ml.util.Identifiable$;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import com.nenovinite.news.utils.GazetteerContainer;

import scala.Function1;

public class TokenFeaturesExtractor extends Transformer implements HasInputCol, HasOutputCol, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1073376553624344597L;
	
	private static final String HAS_UPPER_CASE = "^.*?[А-ЯA-Z].*?$";
	private static final String ALL_UPPER_CASE = "^[А-ЯA-Z]+$";
	private static final String FIRST_UPPER_CASE =  "^[А-ЯA-Z].*$";
	private static final String LOWER_CASE = "^[а-яa-z]+$";
	private static final String HAS_URL = "://";
	
	private String uid_ = Identifiable$.MODULE$.randomUID(this.getClass().toString().toLowerCase());

	private Param<String> inputCol = new Param<String>(this, "content", "input column name");
	private Param<String> outputCol = new Param<String>(this, "tokens", "output column name");

	private VectorUDT outputDataType = new VectorUDT();

	
	public TokenFeaturesExtractor() {}
	
	public TokenFeaturesExtractor(String uid) {
		this.uid_ = uid;
	}

	@Override
	public String uid() {
		return uid_;
	}

	public Function1<String, Vector> createTransformFunc() {
		return new JavaFunction1<String, Vector>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 1L;
			
			private double countOccurences(String content, String substring) {
				double originalLen = content.length();
				double leftOverLength = content.replace(substring, "").length();
				double subStrLen = substring.length();
				
				return (originalLen - leftOverLength) / subStrLen;
			}

			@Override
			public Vector apply(String content) {
				int featuresCount = 12;
				
				double upperCaseCount = 0.0;
				double allUpperCaseCount = 0.0;
				double firstUpperCase = 0.0;
				double lowerUpperCase = 0.0;
				double hasUrl = 0.0;
				double firstPersonPronouns = 0.0;
				double thirdPersonPronouns = 0.0;
				
//				Multiset<String> tokens =
//					    ConcurrentHashMultiset.create(new LinkedList<String>());
				
				List<String> tokens = new LinkedList<>();
			    StringReader input = new StringReader(content);
				LetterTokenizer tokenizer = new LetterTokenizer();
				try {
					tokenizer.setReader(input);

					TokenFilter stopFilter = new StopFilter(tokenizer, GazetteerContainer.BULGARIAN_STOP_WORDS_SET);
					TokenFilter length = new LengthFilter(stopFilter, 1, 1000);
	//				TokenFilter stemmer = new BulgarianStemFilter(length);
	//				TokenFilter ngrams = new ShingleFilter(stemmer, 2, 2);
	
					try (TokenFilter filter = length) {
	
						Attribute termAtt = filter.addAttribute(CharTermAttribute.class);
						filter.reset();
						while (filter.incrementToken()) {
							String token = termAtt.toString().replaceAll(",", "\\.").replaceAll("\n|\r", "");
							if (token.matches(HAS_UPPER_CASE)) {
								upperCaseCount++;
							};
							
							if (token.matches(ALL_UPPER_CASE)) {
								allUpperCaseCount++;
							}
							
							if (token.matches(FIRST_UPPER_CASE)) {
								firstUpperCase++;
							}
							
							if (token.matches(LOWER_CASE)) {
								lowerUpperCase++;
							}
							
							if (token.matches(HAS_URL)) {
								hasUrl++;
							}
							
							token = token.toLowerCase();
							if (GazetteerContainer.FIRST_PERSON.contains(token)) {
								firstPersonPronouns++;
							}
							
							if (GazetteerContainer.THIRD_PERSON.contains(token)) {
								thirdPersonPronouns++;
							}
							
							tokens.add(token);
						}
					}
					
				} catch (IOException e) {
					e.printStackTrace();
					return Vectors.zeros(featuresCount);
				}
				
				double tokensCount = Math.max(1.0, (double) tokens.size());
				
				List<Double> features = new LinkedList<>();
				features.add(tokensCount);
				features.add(upperCaseCount/tokensCount);
				features.add(allUpperCaseCount/tokensCount);
				features.add(firstUpperCase/tokensCount);
				features.add(lowerUpperCase/tokensCount);
				features.add(hasUrl/tokensCount);
				features.add(firstPersonPronouns/tokensCount);
				features.add(thirdPersonPronouns/tokensCount);
				features.add(this.countOccurences(content, "!")/tokensCount);
				features.add(this.countOccurences(content, "#")/tokensCount);
				features.add(this.countOccurences(content, "\"")/tokensCount);
				features.add(this.countOccurences(content, "'")/tokensCount);
				features.add(this.countOccurences(content, "?")/tokensCount);
				double[] primitiveFeatures = ArrayUtils.toPrimitive(features.toArray(new Double[features.size()]));

				Vector vector = Vectors.dense(primitiveFeatures);
				return vector;
			}
		};
	}

	public DataType outputDataType() {
		return outputDataType;
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
				.add(new StructField(this.getOutputCol(), this.outputDataType(), true,  Metadata.empty()));
		return outputFields;
	}
	
	@Override
	public Transformer copy(ParamMap extra) {
		// TODO Auto-generated method stub
		return defaultCopy(extra);
	}

}
