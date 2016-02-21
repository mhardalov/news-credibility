package com.nenovinite.news;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.lang.math.NumberUtils;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.spark.ml.UnaryTransformer;
import org.apache.spark.sql.types.DataType;
import org.apache.spark.sql.types.DataTypes;

import scala.Function1;

public class LuceneTokenizer extends UnaryTransformer<String, List<String>, LuceneTokenizer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1073376553624344597L;

	public final static Analyzer analyzer = new StandardAnalyzer();

	@Override
	public String uid() {
		return String.valueOf(serialVersionUID);
	}

	@Override
	public Function1<String, List<String>> createTransformFunc() {
		return new JavaFunction1<String, List<String>>() {

			/**
			 * 
			 */
			private static final long serialVersionUID = 7097200031942576300L;

			@Override
			public List<String> apply(String s) {
				List<String> wordsList = new ArrayList<>();

				try {
					String text = s.replace("\n", " ").replace("\r", " ").replace("br2n", " ");
					try (TokenStream stream = analyzer.tokenStream("words", new StringReader(text))) {
						stream.reset();
						while (stream.incrementToken()) {
							String word = stream.addAttribute(CharTermAttribute.class).toString();

							if (word.length() < 4 || NumberUtils.isNumber(word)) {
								continue;
							}

							wordsList.add(word);
						}
					}
				} catch (IOException e) {
					// not thrown b/c we're using a string reader...
					throw new RuntimeException(e);
				}

				return wordsList;
			}
		};
	}

	@Override
	public DataType outputDataType() {
		return DataTypes.createArrayType(DataTypes.StringType);
	}

}
