package com.nenovinite.news.features.filters;

import java.io.IOException;

import org.apache.commons.lang3.math.NumberUtils;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.FilteringTokenFilter;

public class NumberFilter extends FilteringTokenFilter {
	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);

	/**
	 * Create a new NumberFilter, that removes numbers from tokens
	 * 
	 * @param in
	 *            TokenStream to filter
	 */
	public NumberFilter(TokenStream in) {
		super(in);
	}

	@Override
	protected boolean accept() throws IOException {
		return !NumberUtils.isNumber(termAtt.toString().replace(",", "."));
	}

}
