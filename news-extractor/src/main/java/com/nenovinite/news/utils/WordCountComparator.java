package com.nenovinite.news.utils;

import java.io.Serializable;
import java.util.Comparator;

import scala.Tuple2;

public class WordCountComparator implements Comparator<Tuple2<String, Integer>>, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6924070143995286045L;

	@Override
	public int compare(Tuple2<String, Integer> o1, Tuple2<String, Integer> o2) {
		return o1._2.compareTo(o2._2);
	}

}
