package com.nenovinite.news.utils;

import java.io.Serializable;
import java.util.Comparator;

import scala.Tuple2;

public class WordIndexComparator implements Comparator<Tuple2<String,Tuple2<Integer,Long>>>, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 6924070143995286045L;

	@Override
	public int compare(Tuple2<String, Tuple2<Integer, Long>> o1, Tuple2<String, Tuple2<Integer, Long>> o2) {
		
		return o1._2._1.compareTo(o2._2._1);
	}
}
