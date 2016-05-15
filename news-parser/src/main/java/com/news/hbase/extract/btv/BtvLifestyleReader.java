package com.news.hbase.extract.btv;

import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;

import org.apache.hadoop.hbase.client.Result;

import com.news.hbase.extract.base.HbaseReaderBase;

public class BtvLifestyleReader extends HbaseReaderBase {
	private static final SimpleDateFormat FORMATTER = new SimpleDateFormat("dd.MM.yyyy HH:mm");

	public BtvLifestyleReader(Result result) throws MalformedURLException, UnsupportedEncodingException, ParseException {
		super(result);

		String dateStr = doc.select("#box_10020914d10495 > .article_date").text();
		publishDate = FORMATTER.parse(dateStr);
		viewCount = Long.valueOf(doc.select("#box_10020907d10500 > .article_views > span").text());
		title = doc.select("#box_10020912d10496 > .article_title").text();
		content = doc.select("#box_10020915d10503 > .article_body").text();
		category = doc.select("#box_10020908d10494").text();
		commentCount = 0l;//Long.valueOf(doc.select("#box_10020918d10535 #commentsAnchor a.tabs.tab-btv.active").text().replaceAll("Коментари \\((\\d+)\\)", "$1")); 
	}

}
