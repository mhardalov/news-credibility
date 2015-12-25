package com.news.hbase.extract.bazikileaks;

import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;

import org.apache.hadoop.hbase.client.Result;
import org.jsoup.nodes.Element;

import com.news.hbase.extract.base.HbaseReaderBase;

public class BazikileaksReader extends HbaseReaderBase {
	private static final SimpleDateFormat FORMATTER = new SimpleDateFormat("dd MMM yyy");
	private boolean isNewsArticle() {
		return this.url.toString().matches("https:\\/\\/neverojatno.wordpress.com\\/\\d+\\/\\d+\\/\\d+\\/[A-Za-zА-Яa-я0-9\\.-]+\\/$");
	}

	public BazikileaksReader(Result result) throws MalformedURLException, UnsupportedEncodingException, ParseException {
		super(result);
		content = doc.select(".post-entry").text();
		if (!isNewsArticle() || !hasContent()) {
			System.out.println("Skipping " + this.url);
			content = "";
			return;
		}
		try {
		String dateStr = doc.select(".post-title .post-date > strong").text() + " " + doc.select(".post-title .post-date > span").text();
		publishDate = FORMATTER.parse(dateStr);
		} catch (Exception e){
			publishDate = null;
		}
		viewCount = 0l;
		title = doc.select(".post-title > h1").text();
		
		category = "";
		for (Element element : doc.select(".post-info.clear-fix > p > a")) {
			if (element.toString().contains("category tag")) {
			category += element.text() + ",";
			}
		}
		category = !category.isEmpty() ? category.substring(0, category.length() - 1) : "";

		// TODO: default value
		String commentCnt = doc.select("h3#comments-title > span").text();
		commentCount = !commentCnt.isEmpty() ? Long.valueOf(commentCnt) : 0l;
	}

}
