package com.news.hbase.extract.nenovinite;

import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.text.ParseException;
import java.text.SimpleDateFormat;

import org.apache.hadoop.hbase.client.Result;

import com.news.hbase.extract.base.HbaseReaderBase;

public class NeNoviniteReader extends HbaseReaderBase {
	private static final SimpleDateFormat FORMATTER = new SimpleDateFormat("dd.MM.yyyy HH:mm");

	public NeNoviniteReader(Result result) throws ParseException, MalformedURLException, UnsupportedEncodingException {
		super(result);

		String dateField = doc.select(".newsBoxN > .date").text().replace("прегледана ", "").replace(" пъти", "");
		String[] splits = dateField.split("\\s+");

		String dateStr = splits[0] + " " + splits[1];
		publishDate = FORMATTER.parse(dateStr);
		viewCount = Long.valueOf(splits[2]);
		title = doc.select(".newsBoxN > h1").text();
		content = doc.select(".newsTextt").text();
		category = doc.select(".caTTL > h2").text();
		//TODO: default value
		commentCount = 0l; 
	}

}
