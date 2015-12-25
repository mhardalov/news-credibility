package com.news.hbase.extract.base;

import java.io.UnsupportedEncodingException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.Date;

import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.util.Bytes;
import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;

public abstract class HbaseReaderBase {
	protected final Document doc;
	protected final String key;
	protected final URL url;
	protected final String pureHtml;
	protected String title;
	protected String content;
	protected Date publishDate;
	protected Long commentCount;
	protected Long viewCount;
	protected String category;

	public HbaseReaderBase(Result result) throws MalformedURLException, UnsupportedEncodingException {
		key = Bytes.toString(result.getRow());
		String urlString = Bytes.toString(result.getValue("f".getBytes(), "bas".getBytes()));
		url = new URL(urlString);
		byte[] html = result.getValue("f".getBytes(), "cnt".getBytes());

		pureHtml = new String(html, "UTF-8");
		// pureHtml = pureHtml.replaceAll("(?i)<br[^>]*>", "br2n");

		doc = Jsoup.parse(pureHtml);

	}

	public String getTitle() {
		return title;
	}

	public String getContent() {
		return content;
	}

	public String getCategory() {
		return category;
	}

	public Date getPublishDate() {
		return publishDate;
	}

	public Long getViewCount() {
		return viewCount;
	}

	public Long getCommentCount() {
		return commentCount;
	}

	public boolean hasContent() {
		return !content.isEmpty();
	}

	public String toJSON() throws JSONException {
		JSONObject json = new JSONObject();
		json.put("key", key);
		json.put("url", url);
		json.put("html", pureHtml);
		json.put("title", getTitle());
		json.put("content", getContent());
		json.put("category", getCategory());
		json.put("publishDate", getCategory());
		json.put("viewCount", getViewCount());
		json.put("commentCount", getCommentCount());

		return json.toString();
	}
}
