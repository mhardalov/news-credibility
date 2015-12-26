package com.news;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.ResultScanner;
import org.apache.hadoop.hbase.client.Scan;
import org.codehaus.jettison.json.JSONException;
import org.codehaus.jettison.json.JSONObject;

import com.news.hbase.extract.base.HbaseReaderBase;
import com.news.hbase.extract.bazikileaks.BazikileaksReader;
import com.news.hbase.extract.nenovinite.NeNoviniteReader;

public class Main {

	public static void fixNews(String file) throws FileNotFoundException, IOException, JSONException {
		List<String> lines = new ArrayList<String>();
		int skipped = 0;
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			String line;
			while ((line = br.readLine()) != null) {
				try {
					JSONObject json = new JSONObject(line);
					lines.add(json.toString());
				} catch (Exception e) {
					skipped++;
				}
			}
		}

		System.out.println(skipped);
		FileWriter writer = new FileWriter(file + "-new");
		for (String str : lines) {
			writer.write(str + "\n");
		}
		writer.close();
	}

	public static void main(String[] args) throws IOException {
		final String serverFQDN = "localhost";
		List<String> jsonStrings = new LinkedList<>();

		Configuration conf = HBaseConfiguration.create();

		conf.clear();
		conf.set("hbase.zookeeper.quorum", serverFQDN);
		conf.set("hbase.zookeeper.property.clientPort", "2181");

		try {
			HTable hTable = new HTable(conf, "webpage");

			try {

				Scan scan = new Scan();
				scan.addColumn("f".getBytes(), "cnt".getBytes());
				scan.addColumn("f".getBytes(), "bas".getBytes());
				scan.setBatch(2000);
				scan.setCaching(2000);

				ResultScanner scanner = hTable.getScanner(scan);
				Iterator<Result> resultsIter = scanner.iterator();

				int i = 1;
				while (resultsIter.hasNext()) {

					Result result = resultsIter.next();
					// HbaseReaderBase jReader = new NeNoviniteReader(result);
					HbaseReaderBase jReader = new BazikileaksReader(result);

					jsonStrings.add(jReader.toJSON());

					if (i % 100 == 0) {
						System.out.println(i);
					}
					i++;
				}

			} finally {
				if (hTable != null) {
					hTable.close();
				}
			}

		} catch (Exception ex) {
			System.out.println("Error caught.");
			ex.printStackTrace();
		}

		FileWriter writer = new FileWriter("/home/momchil/Desktop/bazikleaks/data-extended.json");
		for (String str : jsonStrings) {
			writer.write(str + "\n");
		}
		writer.close();

		System.out.println("End.");

	}

}
