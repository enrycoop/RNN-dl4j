package com.enricocoppolecchia.Evaluation;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class ClassBinaryBalancer {
	public ClassBinaryBalancer(String path, String filename,int column) throws IOException {
		Scanner sc = new Scanner(new File(path + "/" + filename));
		Map<String, Integer> nominal = new HashMap<String, Integer>();
		List<String> lines = new LinkedList<String>();
		while (sc.hasNext()) {
			String line = sc.nextLine();
			String nom = line.split(",")[column];
			if(nominal.containsKey(nom)) {
				nominal.put(nom, nominal.get(nom)+1);
			}
			else {
				nominal.put(nom, 1);
			}
			System.out.println(nom+":"+nominal.get(nom));
			lines.add(line);
		}
		sc.close();
		String nom="";
		int min = Integer.MAX_VALUE;
		for(String n : nominal.keySet()) {
			if(nominal.get(n).compareTo(min)<0) {
				min=nominal.get(n);
				nom = n;
			}
			System.out.println(n+":"+nominal.get(n));

		}
		System.out.println(nom+": "+min);
		PrintWriter testWriter = new PrintWriter(new FileWriter(path + "/" + "balanced.csv"));
		for (String line : lines) {
			String label = line.split(",")[column];
			if(!label.equals(nom) && nominal.get(label)>min) {
				nominal.put(label, nominal.get(label)-1);
			}else {
				testWriter.write(line+"\n");
			}
		}
		testWriter.close();
	}
	public static void main(String[] args) throws IOException {
		new ClassBinaryBalancer("resources/card_type", "test.csv",2);
	}
}
