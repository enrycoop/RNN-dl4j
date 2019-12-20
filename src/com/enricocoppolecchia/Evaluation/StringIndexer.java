package com.enricocoppolecchia.Evaluation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class StringIndexer {
	
	public StringIndexer(String path,String filename,int column) throws IOException {
		Scanner sc = new Scanner(new File(path+"/"+filename));
		List<String> nominal =  new LinkedList<String>();
		List<String> lines = new LinkedList<String>();
		while(sc.hasNext()) {
			String line =  sc.nextLine();
			
			String nom = line.split(",")[column];
			System.out.println(nom);
			if(!nominal.contains(nom))
				nominal.add(nom);
			lines.add(line);
		}
		sc.close();
		PrintWriter testWriter = new PrintWriter(new FileWriter(path+"/"+"card.csv"));
		for(String line:lines) {
			String row = "";
			String[] rows = line.split(",");
			rows[column]=nominal.indexOf(rows[column])+".0";
			for(String s:rows)
				row+=s+",";
			testWriter.write(row.substring(0,row.length()-1)+"\n");
			
		}
		testWriter.close();	
	}
	public static void main(String[] args) throws IOException {
		new StringIndexer("resources/card_type","c.csv",7);
	}
}
