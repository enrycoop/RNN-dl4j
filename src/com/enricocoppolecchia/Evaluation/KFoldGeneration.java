package com.enricocoppolecchia.Evaluation;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class KFoldGeneration {
	int k;
	String path;
	String filename;
	public KFoldGeneration(int k, String path, String filename) {
		this.filename = filename;
		this.path = path;
		this.k= k;
	}
	public void generateFolds() throws Exception{
		//inizializzo scanner
		Scanner sc = new Scanner(new File(path+"/"+filename));
		List<String> lines = new LinkedList<String>();
		
		//acquisisco esempi
		while(sc.hasNext()) {
			lines.add(sc.nextLine());
		}
		Collections.shuffle(lines);
		sc.close();
		int step = lines.size()/k;
		for(int i=0;i<k;i++) {
			PrintWriter testWriter = new PrintWriter(new FileWriter(path+"/test"+i+".csv"));
			PrintWriter trainWriter = new PrintWriter(new FileWriter(path+"/train"+i+".csv"));
			
			for(int j = 0;j<lines.size();j++) {
				if((j>(i*step))&&(j<(step*(i+1)))) {
					testWriter.println(lines.get(j));
				}
				else
					trainWriter.println(lines.get(j));
			}
			
			testWriter.close();
			trainWriter.close();
		}
	}
	public static void main(String[] args) {
	
		try {
			new KFoldGeneration(10, "resources/card_type", "credit.csv").generateFolds();
		} catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
}
