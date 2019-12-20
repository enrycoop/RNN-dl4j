package com.enricocoppolecchia.Evaluation;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class Doubler {
	public Doubler(String path,String filename) throws IOException {
		Scanner sc = new Scanner(new File(path+"/"+filename));
		List<String> newLines =  new LinkedList<String>();
		while(sc.hasNext()) {
			String line =  sc.nextLine();
			String newLine = "";
			for(String number:line.split(",")) {
				if(!number.contains("."))
					newLine+=number+".0"+",";
				else
					newLine+=number+",";
			}
			newLines.add(newLine.substring(0,newLine.length()-1));
		}
		sc.close();
		PrintWriter testWriter = new PrintWriter(new FileWriter(path+"/"+"c.csv"));
		for(String line:newLines) {
			
			testWriter.write(line+"\n");
			
		}
		testWriter.close();	
	}
	public static void main(String[] args) throws IOException {
		new Doubler("resources/card_type","card.csv");
	}
}
