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

public class CrossValidationSplit {
	public CrossValidationSplit(String path, String filename,int column,double split) throws IOException {
		Scanner sc = new Scanner(new File(path + "/" + filename));
		Map<String, Integer> nominal = new HashMap<String, Integer>(); //dizionario classe-frequenza
		List<String> lines = new LinkedList<String>(); //copia delle righe originali
		while (sc.hasNext()) {
			String line = sc.nextLine();
			String nom = line.split(",")[column]; //prelevo l'etichetta
			if(nominal.containsKey(nom)) {
				nominal.put(nom, nominal.get(nom)+1);
			}
			else {										// aggiornamento dizionario
				nominal.put(nom, 1);
			}  
			lines.add(line);
		}
		sc.close();
	
		
		for(String k:nominal.keySet()) {
			System.out.println(k+" before "+nominal.get(k));
		}
			
		
		// aggiornamento frequenza massima per ogni label
		for(String k:nominal.keySet()) 
			nominal.put(k, (int) Math.round(nominal.get(k)*split));
	
		for(String k:nominal.keySet()) {
			System.out.println(k+" after "+nominal.get(k));
		}
		
		List<String> toRemove = new LinkedList<String>();
		int length = lines.size();
		PrintWriter testWriter = new PrintWriter(new FileWriter(path + "/" + "test.csv"));
		for(String k: nominal.keySet()) {
			for(int i=0;i<lines.size();i++) {
				if(nominal.get(k)<=0)
					continue;
				if(lines.get(i).split(",")[column].equals(k)) {
					testWriter.write(lines.get(i)+"\n");
					toRemove.add(lines.get(i));
					nominal.put(k, nominal.get(k)-1);
				}
			}
			System.out.println("size before: "+lines.size());
			lines.removeAll(toRemove);
			System.out.println("size after: "+lines.size());
		}
		System.out.println("Train size: "+lines.size());
		testWriter.close();
		PrintWriter trainWriter = new PrintWriter(new FileWriter(path + "/" + "train.csv"));
		for(String line:lines)
			trainWriter.write(line+"\n");
		trainWriter.close();
		System.out.println("Test size: "+(length-lines.size()));
	}
	public static void main(String[] args) throws IOException {
		new CrossValidationSplit("resources/card_type", "card.csv",2,0.3);
	}
}
