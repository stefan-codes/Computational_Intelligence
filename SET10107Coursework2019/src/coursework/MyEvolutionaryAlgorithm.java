package coursework;

import java.util.ArrayList;
import java.util.Arrays;

import org.apache.commons.math3.*;
import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.stat.descriptive.moment.Mean;
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation;

import model.Fitness;
import model.Individual;
import model.NeuralNetwork;


/**
 * @author Stefan Hristov
 * My implementation of an Evolutionary Algorithm for exploring and evolving
 * the weights for the neural network.
 */
public class MyEvolutionaryAlgorithm extends NeuralNetwork {

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {
		// Initialize a random population
		population = initialise();
		
		// Record a copy of the best Individual in the population
		best = setBest();
		System.out.println("Best From Initialisation " + best);
		
		System.out.println(Parameters.getNumGenes() + "ee");
				
		// Main Evaluation loop
		while(evaluations < Parameters.maxEvaluations) {
		
			// Selection
			ArrayList<Individual> parents = tournamentSelection(Parameters.tournamentSize, Parameters.numberOfParents);
			
			// Crossover
			ArrayList<Individual> children = uniformCrossOver(parents, Parameters.numberOfChildrenBorn, Parameters.numberOfChildrenSurvive);
			
			// Mutation
			bestFitnessMutation(children);
			
			// Evaluate the children - set their fitness
			evaluateIndividuals(children);
			
			// Replacement
			replace(children);
			
			// check to see if the best has improved
			best = setBest();
			
			printMeanFitness();
			// Implemented in NN class. 
			outputStats();
			
			if (best.fitness < 0.01) {
				//terminate earlier
				evaluations = 20000;
				System.out.println("Early stopping!");
			}
			
		}
		
		// Save the trained network to disk
		saveNeuralNetwork();
		
	}

	/**
	 * Replace the least fit person with a child if the child is fitter.
	 * @param children the list of individuals to be used in replacement
	 */
	private void replace(ArrayList<Individual> children) {
		// For each child
		for (Individual child : children) {
			// If they are the better than the worst in the population
			if (child.fitness < getWorst(population).fitness) {
				population.set(population.indexOf(getWorst(population)), child);
			} 
			
			// If its better than the best, reset our counter
			if (child.fitness < best.fitness) {
				Parameters.fitnesslastUpdateIndex = 0;
			} else {
				Parameters.fitnesslastUpdateIndex++;
			}
		}
	}
	
	/**
	 * Mutation which for each gene applies a random positive or negative mutateChange with mutateRate chance.
	 * @param individuals the children passed to be mutated.
	 */
	@SuppressWarnings("unused")
	private void constantMutate(ArrayList<Individual> individuals) {
		// For each individual
		for (Individual individual : individuals) {
			// For each genome in the chromosome
			for (int i = 0; i < individual.chromosome.length; i++) {
				if (Parameters.random.nextDouble() < Parameters.mutateRate) {
					if (Parameters.random.nextBoolean()) {
						//individual.chromosome[i] += (Parameters.mutateChange);
						individual.chromosome[i] += individual.fitness;
					} else {
						individual.chromosome[i] -= individual.fitness;
					}
				}
			}
		}
	}
	
	/**
	 * 
	 */
	private void bestFitnessMutation(ArrayList<Individual> individuals) {
		
		Parameters.mutateRate = Parameters.fitnesslastUpdateIndex * 0.01;
		
		// Make sure mutationRate doesn't get lower than 0.05 and higher than 1
		if (Parameters.mutateRate < 0.05) {
			Parameters.mutateRate = 0.05;
		} else if (Parameters.mutateRate > 1) {
			Parameters.mutateRate = 1;
		}
		
		// For each individual
				for (Individual individual : individuals) {
					// Standard Deviation
					double stDev = best.fitness * 5;
					// Normal Distribution
					NormalDistribution nDist = new NormalDistribution(0, stDev);
					
					// For each gene in the chromosome
					for (int i = 0; i < individual.chromosome.length; i++) {
						// Sample the distribution
						double nDistSample = nDist.sample();
						
						if (Parameters.random.nextDouble() < Parameters.mutateRate) {
							individual.chromosome[i] += nDistSample;
							
//							if (Parameters.random.nextBoolean()) {
//								individual.chromosome[i] += 0.5D;
//							} else {
//								individual.chromosome[i] -= 0.5D;
//							}
						}
					}
				}
	}
	
	/**
	 * TODO: Write the comments
	 * @param individuals
	 */
	private void fitnessFrequencyDistributionMutate(ArrayList<Individual> individuals) {
		// The frequency value of a best fitness value BFF
		double bff = calculateBff();
		Parameters.bffs.add(bff);
		
		// Standard Deviation
		StandardDeviation sDev = new StandardDeviation();
		
		double[] popFitness = new double[population.size()];
		for (int i = 0; i < population.size(); i++) {
			popFitness[i] = population.get(i).fitness;
		}
		
		double standardDeviation = sDev.evaluate(popFitness);
		if (standardDeviation < 0.0001) {
			standardDeviation = 0.0001;
		}
		
		// Normal Distribution
		NormalDistribution nDist = new NormalDistribution(0, standardDeviation);
		double nDistSample = nDist.sample();
		System.out.println("nDistSample is: "+ nDistSample);
		
		// Mutation Rate (Probability) every 50 generations
		if (Parameters.bffs.size() > Parameters.lGenerations) {
			
			// Get the mean BFF
			double meanBff = 0.0;
			for (double d : Parameters.bffs) {
				meanBff += d;
			}
			meanBff = meanBff/Parameters.bffs.size();
			
			double mu = 3*standardDeviation*(meanBff/Parameters.linkingCoefficient);
			Parameters.mutationProbability = Parameters.mutateRateConst + mu;
			
			Parameters.bffs.clear();
		}
				
		// For each individual
		for (Individual individual : individuals) {
			// For each gene in the chromosome
			for (@SuppressWarnings("unused") double gene : individual.chromosome) {
				if (Parameters.random.nextDouble() < Parameters.mutationProbability) {
					if (Parameters.random.nextBoolean()) {
						gene += (nDistSample);
					} else {
						gene -= (nDistSample);
					}
				}	
			}
		}
		
		
		
		System.out.println("Mutation Rate is: " + Parameters.mutationProbability);
	}
	
	/**
	 * Calculate the frequency value of the best fitness value Range between 1-0
	 * @return
	 */
	private double calculateBff() {
		double bff = 0.0; // The frequency value of a best fitness value 0-1
		double bestFitness = getBest(population).fitness;
		double bandwithInterval = bestFitness * Parameters.bandWidthRange;
		
		// Number of individuals in a specific fitness bracket
		double ni = 0.0;
		
		// For all the individuals in the population
		for(Individual individual : population) {
			// if the ti is within tb + e
			if (individual.fitness < bestFitness + bandwithInterval) {
				ni++;
			}
		}
		
		bff = ni/Parameters.popSize;
		
		return bff;
	}
	
	/**
	 * Crossover where each node (hidden and output) have their weights and bias together.
	 * @param parents the individuals for the gene recombination
	 * @param chBorn children produced
	 * @param chSurvive children survived and replaced in population
	 * @return a list of children to use for replacement. 
	 */
	@SuppressWarnings("unused")
	private ArrayList<Individual> customCrossOver(ArrayList<Individual> parents, int chBorn, int chSurvive){
		ArrayList<Individual> children = new ArrayList<>();
		ArrayList<Integer> indices = new ArrayList<>();
		int randomParentID = 0;
		Individual parent = null;
		double[] chromosome = new double[Parameters.getNumGenes()];
		
		// For the children born
		for (int i = 0; i < chBorn; i ++) {
			
			// For each hidden neuron
			for (int j = 0; j < Parameters.getNumHidden(); j++) {
				// Get the indices for each neuron
				indices = getHiddenNeuronWeightsAndBias(j);
				// Select random parent
				randomParentID = Parameters.random.nextInt(parents.size());
				parent = parents.get(randomParentID);
				
				for (Integer a : indices) {
					chromosome[a] = parent.chromosome[a];
				}	
			}
			
			// For each output
			for (int j = 0; j < Parameters.numOutputs; j++) {
				// Get the indices for each output
				indices = getOutputWeightsAndBias(j);
				// Select random parent
				randomParentID = Parameters.random.nextInt(parents.size());
				parent = parents.get(randomParentID);
				
				for (Integer a : indices) {
					chromosome[a] = parent.chromosome[a];
				}	
			}
			
			Individual child = new Individual();
			child.chromosome = chromosome;
			// I allow twins, so no need to check for the same chromosome in the children.
			children.add(child);
		}
		
		// If children to survive is lower than born
		if(chSurvive < chBorn) {
			// TODO: implement competition.
		}
				
				
				
				
		return children;
	}
	
	/**
	 * Perform a uniform cross over on the the parents. It is random for each genome. 
	 * @param parents the list of Individuals used as parents
	 * @param chBorn the children's chromosomes to be produced
	 * @param chSurvive the number of chromosomes to be kept "survive"
	 * @return a list of individuals - the new children
	 */
	private ArrayList<Individual> uniformCrossOver(ArrayList<Individual> parents, int chBorn, int chSurvive){
		ArrayList<Individual> children = new ArrayList<>();
		int randomParentID = 0;
		double[] chromosome = new double[Parameters.getNumGenes()];
		
		// For the children born
		for (int i = 0; i < chBorn; i ++) {
			// For the number of genes in a chromosome
			for (int j = 0; j < Parameters.getNumGenes(); j++) {
				// Select random parent
				randomParentID = Parameters.random.nextInt(parents.size());
				chromosome[j] = parents.get(randomParentID).chromosome[j];
			}
			
			Individual child = new Individual();
			child.chromosome = chromosome;
			// I allow twins, so no need to check for the same chromosome in the children.
			children.add(child);
		}
		
		// If children to survive is lower than born
		if(chSurvive < chBorn) {
			// TODO: implement competition.
		}
		
		return children;
	}
	
	/**
	 * Randomly selects parent candidates and picks the fittest of them.
	 * Repeat nParents times.
	 * @param tSize = the size of the tournament
	 * @param nParents = number of parents to be returned.
	 * @return the selected parents
	 */
	private ArrayList<Individual> tournamentSelection(int tSize, int nParents){
		ArrayList<Individual> parents = new ArrayList<>();
		ArrayList<Individual> tournamentGroup = new ArrayList<>();
		Individual fittest = null;
		
		// While we don't have enough parents
		while(parents.size() < nParents) {
			// While we don't have enough individuals in the tournament group
			while(tournamentGroup.size() < tSize) {
				// Get a random individual from the population
				Individual randomIndividual = population.get(Parameters.random.nextInt(population.size()));
				// Make sure we don't have it selected already
				if(!tournamentGroup.contains(randomIndividual)) {
					tournamentGroup.add(randomIndividual);
				}
			}
			
			// Get the fittest person from the tournament group
			for (Individual i : tournamentGroup) {
				if(fittest == null) {
					fittest = i;
				} else if (i.fitness < fittest.fitness) {
					fittest = i;
				}
			}
			
			// Make sure we haven't selected this individual as a parent
			if(!parents.contains(fittest)) {
				parents.add(fittest);
			}
			
			// Reset your fittest and tournamentGroup
			fittest = null;
			tournamentGroup.clear();
			}
		
		
		return parents;
	}
	
	/**
	 * Uses rank position to determine the chance of selection
	 * @param tSize
	 * @param nParents
	 * @return
	 */
	@SuppressWarnings("unused")
	private ArrayList<Individual> rankedBasedSelection(int tSize, int nParents) {
		ArrayList<Individual> parents = new ArrayList<>();
			// TODO: need an implementation
		return parents;
	}
	
	/**
	 * Uses fitness proportioned chance of selection 
	 * @param tSize
	 * @param nParents
	 * @return
	 */
	@SuppressWarnings("unused")
	private ArrayList<Individual> rouletteWheelSelection(int tSize, int nParents) {
		ArrayList<Individual> parents = new ArrayList<>();
		double totalFitness = 0.0;
		double rouletteValue = 0.0;
	
		// get the sum of the total fitness ( 1/fitness )
		for (Individual individual : population) {
			totalFitness += 1/individual.fitness;
		}
				
		while(parents.size() < 2) {
			int index = 0;
			// get a random value between 0 and the totalFitness
			rouletteValue = Parameters.random.nextDouble() * totalFitness;
			
			// start subtracting the inverse fitness from the rouletteValue
			while (rouletteValue >= 0) {
				rouletteValue -= 1/population.get(index).fitness; 
				index++;
			}
			
			index--;
			// make sure we have not selected this individual as a parent already
			if (!parents.contains(population.get(index))) {
				parents.add(population.get(index));
			}
		}
		
		return parents;
	}
	
	// Helping method for my custom cross over (node related)
	private ArrayList<Integer> getHiddenNeuronWeightsAndBias(int neuron){
		ArrayList<Integer> indices = new ArrayList<>();
		int index = 0;
		
		// Get all input indices
		for (int i = 0; i < Parameters.numInputs; i++) {
			index = neuron*Parameters.getNumHidden() + i;
			indices.add(index);
		}
		
		// Get the bias
		index = Parameters.numInputs*Parameters.getNumHidden() + neuron;
		indices.add(index);
		
		return indices;
	}
	
	// Helping method for my custom cross over (node related)
	private ArrayList<Integer> getOutputWeightsAndBias(int output){
		ArrayList<Integer> indices = new ArrayList<>();
		int index = 0;
		
		// Get all input indices
		for (int i = 0; i < Parameters.getNumHidden(); i++) {
			index = Parameters.getNumHidden() * Parameters.numInputs + Parameters.getNumHidden() + output*Parameters.getNumHidden() + i;
			indices.add(index);
		}
		
		// Get the bias
		index = Parameters.getNumHidden() * Parameters.numInputs + Parameters.getNumHidden() + Parameters.getNumHidden()*Parameters.numOutputs + output;
		indices.add(index);
		
		return indices;
	}
	
	/**
	 * Find the individual with the highest fitness
	 * @param individuals the list to be looked through
	 * @return the worst individual
	 */
	private Individual getWorst(ArrayList<Individual> individuals) {
		Individual worst = null;
		for (Individual individual : individuals) {
			if (worst == null) {
				worst = individual;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
			}
		}
		return worst;
	}
	
	/**
	 * Find the individual with the lowest fitness
	 * @param individuals the list to be looked through
	 * @return the best individual
	 */
	private Individual getBest(ArrayList<Individual> individuals) {
		Individual best = null;
		for (Individual individual : individuals) {
			if (best == null) {
				best = individual;
			} else if (individual.fitness < best.fitness) {
				best = individual;
			}
		}
		return best;
	}
	
	/**
	 * Returns a copy of the most fit individual in the population
	 * 
	 */
	private Individual setBest() {
		best = null;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}
	
	private void printMeanFitness() {
		double mean = 0;
		for (Individual i : population) {
			mean += i.fitness;
		}
		
		mean = mean / population.size();
		
		System.out.println("Mean fitness is: " + mean);
	}
	
	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}

	/**
	 * The activation function. tanh()
	 * (non-Javadoc)
	 * @see model.NeuralNetwork#activationFunction(double)
	 */
	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
	
	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		evaluateIndividuals(population);
		return population;
	}
	
}
