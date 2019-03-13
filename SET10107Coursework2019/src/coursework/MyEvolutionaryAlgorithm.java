package coursework;

import java.util.ArrayList;

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
		
		// Evaluate and select parents
		
		/**
		 *  Reproduction which involves a new individual
		 *  with gene combination and mutation.
		 *  Crossover, Mutation
		 */
		
		// Replacement in population? Replace worst? dynamic?
		
	}

	@Override
	public double activationFunction(double x) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	/**
	 * Pre-written function from the coursework.
	 */
	
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
	
	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 */
	private void evaluateIndividuals(ArrayList<Individual> individuals) {
		for (Individual individual : individuals) {
			individual.fitness = Fitness.evaluate(individual, this);
		}
	}

}
