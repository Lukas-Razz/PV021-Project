package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.error.Loss;
import cz.pv021.neuralnets.functions.OutputFunction;
import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of output layer.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-15
 */
public class OutputLayerImpl implements OutputLayer {
    final Logger logger = LoggerFactory.getLogger(OutputLayerImpl.class);
    
    private final int id;
    private Loss loss;
    private final OutputFunction outputFunction;
    private InputMerger inputMerger;
    private double[] bias;
    private final List<double[]> biasErrors;
    // Error with respect to inner potentials = inner potential gradient.
    private final double[] err_wrt_innerP;
    private double expectedOutput;
    private final double[] innerPotentials;
    private final int numberOfUnits;
    private double[] output;
    private final List<double[][]> weightErrors;
    private double[][] weights;
    
    public OutputLayerImpl (int id, int numberOfUnits, OutputFunction outputFunction) {
        this.id = id;
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
        this.bias = new double[numberOfUnits];
        this.outputFunction = outputFunction;
        
        this.innerPotentials = new double[numberOfUnits];
        this.err_wrt_innerP = new double[numberOfUnits];
        
        weightErrors = new ArrayList<>();
        biasErrors = new ArrayList<>();
    }
    
    @Override
    public void backwardPass () {
        String logPrefix = "OL #" + id + " / backwardPass";
        double[][] err_wrt_weight = new double[numberOfUnits][inputMerger.getNumberOfUnits()];
        double[] preSoftmax = outputFunction.derivative(innerPotentials);
        
        for (int i = 0; i < numberOfUnits; i++) {
            double error = i == expectedOutput ? loss.derivative(output[i], 1) : loss.derivative(output[i], 0); //pro klasifikaci
            err_wrt_innerP[i] = error * preSoftmax[i];
            
            for (int j=0; j<inputMerger.getNumberOfUnits(); j++) {
                err_wrt_weight[i][j] = err_wrt_innerP[i] * inputMerger.getOutput()[j]; // innerPotential of neuron "i" * output of neuron "j"
            }
        }
        // System.out.println (logPrefix + "gradient = " + Arrays.toString (this.getInnerPotentialGradient ()));
        biasErrors.add(err_wrt_innerP); // err_wrt_innerP = err_wrt_bias
        weightErrors.add(err_wrt_weight);
    }
    
    @Override
    // Inner potential is remembered for backward pass.
    public void forwardPass () {
        String logPrefix = "OL #" + id + " / forwardPass: ";
        double[] input = inputMerger.getOutput ();
        // System.out.println (logPrefix + "input = " + Arrays.toString (input));
        
        for (int n = 0; n < numberOfUnits; n++) {
            innerPotentials[n] = bias[n];
            for (int i = 0; i < weights[n].length; i++) {
                innerPotentials[n] += input[i] * weights[n][i];
            }
        }
        this.output = outputFunction.apply (innerPotentials);
    }
    
    @Override
    public List <LayerParameters> getErrors () {
        List<LayerParameters> errors = new ArrayList<>();
        for(int i=0; i<weightErrors.size(); i++) {
            errors.add(new LayerParameters(weightErrors.get(i), biasErrors.get(i), id));
        }
        return errors;
    }
    
    @Override
    public int getId () {
        return id;
    }
    
    @Override
    public double[] getInnerPotentials () {
        return innerPotentials;
    }
    
    @Override
    public double[] getInnerPotentialGradient () {
        return err_wrt_innerP;
    }

    @Override
    public List <LayerWithOutput> getInputLayers () {
        return inputMerger.getLayers ();
    }
    
    @Override
    public int getInputSize () {
        return inputMerger.getNumberOfUnits ();
    }
    
    @Override
    public int getNumberOfUnits () {
        return numberOfUnits;
    }
    
    @Override
    public double[] getOutput () {
        return output;
    }

    @Override
    public LayerParameters getParameters () {
        return new LayerParameters(weights, bias, id);
    }
    
    @Override
    public double[][] getWeights () {
        return weights;
    }
    
    @Override
    public void initializeWeights (long seed) {
        Random r = new Random (seed);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = Math.abs(r.nextGaussian ()); //Softmax nerad minus
            }
        }
        for (int i = 0; i < bias.length; i++) {
            bias[i] = 0;
        }
    }
    
    @Override
    public void resetGradients () {
        biasErrors.clear();
        weightErrors.clear();
    }
    
    @Override
    public void resetWeights () {
        this.weights = new double[numberOfUnits][inputMerger.getNumberOfUnits ()];
    }
    
    @Override
    public void setExpectedOutput (double expectedOutput) {
        this.expectedOutput = expectedOutput;
    }

    @Override
    public void setInputLayers (List <LayerWithOutput> layers) {
        this.inputMerger = new InputMerger (layers);
    }

    @Override
    public void setLoss (Loss loss) {
        this.loss = loss;
    }
    
    @Override
    public OutputFunction getActivationFunction() {
        return outputFunction;
    }
    
    @Override
    public void setParameters (LayerParameters parameters) {
        weights = parameters.getWeights();
        bias = parameters.getBias();
    }
    
    @Override
    public String toString () {
        return ("OutputLayerImpl {id=" + id + "}");
    }
}
