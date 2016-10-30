/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.ActivationFunction;
import cz.pv021.neuralnets.functions.OutputFunction;
import java.util.Random;

/**
 *
 * @author lukas
 */
public class OutputLayer implements ILayer {

    private ILayer upperLayer; //Vstupni
    
    private final int numberOfUnits;
    
    private double[][] weights;
    private double[] output;
    
    private OutputFunction outputFunction;
    
    public OutputLayer(int numberOfUnits, OutputFunction outputFunction) {
        this.numberOfUnits = numberOfUnits;
        output = new double[numberOfUnits];
        this.outputFunction = outputFunction;
    }
    
    @Override
    public void forwardPass() {
        double[] input = upperLayer.GetOutput();
        double[] innerPotencials = new double[numberOfUnits];
        
        for(int n=0; n<numberOfUnits; n++)
        {
            for(int i=0; i<weights[n].length; i++)
            {
                innerPotencials[n] += input[i] * weights[n][i];
            }
        }
        
        output = outputFunction.Function(innerPotencials);
    }

    @Override
    public void backwardPass() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void SetUpperLayer(ILayer layer) {
        upperLayer = layer;
        weights = new double[numberOfUnits][layer.GetNumberOfUnits()];
    }

    @Override
    public void SetLowerLayer(ILayer layer) {
        throw new IllegalStateException("Output layer does not have an lower layer");
    }

    @Override
    public int GetNumberOfUnits() {
        return numberOfUnits;
    }

    @Override
    public double[] GetOutput() {
        return output;
    }

    @Override
    public void initializeWeights(long seed) {
        Random r = new Random(seed);
        for(int i=0; i<weights.length; i++)
        {
            for(int j=0; j<weights[i].length; j++)
            {
                weights[i][j] = r.nextGaussian();
            }
        }
    }
    
}
