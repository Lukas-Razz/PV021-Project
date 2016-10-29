/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.ActivationFunction;

/**
 *
 * @author lukas
 */
public class FullyConnectedLayer implements ILayer
{
    private ILayer upperLayer; //Vstupni
    private ILayer lowerLayer; //Vystupni
    
    private final int numberOfUnits;
    
    private double[][] weights;
    private double[] output;
    
    private ActivationFunction activationFunction;
    
    public FullyConnectedLayer(int numberOfUnits) {
        this.numberOfUnits = numberOfUnits;
        output = new double[numberOfUnits];
    }

    @Override
    public void forwardPass() {
        double[] input = upperLayer.GetOutput();
        for(int n=0; n<numberOfUnits; n++)
        {
            int innerPotencial = 0;
            for(int i=0; i<weights[n].length; i++)
            {
                innerPotencial += input[i] * weights[n][i];
            }
            output[n] = activationFunction.Function(innerPotencial);
        }
    }

    @Override
    public void backwardPass() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void SetUpperLayer(ILayer layer) {
        this.upperLayer = layer;
        weights = new double[numberOfUnits][layer.GetNumberOfUnits()];
    }

    @Override
    public void SetLowerLayer(ILayer layer) {
        this.lowerLayer = layer;
    }

    @Override
    public int GetNumberOfUnits() {
        return numberOfUnits;
    }

    @Override
    public double[] GetOutput() {
        return output;
    }
}
