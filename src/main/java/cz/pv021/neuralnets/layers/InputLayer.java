/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cz.pv021.neuralnets.layers;

/**
 *
 * @author lukas
 */
public class InputLayer implements IInputLayer {
    
    private ILayer lowerLayer; //Vystupni
    
    private final int numberOfUnits;
    
    private double[] output;
    
    public InputLayer(int numberOfUnits) {
        this.numberOfUnits = numberOfUnits;
        output = new double[numberOfUnits];
    }
    

    @Override
    public void forwardPass() {
    }

    @Override
    public void backwardPass() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void SetUpperLayer(ILayer layer) {
        throw new IllegalStateException("Input layer does not have an upper layer");
    }

    @Override
    public void SetLowerLayer(ILayer layer) {
        lowerLayer = layer;
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
    public void clampInput(double[] input) {
        output = input;
    }

    @Override
    public void initializeWeights(long seed) {
    }
    
    
}
