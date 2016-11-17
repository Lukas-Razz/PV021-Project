package cz.pv021.neuralnets.utils;

/**
 * Parameters of one layer.
 * 
 * @author Lukas Daubner
 * @since   2016-11-17
 * @version 2016-11-17
 */
public class LayerParameters {
    
    private double[][] weights;
    private double[] bias;

    public LayerParameters(double[][] weights, double[] bias) {
        this.weights = weights;
        this.bias = bias;
    }

    public double[][] getWeights() {
        return weights;
    }

    public void setWeights(double[][] weights) {
        this.weights = weights;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }
    
    
    
}
