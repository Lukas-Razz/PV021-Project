package cz.pv021.neuralnets.utils;

/**
 * Parameters of one layer.
 * 
 * @author Lukas Daubner
 * @since   2016-11-17
 * @version 2016-12-10
 */
public class LayerParameters {
    
    private double[][] weights;
    private double[] bias;
    
    private int layerId;

    public LayerParameters(double[][] weights, double[] bias) {
        this.weights = weights;
        this.bias = bias;
    }
    
    public LayerParameters(double[][] weights, double[] bias, int layerId) {
        this.weights = weights;
        this.bias = bias;
        this.layerId = layerId;
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

    public int getLayerId() {
        return layerId;
    }

    public void setLayerId(int layerId) {
        this.layerId = layerId;
    }
}
