package cz.pv021.neuralnets.layers;

/**
 * Interface of input layer of neural network.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-21
 */
public interface InputLayer extends LayerWithOutput {
    public void setInput (double[] input);   
}
