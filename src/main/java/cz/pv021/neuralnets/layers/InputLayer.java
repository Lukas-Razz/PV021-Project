package cz.pv021.neuralnets.layers;

/**
 * Interface of input layer of neural network.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-13
 */
public interface InputLayer <A> extends LayerWithOutput {
    public void setInput (double[] input);
    
    public void setInputObject (A input);
}
