package cz.pv021.neuralnets.layers;

/**
 * Interface of input layer of neural network.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-15
 */
public interface InputLayer extends LayerWithOutput {
    @Override
    public InputLayer makeCopy (int id);
    
    public void setInput (double[] input);
}
