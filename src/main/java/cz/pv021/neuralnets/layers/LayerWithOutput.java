package cz.pv021.neuralnets.layers;

/**
 * Any layer with following layer, i.e. any except the output layer.
 * 
 * @author  Josef Plch
 * @since   2016-11-08
 * @version 2016-12-13
 */
public interface LayerWithOutput extends Layer {
    /**
     * Get the output (i.e. next) layer.
     * 
     * @return Output layer.
     */
    public LayerWithInput getOutputLayer ();
    
    /**
     * Set the output (i.e. next) layer.
     * 
     * @param outputLayer Output layer.
     */
    public void setOutputLayer (LayerWithInput outputLayer);
}
