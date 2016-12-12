package cz.pv021.neuralnets.layers;

/**
 * Any layer with following layer, i.e. any except the output layer.
 * 
 * @author  Josef Plch
 * @since   2016-11-08
 * @version 2016-12-11
 */
public interface LayerWithOutput extends Layer {
    /**
     * Get the next (output) layer.
     * 
     * @return Output layer.
     */
    public LayerWithInput getLowerLayer ();
    
    /**
     * Set the next (output) layer.
     * 
     * @param nextLayer Output layer.
     */
    public void setLowerLayer (LayerWithInput nextLayer);
}
