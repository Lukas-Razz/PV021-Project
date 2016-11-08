package cz.pv021.neuralnets.layers;

/**
 * Any layer with preceding layer, i.e. any except the input layer.
 * 
 * @author  Josef Plch
 * @since   2016-11-08
 * @version 2016-11-08
 */
public interface LayerWithInput extends Layer {
    /**
     * Zpětná propagace.
     */ 
    public void backwardPass ();

    /**
     * Inference.
     */
    public void forwardPass ();
    
    public void initializeWeights (long seed);
    
    /**
     * Nastav předchozí vrstvu.
     * 
     * @param layer 
     */
    public void setUpperLayer (LayerWithOutput layer);
}
