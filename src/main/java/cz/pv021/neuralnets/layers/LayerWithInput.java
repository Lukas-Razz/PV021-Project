package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;

/**
 * Any layer with preceding layer, i.e. any except the input layer.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-11-08
 * @version 2016-12-11
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
    
    public List<LayerParameters> getErrors ();
    
    public double[] getInnerPotentialGradient ();
    
    public LayerParameters getParameters ();
    
    public LayerWithOutput getUpperLayer ();
    
    public void initializeWeights (long seed);
    
    public void resetGradients();
    
    public void setParameters (LayerParameters parameters);
    
    /**
     * Nastav předchozí vrstvu.
     * 
     * @param layer 
     */
    public void setUpperLayer (LayerWithOutput layer);
}
