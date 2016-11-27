package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.List;

/**
 * Any layer with preceding layer, i.e. any except the input layer.
 * 
 * @author  Josef Plch
 * @since   2016-11-08
 * @version 2016-11-27
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
    
    public LayerParameters getParameters();
    public void setParameters(LayerParameters parameters);
    
    public double[] getInnerPotentialGradient();
    
    public List<LayerParameters> getErrors();
    
    /**
     * Nastav předchozí vrstvu.
     * 
     * @param layer 
     */
    public void setUpperLayer (LayerWithOutput layer);
}
