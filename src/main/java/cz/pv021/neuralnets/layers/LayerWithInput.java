package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.utils.LayerParameters;
import java.util.Collections;
import java.util.List;

/**
 * Any layer with preceding layer, i.e. any except the input layer.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-11-08
 * @version 2016-12-16
 */
public interface LayerWithInput extends Layer {
    /**
     * Backpropagation.
     */ 
    public void backwardPass ();

    /**
     * Inference.
     */
    public void forwardPass ();
    
    public double[] getBias ();
    
    public List <LayerParameters> getErrors ();
    
    public double[] getInnerPotentials ();    
    
    public double[] getInnerPotentialGradient ();
    
    public List <LayerWithOutput> getInputLayers ();
    
    /**
     * Get the total number of neurons in the input layers.
     * 
     * @return Input size.
     */
    public int getInputSize ();
    
    public LayerParameters getParameters ();
    
    public double[][] getWeights ();
    
    public void initializeWeights (long seed);
    
    public void resetGradients ();
    
    public void resetWeights ();
    
    public void setBias (double[] bias);
    
    public void setInnerPotentials (double[] innerPotentials);    
    
    public void setInnerPotentialGradient (double[] innerPotentialGradient);
    
    public default void setInputLayer (LayerWithOutput layer) {
        this.setInputLayers (Collections.singletonList (layer));
    }
    
    /**
     * Set the input (i.e. previous) layers.
     * 
     * @param layers
     */
    public void setInputLayers (List <LayerWithOutput> layers);
    
    public void setParameters (LayerParameters parameters);
    
    public void setWeights (double[][] weights);    
}
