package cz.pv021.neuralnets.layers;

import cz.pv021.neuralnets.functions.HiddenFunction;

/**
 * Interface of hidden (middle) neural network layer. It is just a unification
 * of LayerWithInput and LayerWithOutput interfaces.
 * 
 * @author  Josef Plch
 * @since   2016-11-07
 * @version 2016-12-16
 */
public interface HiddenLayer extends LayerWithInput, LayerWithOutput {
    public HiddenFunction getActivationFunction ();
    
    @Override
    public HiddenLayer deepCopy (int id);
}
