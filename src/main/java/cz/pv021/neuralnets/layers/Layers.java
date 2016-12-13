package cz.pv021.neuralnets.layers;

/**
 * Helper class for layers.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-13
 */
public abstract class Layers {
    /**
     * Connect two layers.
     * 
     * @param outputLayer Output layer (i.e. previous).
     * @param inputLayer  Lower layer (i.e. next).
     */
    public static void connect (LayerWithOutput outputLayer, LayerWithInput inputLayer) {
        inputLayer.setInputLayer (outputLayer);
        outputLayer.setOutputLayer (inputLayer);
    }
}
