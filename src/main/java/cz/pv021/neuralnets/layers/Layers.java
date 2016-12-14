package cz.pv021.neuralnets.layers;

import java.util.Collections;
import java.util.List;

/**
 * Helper class for layers.
 * 
 * @author  Lukáš Daubner, Josef Plch
 * @since   2016-10-30
 * @version 2016-12-13
 */
public abstract class Layers {
    /**
     * Connect two layers.
     * 
     * @param inputLayer  Output layer (i.e. previous).
     * @param outputLayer Lower layer (i.e. next).
     */
    public static void connect (LayerWithOutput inputLayer, LayerWithInput outputLayer) {
        connect (Collections.singletonList (inputLayer), outputLayer);
    }
    
    /**
     * Connect several input layers to a single output layer.
     * 
     * @param inputLayers Output layers (i.e. previous).
     * @param outputLayer Lower layer (i.e. next).
     */
    public static void connect (List <LayerWithOutput> inputLayers, LayerWithInput outputLayer) {
        outputLayer.setInputLayers (inputLayers);
        for (LayerWithOutput inputLayer : inputLayers) {
            inputLayer.setOutputLayer (outputLayer);
        }
    }
}
