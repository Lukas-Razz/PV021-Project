package cz.pv021.neuralnets.layers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of InputLayer.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-11-21
 */
public class InputLayerImpl implements InputLayer {
    final Logger logger = LoggerFactory.getLogger (InputLayerImpl.class);
    private LayerWithInput lowerLayer; // Vystupni
    private final int numberOfUnits;
    private double[] output;
    
    public InputLayerImpl (int numberOfUnits) {
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
    }

    public int getNumberOfUnits () {
        return numberOfUnits;
    }

    @Override
    public double[] getOutput () {
        return output;
    }
    
    @Override
    public void setInput (double[] input) {
        this.output = input;
    }

    @Override
    public void setLowerLayer (LayerWithInput layer) {
        this.lowerLayer = layer;
    }
}
