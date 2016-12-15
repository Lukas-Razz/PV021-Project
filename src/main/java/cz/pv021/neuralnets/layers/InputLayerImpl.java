package cz.pv021.neuralnets.layers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implementation of InputLayer.
 * 
 * @author  Lukáš Daubner
 * @since   2016-10-30
 * @version 2016-12-15
 */
public class InputLayerImpl implements InputLayer {
    final Logger logger = LoggerFactory.getLogger (InputLayerImpl.class);
    private final int id;
    private LayerWithInput outputLayer;
    private final int numberOfUnits;
    private double[] output;
    
    public InputLayerImpl (int id, int numberOfUnits) {
        this.id = id;
        this.numberOfUnits = numberOfUnits;
        this.output = new double[numberOfUnits];
    }
    
    @Override
    public int getId () {
        return id;
    }
    
    @Override
    public LayerWithInput getOutputLayer () {
        return outputLayer;
    }
    
    @Override
    public int getNumberOfUnits () {
        return numberOfUnits;
    }

    @Override
    public double[] getOutput () {
        return output;
    }
    
    @Override
    public InputLayerImpl makeCopy (int id) {
        return (new InputLayerImpl (id, numberOfUnits));
    }
    
    @Override
    public void setInput (double[] input) {
        this.output = input;
    }

    @Override
    public void setOutputLayer (LayerWithInput layer) {
        this.outputLayer = layer;
    }
    
    @Override
    public String toString () {
        return ("InputLayerImpl {id=" + id + "}");
    }
}
